"""See _CONFIGS for the list of available configs."""
from chunktrust.paths import resolve_legacy_path as _ct_path

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=False)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=False)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions", )

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False


class GroupFactory(Protocol):

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=False)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len), ),
                ], )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(model_config.max_token_len), ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=False)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=False)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=False)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=False)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(default=_transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": {
                "cam_high": "observation.images.top"
            },
            "state": "observation.state",
            "actions": "action",
            "actions_is_pad": "action_is_pad",
        })
    ]))
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action", )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        input_transforms = []
        if getattr(model_config, "use_qha", False):
            history_len = int(getattr(model_config, "qha_history_max_length", 0))
            if history_len > 0:
                input_transforms.append(
                    _transforms.SplitHistoryActions(
                        history_len=history_len,
                        future_len=model_config.action_horizon,
                    ))

        input_transforms.append(
            aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi))

        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=False)
class MultiDataConfig:
    """Container holding per-task DataConfig objects for multi-task joint training.

    Detected by ``create_data_loader()`` to switch from single-dataset to
    ConcatDataset mode.  Each sub-config carries its own norm_stats, so
    per-task normalization is preserved.
    """

    sub_configs: list[DataConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=False)
class MultiLeRobotAlohaDataConfig(DataConfigFactory):
    """Data config factory that creates a ConcatDataset over multiple LeRobot repos.

    Each repo gets its own norm_stats (loaded from the respective asset directory).
    The observation/action spaces must be identical across all repos.
    """

    repo_id: str = ""  # overridden; use repo_ids instead
    repo_ids: list[str] = dataclasses.field(default_factory=list)
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = False

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(inputs=[
            _transforms.RepackTransform({
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "actions_is_pad": "action_is_pad",
                "prompt": "prompt",
            })
        ])
    )
    action_sequence_keys: Sequence[str] = ("action",)

    def normalized_repo_ids(self) -> list[str]:
        repo_ids: list[str] = []

        # Handle tyro CLI quirks: a JSON list may arrive as either a str
        # or as a one-element list containing the JSON str.
        raw = self.repo_ids
        if isinstance(raw, str):
            import json as _json
            try:
                repo_ids = _json.loads(raw)
            except (_json.JSONDecodeError, TypeError):
                repo_ids = [raw]
        elif isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], str) and raw[0].startswith("["):
                # tyro may wrap the raw JSON string in a list.
                import json as _json
                try:
                    repo_ids = _json.loads(raw[0])
                except (_json.JSONDecodeError, TypeError):
                    repo_ids = raw
            else:
                repo_ids = raw

        if not isinstance(repo_ids, list) or len(repo_ids) == 0:
            raise ValueError(f"repo_ids must be a non-empty list of repo id strings. Got: {self.repo_ids!r}")
        return [str(repo_id) for repo_id in repo_ids]

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> MultiDataConfig:
        repo_ids = self.normalized_repo_ids()
        sub_configs = []
        for repo_id in repo_ids:
            input_transforms = []
            if getattr(model_config, "use_qha", False):
                history_len = int(getattr(model_config, "qha_history_max_length", 0))
                if history_len > 0:
                    input_transforms.append(
                        _transforms.SplitHistoryActions(
                            history_len=history_len,
                            future_len=model_config.action_horizon,
                        ))

            input_transforms.append(
                aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi))

            data_transforms = _transforms.Group(
                inputs=input_transforms,
                outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
            )
            if self.use_delta_joint_actions:
                delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )

            model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

            asset_id = repo_id
            norm_stats = self._load_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

            sub_configs.append(DataConfig(
                repo_id=repo_id,
                asset_id=asset_id,
                norm_stats=norm_stats,
                repack_transforms=self.repack_transforms,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=self.action_sequence_keys,
                use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
                prompt_from_task=True,
                local_files_only=True,
            ))
        return MultiDataConfig(sub_configs=sub_configs)


@dataclasses.dataclass(frozen=False)
class LeRobotLiberoDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(inputs=[
            _transforms.RepackTransform({
                "observation/image": "image",
                "observation/wrist_image": "wrist_image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            })
        ])

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=False)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints/"
    # Optional directory tag under checkpoint_base_dir used instead of the config name.
    # This lets experiments share the same training config/assets while saving checkpoints
    # under a separate root such as "<config>_qha".
    checkpoint_root_tag: str = ""

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Torch DataLoader prefetch factor per worker (only when num_workers > 0).
    dataloader_prefetch_factor: int = 2
    # Whether to pin host memory in DataLoader workers (only when num_workers > 0).
    dataloader_pin_memory: bool = True
    # Number of host batches to prefetch from the torch-backed iterator on a background thread.
    # Set to 0 to disable host-side thread prefetch.
    host_prefetch: int = 8
    # If true, load training samples from the preprocessed cache instead of running the
    # raw sample-level transform pipeline online. Missing or mismatched caches are fatal.
    use_preprocessed_cache: bool = False
    # Optional root directory for preprocessed caches. When empty, defaults to:
    # <assets_base_dir>/preprocessed/<train_config_name>
    preprocessed_cache_root: str = ""
    # Number of batches to prefetch from the data iterator on a background thread.
    # This is the device staging queue depth after host batches are converted with jax.device_put.
    # Set to 0 to disable device staging prefetch.
    device_prefetch: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # Compute expensive horizon alignment diagnostics every N steps.
    # 1 means compute on every step.
    horizon_metrics_interval: int = 10
    # Compute and device-sync parameter norm every N steps.
    # <=0 falls back to log_interval.
    param_norm_interval: int = 500
    # Optional profiler output directory. Empty disables trace capture.
    profile_dir: str = ""
    # First step at which profiler capture starts.
    profile_start_step: int = 0
    # Number of steps to capture. 0 disables trace capture.
    profile_num_steps: int = 0
    # Whether to save a device memory profile after the trace window finishes.
    profile_capture_memory: bool = False
    # If true, precompile train step functions before entering the main loop.
    eager_compile_step_fns: bool = True
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = None
    # If true, save only qha.* inference params at each checkpoint step instead of full train state.
    # This is intended for joint QHA-only training where checkpoints are later overlaid onto a base model.
    save_qha_only_checkpoints: bool = False
    # Optional JSON manifest that maps each joint-training repo_id to a teacher checkpoint.
    teacher_checkpoint_manifest: str = ""
    # Empty/off by default. Set to "online_per_task" to compute QHA teacher labels from per-task checkpoints.
    teacher_runtime_mode: str = ""
    # Base directory for routed teacher checkpoints. Empty falls back to checkpoint_base_dir.
    teacher_checkpoint_base_dir: str = ""
    # Compatibility field for teacher manifests that omit pi0_step. Direct model-space teacher inference does not
    # currently use this value, but keeping it in config preserves the eval/dump convention.
    teacher_pi0_step_default: int = 50
    # Maximum number of task-routed frozen pi0 teacher/backbone runtimes kept resident at once.
    # Defaults to 1 for memory safety in online_per_task training.
    teacher_max_loaded: int = 1
    teacher_preload_on_startup: bool = False
    # Teacher execution backend. "local" keeps the existing single-process path;
    # "gpu_workers" launches one teacher worker process per listed GPU.
    teacher_execution_backend: str = "local"
    teacher_worker_devices: str = ""
    teacher_worker_startup_timeout_s: int = 900
    teacher_worker_request_timeout_s: int = 600
    # Number of host batches whose online teacher outputs are prepared ahead of the train loop.
    # v1 supports this only for the gpu_workers backend.
    teacher_prefetch: int = 0

    # Optional offline teacher manifest (.npz from eval_dataset_horizon_dump_test.py) used to
    # rebalance horizon-head training samples with a weighted sampler.
    horizon_resample_manifest: str = ""
    # Which teacher statistic from the manifest to use for binning.
    # Supported: teacher_argmax_k | teacher_expected_horizon
    horizon_resample_key: str = "teacher_argmax_k"
    # Comma-separated upper bounds for coarse bins. Example "2,4,8,16" gives:
    # <=2, 3-4, 5-8, 9-16, >16
    horizon_resample_bins: str = "2,4,8,16"
    # Inverse-frequency exponent. 0 disables reweighting, 1 means full inverse count.
    horizon_resample_gamma: float = 0.5
    # Cap the rarest-bin weight relative to the most common bin.
    horizon_resample_max_weight_ratio: float = 10.0
    # If true, samples with teacher_expected_horizon <= threshold are given zero sampling weight.
    horizon_resample_filter_expected_horizon_apply: bool = False
    # Threshold used by the expected-horizon hard filter.
    horizon_resample_filter_expected_horizon_min: float = 0.0
    # Whether weighted resampling uses replacement.
    horizon_resample_replacement: bool = True
    # Number of samples drawn per dataloader epoch. <=0 means len(dataset).
    horizon_resample_num_samples: int = 0
    # Multi-task batch construction. "mixed" keeps the existing weighted sample-level sampler.
    # "task_homogeneous" draws one task per batch. "task_balanced_mixed" draws an approximately
    # equal number of samples per task in every batch.
    multi_task_batch_mode: str = "mixed"
    # In task_homogeneous mode, keep the same task for this many consecutive batches before switching.
    # Values >1 amortize online teacher checkpoint loads and improve GPU utilization.
    multi_task_batches_per_task_block: int = 1

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        root_name = self.checkpoint_root_tag or self.name
        return (pathlib.Path(self.checkpoint_base_dir) / root_name / self.exp_name).resolve()

    @property
    def preprocessed_cache_root_dir(self) -> pathlib.Path:
        """Get the root directory used for preprocessed sample caches."""
        if self.preprocessed_cache_root:
            return pathlib.Path(self.preprocessed_cache_root).resolve()
        return (pathlib.Path(self.assets_base_dir) / "preprocessed" / self.name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.horizon_metrics_interval <= 0:
            raise ValueError("horizon_metrics_interval must be a positive integer.")
        if self.param_norm_interval < 0:
            raise ValueError("param_norm_interval must be >= 0.")
        if self.profile_start_step < 0:
            raise ValueError("profile_start_step must be >= 0.")
        if self.profile_num_steps < 0:
            raise ValueError("profile_num_steps must be >= 0.")
        resample_key = self.horizon_resample_key.strip().lower()
        if resample_key not in ("teacher_argmax_k", "teacher_expected_horizon"):
            raise ValueError(
                "horizon_resample_key must be one of: teacher_argmax_k, teacher_expected_horizon. "
                f"Got {self.horizon_resample_key}."
            )
        self.horizon_resample_key = resample_key
        raw_bins = str(self.horizon_resample_bins).strip()
        if raw_bins:
            try:
                parsed_bins = [float(x) for x in raw_bins.split(",") if x.strip()]
            except ValueError as exc:
                raise ValueError(
                    "horizon_resample_bins must be a comma-separated list of numeric upper bounds, "
                    f"got {self.horizon_resample_bins!r}."
                ) from exc
            if not parsed_bins:
                raise ValueError("horizon_resample_bins must not be empty when provided.")
            if any(x <= 0 for x in parsed_bins):
                raise ValueError("horizon_resample_bins values must be > 0.")
            if any(b <= a for a, b in zip(parsed_bins[:-1], parsed_bins[1:])):
                raise ValueError("horizon_resample_bins must be strictly increasing.")
        if self.horizon_resample_gamma < 0:
            raise ValueError("horizon_resample_gamma must be >= 0.")
        if self.horizon_resample_max_weight_ratio < 1.0:
            raise ValueError("horizon_resample_max_weight_ratio must be >= 1.")
        if self.horizon_resample_filter_expected_horizon_min < 0:
            raise ValueError("horizon_resample_filter_expected_horizon_min must be >= 0.")
        if self.horizon_resample_num_samples < 0:
            raise ValueError("horizon_resample_num_samples must be >= 0.")
        teacher_runtime_mode = str(self.teacher_runtime_mode).strip().lower()
        if teacher_runtime_mode not in ("", "off", "online_per_task"):
            raise ValueError(
                "teacher_runtime_mode must be empty/off or online_per_task. "
                f"Got {self.teacher_runtime_mode}."
            )
        self.teacher_runtime_mode = "" if teacher_runtime_mode == "off" else teacher_runtime_mode
        if self.teacher_pi0_step_default <= 0:
            raise ValueError("teacher_pi0_step_default must be positive.")
        if self.teacher_max_loaded <= 0:
            raise ValueError("teacher_max_loaded must be positive.")
        if self.teacher_preload_on_startup and self.teacher_runtime_mode != "online_per_task":
            raise ValueError("teacher_preload_on_startup requires teacher_runtime_mode=online_per_task.")
        teacher_execution_backend = str(self.teacher_execution_backend).strip().lower()
        if teacher_execution_backend not in ("local", "gpu_workers"):
            raise ValueError("teacher_execution_backend must be local or gpu_workers.")
        if teacher_execution_backend == "gpu_workers" and self.teacher_runtime_mode != "online_per_task":
            raise ValueError("teacher_execution_backend=gpu_workers requires teacher_runtime_mode=online_per_task.")
        self.teacher_execution_backend = teacher_execution_backend
        if self.teacher_worker_startup_timeout_s <= 0:
            raise ValueError("teacher_worker_startup_timeout_s must be positive.")
        if self.teacher_worker_request_timeout_s <= 0:
            raise ValueError("teacher_worker_request_timeout_s must be positive.")
        if self.teacher_prefetch < 0:
            raise ValueError("teacher_prefetch must be >= 0.")
        if self.teacher_prefetch > 0:
            if self.teacher_runtime_mode != "online_per_task":
                raise ValueError("teacher_prefetch > 0 requires teacher_runtime_mode=online_per_task.")
            if teacher_execution_backend != "gpu_workers":
                raise ValueError("teacher_prefetch > 0 is only supported with teacher_execution_backend=gpu_workers.")
        multi_task_batch_mode = str(self.multi_task_batch_mode).strip().lower()
        if multi_task_batch_mode not in ("mixed", "task_homogeneous", "task_balanced_mixed"):
            raise ValueError("multi_task_batch_mode must be mixed, task_homogeneous, or task_balanced_mixed.")
        self.multi_task_batch_mode = multi_task_batch_mode


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    ###
    ### finetune config for robotwin
    ###
    # pi0_base by lora
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora",
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="handover_mic_aloha-agilex_clean_50_repo",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "actions_is_pad": "action_is_pad",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=64,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
    # pi0_fast_base by lora
    TrainConfig(
        name="pi0_fast_aloha_robotwin_lora",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="blocks_ranking_rgb_aloha-agilex_clean_50_repo",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "actions_is_pad": "action_is_pad",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig(
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=2,  # refer line 359
    ),
    # pi0_base by full
    TrainConfig(
        name="pi0_base_aloha_robotwin_full",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="blocks_ranking_rgb_aloha-agilex_clean_50_repo",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "actions_is_pad": "action_is_pad",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0.Pi0Config().get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=4,  # refer line 359
    ),
    # pi0_fast_base by full
    TrainConfig(
        name="pi0_fast_aloha_robotwin_full",
        model=pi0_fast.Pi0FASTConfig(),
        data=LeRobotAlohaDataConfig(
            repo_id="blocks_ranking_rgb_aloha-agilex_clean_50_repo",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "actions_is_pad": "action_is_pad",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig().get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
    ###
    ### joint QHA multi-task training config
    ###
    # Joint QHA training: one QHA adapter trained across all 8 RoboTwin tasks.
    # Backbone + LoRA loaded from pi0_base; QHA initialized randomly.
    # Only qha.* parameters are trainable (qha_only mode).
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora_qha_joint",
        model=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_qha=True,
            qha_train_mode="qha_only",
            qha_candidate_mode="dense_full",
            qha_history_max_length=64,
        ),
        data=MultiLeRobotAlohaDataConfig(
            repo_ids=[
                "blocks_ranking_rgb_aloha-agilex_clean_50_repo",
                "handover_block_aloha-agilex_clean_50_repo",
                "handover_mic_aloha-agilex_clean_50_repo",
                "hanging_mug_aloha-agilex_clean_50_repo",
                "place_a2b_left_aloha-agilex_clean_50_repo",
                "place_bread_basket_aloha-agilex_clean_50_repo",
                "place_bread_skillet_aloha-agilex_clean_50_repo",
                "place_can_basket_aloha-agilex_clean_50_repo",
            ],
            assets=AssetsConfig(
                assets_dir=_ct_path('ROBOTWIN_ROOT', 'policy/pi0/assets/pi0_base_aloha_robotwin_lora'),
            ),
            adapt_to_pi=False,
        ),
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_qha=True,
            qha_train_mode="qha_only",
        ).get_freeze_filter(),
        batch_size=64,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "s3://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=r".*(lora|qha).*",
        ),
        num_train_steps=30000,
        fsdp_devices=1,
        preprocessed_cache_root=_ct_path('ROBOTWIN_ROOT', 'policy/pi0/assets/preprocessed/pi0_base_aloha_robotwin_lora'),
        teacher_checkpoint_base_dir=_ct_path('ROBOTWIN_ROOT', 'policy/pi0/checkpoints'),
        save_qha_only_checkpoints=True,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
