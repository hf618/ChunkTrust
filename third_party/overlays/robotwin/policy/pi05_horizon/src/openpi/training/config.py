"""See _CONFIGS for the list of available configs."""
from chunktrust.paths import resolve_legacy_path as _ct_path

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
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


@dataclasses.dataclass(frozen=True)
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
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
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

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
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


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

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
                    )
                )

        input_transforms.append(aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi))
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
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=False)
class MultiDataConfig:
    """Container for per-task DataConfig objects used by joint training."""

    sub_configs: list[DataConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class MultiLeRobotAlohaDataConfig(DataConfigFactory):
    """Create a multi-task ConcatDataset over several Aloha LeRobot repos."""

    repo_id: str = ""
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
        raw = self.repo_ids
        if isinstance(raw, str):
            import json as _json
            try:
                repo_ids = _json.loads(raw)
            except (_json.JSONDecodeError, TypeError):
                repo_ids = [raw]
        elif isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], str) and raw[0].startswith("["):
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
        sub_configs = []
        for repo_id in self.normalized_repo_ids():
            input_transforms = []
            if getattr(model_config, "use_qha", False):
                history_len = int(getattr(model_config, "qha_history_max_length", 0))
                if history_len > 0:
                    input_transforms.append(
                        _transforms.SplitHistoryActions(
                            history_len=history_len,
                            future_len=model_config.action_horizon,
                        )
                    )

            input_transforms.append(aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi))
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

            asset_id = repo_id
            sub_configs.append(
                DataConfig(
                    repo_id=repo_id,
                    asset_id=asset_id,
                    norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
                    action_sequence_keys=self.action_sequence_keys,
                    use_quantile_norm=model_config.model_type != ModelType.PI0,
                    prompt_from_task=True,
                    local_files_only=True,
                )
            )
        return MultiDataConfig(sub_configs=sub_configs)


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
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
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

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
    checkpoint_base_dir: str = "./checkpoints"
    # Optional directory tag under checkpoint_base_dir used instead of the config name.
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
    # Number of host batches to prefetch from the torch-backed iterator.
    host_prefetch: int = 8
    # If true, load training samples from the preprocessed cache.
    use_preprocessed_cache: bool = False
    # Optional root directory for preprocessed caches. When empty, defaults to:
    # <assets_base_dir>/preprocessed/<train_config_name>
    preprocessed_cache_root: str = ""
    # Number of batches to prefetch after jax.device_put.
    device_prefetch: int = 2
    # If set, only load the smallest prefix of episodes that covers this many frames.
    smoke_test_max_frames: int | None = None
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # Compute expensive horizon alignment diagnostics every N steps.
    horizon_metrics_interval: int = 10
    # Compute parameter norm every N steps for logging.
    param_norm_interval: int = 500
    # Optional profiler output directory. Empty disables trace capture.
    profile_dir: str = ""
    # First step at which profiler capture starts.
    profile_start_step: int = 0
    # Number of steps to capture. 0 disables trace capture.
    profile_num_steps: int = 0
    # Whether to save a device memory profile after the trace window finishes.
    profile_capture_memory: bool = False
    # Whether to precompile train step functions before entering the loop.
    eager_compile_step_fns: bool = True
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # If true, save only qha.* inference params at checkpoint steps.
    save_qha_only_checkpoints: bool = False
    # Optional JSON manifest mapping each joint-training repo_id to a routed pi05 teacher checkpoint.
    teacher_checkpoint_manifest: str = ""
    # Empty/off by default. Set to "online_per_task" for task-routed frozen pi05 supervision.
    teacher_runtime_mode: str = ""
    # Base directory for routed teacher checkpoints. Empty falls back to checkpoint_base_dir.
    teacher_checkpoint_base_dir: str = ""
    # Compatibility field for manifests that omit pi0_step.
    teacher_pi0_step_default: int = 50
    # Maximum number of routed pi05 backbones kept resident at once.
    teacher_max_loaded: int = 1
    # If true, eagerly load all routed teacher/backbone checkpoints at startup.
    teacher_preload_on_startup: bool = False
    # Teacher execution backend: local keeps current in-process behavior; gpu_workers uses one worker process per GPU.
    teacher_execution_backend: str = "local"
    # Comma-separated physical GPU ids for gpu_workers. Empty uses CUDA_VISIBLE_DEVICES or all visible JAX devices.
    teacher_worker_devices: str = ""
    # Seconds to wait for gpu_workers to preload assigned teacher checkpoints.
    teacher_worker_startup_timeout_s: int = 900
    # Seconds to wait for a gpu_workers batch request.
    teacher_worker_request_timeout_s: int = 600
    # Number of teacher-attached host batches to prefetch asynchronously. Only supported with gpu_workers.
    teacher_prefetch: int = 0
    # Multi-task batch construction mode: mixed, task_homogeneous, or task_balanced_mixed.
    multi_task_batch_mode: str = "mixed"
    # Consecutive batches per task block for task_homogeneous mode.
    multi_task_batches_per_task_block: int = 1

    # Optional offline teacher manifest used to rebalance horizon-head training samples.
    horizon_resample_manifest: str = ""
    # Which teacher statistic from the manifest to use for binning.
    horizon_resample_key: str = "teacher_argmax_k"
    # Comma-separated upper bounds for coarse bins.
    horizon_resample_bins: str = "2,4,8,16"
    # Inverse-frequency exponent.
    horizon_resample_gamma: float = 0.5
    # Cap the rarest-bin weight relative to the most common bin.
    horizon_resample_max_weight_ratio: float = 10.0
    # If true, samples with teacher_expected_horizon <= threshold are zero-weighted.
    horizon_resample_filter_expected_horizon_apply: bool = False
    # Threshold used by the expected-horizon hard filter.
    horizon_resample_filter_expected_horizon_min: float = 0.0
    # Whether weighted resampling uses replacement.
    horizon_resample_replacement: bool = True
    # Number of samples drawn per dataloader epoch. <=0 means len(dataset).
    horizon_resample_num_samples: int = 0

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
        teacher_runtime_mode = str(self.teacher_runtime_mode).strip().lower()
        if teacher_runtime_mode not in ("", "off", "online_per_task"):
            raise ValueError(
                "teacher_runtime_mode must be empty/off or online_per_task. "
                f"Got {self.teacher_runtime_mode}."
            )
        object.__setattr__(self, "teacher_runtime_mode", "" if teacher_runtime_mode == "off" else teacher_runtime_mode)
        if self.teacher_pi0_step_default <= 0:
            raise ValueError("teacher_pi0_step_default must be positive.")
        if self.teacher_max_loaded <= 0:
            raise ValueError("teacher_max_loaded must be positive.")
        teacher_execution_backend = str(self.teacher_execution_backend).strip().lower()
        if teacher_execution_backend not in ("local", "gpu_workers"):
            raise ValueError("teacher_execution_backend must be local or gpu_workers.")
        if teacher_execution_backend == "gpu_workers" and self.teacher_runtime_mode != "online_per_task":
            raise ValueError("teacher_execution_backend=gpu_workers requires teacher_runtime_mode=online_per_task.")
        object.__setattr__(self, "teacher_execution_backend", teacher_execution_backend)
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
        object.__setattr__(self, "multi_task_batch_mode", multi_task_batch_mode)
        if self.multi_task_batches_per_task_block <= 0:
            raise ValueError("multi_task_batches_per_task_block must be positive.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    ###
    ### finetune config for robotwin
    ###
    # pi05_base by full
    TrainConfig(
        name="pi05_base_aloha_robotwin_full",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="blocks_ranking_rgb_aloha-agilex_clean_50_repo",
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
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        save_interval=5000,
        keep_period=None,
        fsdp_devices=1,  # refer line 359
    ),
    # Inference-only companion for the formal clean50 multi-task checkpoint.
    # Norm stats and the resolved model snapshot are loaded from checkpoint assets.
    TrainConfig(
        name="pi05_base_aloha_robotwin_full_multitask50",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="robotwin_aloha_agilex_clean50_multitask50_global",
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
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        fsdp_devices=1,
        checkpoint_root_tag="pi05_base_aloha_robotwin_full_multitask50",
        wandb_enabled=False,
    ),
    # pi05_base by lora
    TrainConfig(
        name="pi05_base_aloha_robotwin_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="blocks_ranking_rgb_aloha-agilex_clean_50_repo",
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        batch_size=64,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=20_000),
        num_train_steps=20_000,
        save_interval=5000,
        keep_period=None,
        fsdp_devices=1,  # refer line 359
    ),
    # pi0_base by lora
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="test",  # your datasets repo_id
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
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
    # pi0_fast_base by lora
    TrainConfig(
        name="pi0_fast_aloha_robotwin_lora",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
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
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="your_repo_id",  # your datasets repo_id
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
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
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
            repo_id="your_repo_id",  # your datasets repo_id
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
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        freeze_filter=pi0_fast.Pi0FASTConfig().get_freeze_filter(),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30000,
        fsdp_devices=1,  # refer line 359
    ),
    # Pi05_horizon joint QHA training: task-routed frozen pi05 full qnorm teachers/backbones,
    # one shared trainable qha.* head, and QHA-only checkpoints.
    TrainConfig(
        name="pi05_base_aloha_robotwin_full_qha_joint",
        model=pi0_config.Pi0Config(
            pi05=True,
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
                assets_dir=_ct_path('ROBOTWIN_ROOT', 'policy/pi05/assets/pi05_base_aloha_robotwin_full'),
            ),
            adapt_to_pi=False,
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            use_qha=True,
            qha_train_mode="qha_only",
        ).get_freeze_filter(),
        batch_size=64,
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=30000,
        fsdp_devices=1,
        preprocessed_cache_root=_ct_path('ROBOTWIN_ROOT', 'policy/pi05/assets/preprocessed/pi05_base_aloha_robotwin_full'),
        checkpoint_root_tag="pi05_base_aloha_robotwin_full_qha_joint",
        teacher_checkpoint_base_dir=_ct_path('ROBOTWIN_ROOT', 'policy/pi05/checkpoints'),
        save_qha_only_checkpoints=True,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
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
