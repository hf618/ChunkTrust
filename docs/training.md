# QHA training

QHA learns a prior over execution horizons while the base generator stays frozen.
Targets are preferences derived from action-expert evidence, not ground-truth
optimal horizons or environment rewards.

Install the [OpenPI backend environment](backends.md) and activate its `.venv`
before using the training commands. With a prepared checkout:

```bash
cd "$ROBOTWIN_ROOT/policy/pi05_horizon"
uv sync --frozen
uv pip install --no-deps -e "$CHUNKTRUST_ROOT"
source .venv/bin/activate
```

## Lightweight architecture check

Inside the OpenPI environment, with its `src` on `PYTHONPATH`:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python "$CHUNKTRUST_ROOT/scripts/smoke_qha.py"
```

This is a three-step optimization test with synthetic features, not the paper
training recipe.

## Six-task held-out recipe

Prepare a dedicated checkout:

```bash
python scripts/prepare_backend.py robotwin --heldout --destination workspaces/robotwin-heldout
```

The overlay includes `scripts/train_qha_heldout_8gpu.sh`, `scripts/train.py`, the
training loader, teacher workers, train/held-out split, and teacher manifest.
Set `ROBOTWIN_ROOT` to that checkout. Put datasets, normalization assets and
frozen teacher checkpoints at the paths specified by the resolved train config.

Training tasks: Handover Block, Handover Mic, Hanging Mug, Place A2B Left, Place
Bread Skillet, Place Can Basket. Blocks Ranking RGB and Place Bread Basket are
held out from QHA training. The teacher manifest uses **qnorm teachers**, whereas
the identified held-out evaluation uses **non-qnorm bases**. Keep this distinction.

From `policy/pi05_horizon`:

```bash
bash scripts/train_qha_heldout_8gpu.sh --mode smoke --exp-name qha-smoke --dry-run
bash scripts/train_qha_heldout_8gpu.sh --mode formal --exp-name qha-heldout-reproduction
```

This registered launcher uses two student GPUs and six teacher GPUs, batch 384,
and 10,000 steps. It requires the original preprocessed dataset/cache schema.
It is distinct from the eight-task batch-256 recipe described for Table 2A.
Do not substitute a cached teacher protocol or an eight-task recipe and label it
as this training run.

The released head supports evaluation before a full training reproduction is
attempted. Complete training-data/cache availability is tracked in the asset
catalog; unprovided data is a reproduction dependency, not automatically fetched.

## Eight-task QHA

Prepare the backend with `--heldout` to include the online-teacher runtime used
by both task splits. The eight-task launcher selects the separate
`pi05_base_aloha_robotwin_full_qha_joint` config; it does not use the held-out split.
After installing the OpenPI backend environment, activate it before launching:

```bash
export ROBOTWIN_ROOT="$CHUNKTRUST_ROOT/workspaces/robotwin-qha"
bash "$CHUNKTRUST_ROOT/scripts/train_qha_eight_task.sh" --exp-name qha-eight-task --dry-run
bash "$CHUNKTRUST_ROOT/scripts/train_qha_eight_task.sh" --exp-name qha-eight-task
```

This pi0.5 recipe reconstructs the manuscript's batch-256, 10,000-step setup:
32 examples per task, 64 history and 50 future steps, eight queries of width 256,
AdamW, 1,000 warmup steps, peak learning rate 5e-5, final rate 5e-6, and EMA 0.99.
The student uses GPUs 0 and 1; eight frozen teachers are routed over GPUs 2–7.
Two teacher workers each hold two policies, so allow enough GPU memory for both.
The launcher is validated for configuration and CLI parsing, not a completed
new training run.

`configs/qha_8task/pi05_teachers.json` identifies the eight qnorm teacher
checkpoints expected under `policy/pi05/checkpoints`. Supply these teacher
weights, clean50 LeRobot datasets, normalization assets and preprocessed caches
before launching; they are distinct from the non-qnorm evaluation bases.
Training includes the six tasks above plus Blocks Ranking RGB and Place Bread
Basket. Outputs are saved beneath
`policy/pi05_horizon/checkpoints/pi05_base_aloha_robotwin_full_qha_joint/<exp-name>/`.

### Released eight-task checkpoints

The pi0 and pi0.5 heads under `qha_pi0_8task_step5000` and
`qha_pi05_8task_step5000` are intermediate step-5000 checkpoints from the
historical `b256_s10k_balanced` runs. Their original model snapshots are included.
They are not identified as the final Table 2A checkpoints: association with the
reported results, gamma and evidence variant remains unresolved. Do not infer a
10,000-step checkpoint from a run name containing `s10k`.

The runnable eight-task launcher currently covers pi0.5. The historical pi0
head uses a LoRA base, and the available pi0 trainer does not include the routed
online-teacher training path. A pi0 command is not presented as runnable until
that matching training source is recovered. Its checkpoint is released for
inspection and use with the corresponding pi0 base.
