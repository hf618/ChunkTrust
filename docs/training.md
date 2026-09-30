# QHA training

QHA learns a prior over execution horizons while the base generator stays frozen.
Targets are preferences derived from action-expert evidence, not ground-truth
optimal horizons or environment rewards.

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

The source architecture and training code are included. The mapping between
approved Table 2A rows, selected head, gamma and evidence variant remains
unresolved. Historical step-5000 heads must not be treated as identified Table 2A
weights based on directory names alone.
