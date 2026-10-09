# Models and downloadable assets

Model weights are hosted at [Niugan/ChunkTrust](https://huggingface.co/Niugan/ChunkTrust).
`configs/assets.json` lists published artifacts at immutable HF revisions with
per-file SHA-256 hashes. Download and verify without installing a model framework:

```bash
python scripts/download_asset.py qha_heldout --destination checkpoints/qha-heldout/10000
```

The complete video is hosted on [GitHub Releases](https://github.com/hf618/ChunkTrust/releases/tag/media-v1),
at 3840×2160, 30 fps, 177.5 seconds, with English narration and captions.
Download the [original MP4](https://github.com/hf618/ChunkTrust/releases/download/media-v1/ChunkTrust_work_intro_4k.mp4)
and [SHA256SUMS](https://github.com/hf618/ChunkTrust/releases/download/media-v1/SHA256SUMS).
Its SHA-256 is
`19c35d003ec93128ec41315dd1051f30274d241a319a9ca6f56e8214282868ff`.

## Identified weights

| Artifact | Purpose | Availability |
|---|---|---|
| pi0.5 QHA held-out6, step 10000 | Six-task trained head, two held-out tasks | Published; 7.14 MB with assets |
| pi0.5 RoboTwin multitask50, step 30000 | RoboTwin 50-task Base/AHS | Published; 12.44 GB |
| Eight task-specific pi0.5 bases, step 20000 | Task-specific evaluation and RTC | All eight published; 12.44 GB each |
| Eight task-specific pi0 bases, step 30000 | Task-specific evaluation | All eight published; 5.40 GB each |
| pi0.5 RoboCasa GR1, step 30000 | RoboCasa Base/AHS | Existing [ModelScope package](https://modelscope.cn/models/NewGain/pi05_robocasa_gr1_tabletop24) |
| Eight-task pi0 and pi0.5 QHA heads, step 5000 | Intermediate training snapshots | Published; Table 2A association remains unresolved |

A QHA head is not a standalone robot policy. It must be overlaid on the matching
base checkpoint, using the retained normalization and embodiment settings.
The held-out evaluation bases are non-qnorm; teacher training uses qnorm variants.
The training teacher weights/cache remain a separate dependency.

Model-directory size includes all parameter shards and normalization assets;
a partially uploaded directory is not a usable model. Only entries marked
`published` in the asset catalog are accepted by the download script.

## External base policies and data

GR00T, Qwen3GR00T and X-VLA use the checkpoint identities recorded by their
respective evaluation configs and upstream repositories. Their model licenses,
access requirements and simulator assets apply independently of this code repo.
Do not substitute a newer model revision when reproducing a frozen experiment.

The release currently provides evaluation manifests and selected recorded traces,
not the full RoboTwin demonstration datasets or preprocessed online-teacher
training cache. Obtain benchmark assets through the benchmark's official
instructions. A training-data package is not marked available until its schema,
task split and checksums have been verified.

The public model catalog is also available at
[code/assets.json](https://huggingface.co/Niugan/ChunkTrust/blob/main/code/assets.json).
Download that catalog, then pass `--catalog /path/to/assets.json` to
`scripts/download_asset.py`. Each entry still pins an immutable model revision
and validates every downloaded file against its SHA-256.
