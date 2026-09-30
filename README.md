<div align="center">

# ChunkTrust
### Adapting Execution Horizons for Robot Policies with Action-Expert Evidence

**Learning when to observe again, from the policy's own action predictions.**

[![Paper](https://img.shields.io/badge/Paper-PDF-181717?style=for-the-badge)](https://hf618.github.io/ChunkTrust.github.io/assets/paper/ChunkTrust.pdf)
[![Project Page](https://img.shields.io/badge/Project_Page-205B47?style=for-the-badge)](https://hf618.github.io/ChunkTrust.github.io/)
[![Models](https://img.shields.io/badge/Models-Hugging_Face-205B47?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Niugan/ChunkTrust)
[![Core checks](https://github.com/hf618/ChunkTrust/actions/workflows/core.yml/badge.svg)](https://github.com/hf618/ChunkTrust/actions/workflows/core.yml)

</div>

<details>
<summary>Authors and affiliations</summary>

**Fanding Huang**¹²\*, **Jingyan Jiang**⁴\*, **Shifeng Bao**³\*, **Mingkang Pu**¹,
**Shiwei Li**⁵, **Jing Xu**⁶, **Shijia Xu**⁷, **Guanbo Huang**¹, **Chenghao Gu**¹,
**Yuzhi Huang**¹, **Chenxin Li**⁸, **Faisal Nadeem Khan**¹, **Huan Yang**²,
**Yan Wang**¹, **Cheng Chi**³†, **Zhi Wang**¹†

¹ Tsinghua University · ² Beijing Academy of Artificial Intelligence ·
³ Renmin University of China · ⁴ Shenzhen Technology University ·
⁵ Hefei University of Technology · ⁶ Jiangnan University ·
⁷ Chongqing University · ⁸ The Chinese University of Hong Kong

\* Equal contribution. † Corresponding authors.

</details>

<div align="center">
  <video src="https://github.com/user-attachments/assets/921168dc-0bb9-48a2-ba1c-035c392da6b7" controls width="100%"></video>
</div>

## 🔥 News

- **2026.09.30:** Our [Project Page](https://hf618.github.io/ChunkTrust.github.io/) is live!

## 🧭 Overview

**Optimal replanning horizons depend on the task phase.** ChunkTrust adapts how many actions a robot commits to from each predicted chunk while keeping the base policy frozen.

<p align="center">
  <img src="assets/method.png" alt="ChunkTrust combines generation stability, motion continuity, episode-local memory and a learned horizon prior" width="100%">
</p>

Current action-expert evidence, accumulated episode memory, and a prior learned across episodes contribute to **one horizon decision**. AHS runs without training. Adding QHA supplies a learned prior to the same selector.

## 📖 Abstract

Robot foundation policies predict action chunks, but how many actions to execute before replanning depends on the current task phase. We introduce **ChunkTrust**, which treats the execution horizon as a latent variable inferred from action-expert evidence rather than a fixed hyperparameter. Its training-free *Action-aware Horizon Selector* (AHS) combines intra-chunk spectral stability of generation traces with inter-chunk continuity between executed history and predicted actions. An online Beta posterior with kernel forgetting tracks horizon preferences across replans. A lightweight *Query-based Horizon Adapter* (QHA) optionally learns a context-conditioned dense prior from complementary evidence, fused with current evidence and episode-local Beta memory while the base policy remains frozen. Across RoboTwin2.0 and RoboCasa GR1 Tabletop, AHS improves overall task-averaged success for each evaluated base-policy configuration, including gains of 6.80 percentage points on π0.5 over all 50 RoboTwin2.0 tasks and 9.67 percentage points on Qwen3GR00T in RoboCasa. AHS+QHA raises the gain over Base to 9.44 percentage points on the eight-task π0.5 evaluation. On four real-world household tasks, AHS improves the equal-task mean normalized process score from 50.4% to 57.5%. Ablations examine the contributions of both evidence terms, temporal memory, and the learned prior.

## 🧩 Method

1. **Complementary action-expert evidence.** Intra-chunk generation stability measures how the velocity-prefix spectrum evolves during denoising. Inter-chunk motion continuity checks the boundary between executed history and the predicted prefix. The Fourier transform runs along the action horizon.
2. **Episode-local memory with AHS.** A Beta posterior accumulates horizon preferences across replans. Kernel forgetting lets the decision respond to phase changes. Reset this memory at the start of each episode.
3. **A learned prior with QHA.** A lightweight query head uses frozen policy features to learn dense horizon preferences across episodes from evidence-derived targets.
4. **One execution decision.** Fuse the learned prior, when present, with current evidence and episode memory, choose a prefix, execute it, and observe again. The base action generator remains frozen.

The [integration guide](docs/integration.md) describes traces, action coordinates, selector state and prior fusion.

## 📊 Results

Selected results from the [paper](https://hf618.github.io/ChunkTrust.github.io/assets/paper/ChunkTrust.pdf). Simulation rows report task-averaged success. The real-robot row reports normalized process score.

| Benchmark | Policy | Scope | Base | ChunkTrust | Method |
| --- | --- | --- | ---: | ---: | --- |
| RoboTwin2.0 | π0.5 | 50 tasks | 56.70% | **63.50%** | AHS |
| RoboTwin2.0 | π0.5 | 8 tasks | 29.63% | **39.06%** | AHS + QHA |
| RoboCasa GR1 Tabletop | Qwen3GR00T | 24 tasks | 47.83% | **57.50%** | AHS |
| Real robot | π0.5 | 4 household tasks | 50.4% | **57.5%** | AHS |

The 50-task and eight-task RoboTwin evaluations use different checkpoint regimes. Full protocols and per-task outcomes are in the paper.

## 🛠️ Installation

```bash
git clone https://github.com/hf618/ChunkTrust.git
cd ChunkTrust
conda env create -f environment.yml
conda activate chunktrust
python -m pip install -e . --no-build-isolation
```

For selector smoke tests:

```bash
python -m pip install -e '.[test]' --no-build-isolation
python examples/quickstart.py
pytest -q
```

Alternatively, in a Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
```

Policy training and simulation use the [backend environments](docs/backends.md).

## 📦 Data and Checkpoints

Find model weights at [Niugan/ChunkTrust](https://huggingface.co/Niugan/ChunkTrust). Task splits and evaluation manifests are in `configs/`, with recorded results in `results/`.

Training data for the simulation experiments come from **RoboTwin** and **RoboCasa**. The QHA protocols below use RoboTwin clean50 demonstrations.

QHA learns horizon preferences from action-expert evidence while the base policy stays frozen.

| Protocol | Policy | Training | Checkpoint |
| --- | --- | --- | --- |
| Eight-task augmentation | π0.5 | Batch 256, 10,000 steps | [Intermediate head, step 5,000](https://huggingface.co/Niugan/ChunkTrust/tree/main/checkpoints/qha_pi05_8task_step5000) |
| Eight-task augmentation | π0 | See [training details](docs/training.md#eight-task-qha) | [Intermediate head, step 5,000](https://huggingface.co/Niugan/ChunkTrust/tree/main/checkpoints/qha_pi0_8task_step5000) |
| Six-task training, two held-out tasks | π0.5 | Batch 384, 10,000 steps | [Head, step 10,000](https://huggingface.co/Niugan/ChunkTrust/tree/main/checkpoints/qha_pi05_heldout6_step10000) |

**Download QHA heads:**

```bash
python scripts/download_asset.py qha_pi05_8task_step5000 --destination checkpoints/qha-pi05-eight/5000
python scripts/download_asset.py qha_pi0_8task_step5000 --destination checkpoints/qha-pi0-eight/5000
python scripts/download_asset.py qha_heldout --destination checkpoints/qha-heldout/10000
```

Each head is loaded alongside its matching frozen base policy. See [training details](docs/training.md) for the eight-task recipe and checkpoint provenance.

## 🏋️ QHA Training

Prepare the π0.5 QHA backend from the repository root, then install its [environment and training data](docs/training.md):

```bash
export CHUNKTRUST_ROOT="$PWD"
python scripts/prepare_backend.py robotwin --heldout --destination workspaces/robotwin-qha
export ROBOTWIN_ROOT="$CHUNKTRUST_ROOT/workspaces/robotwin-qha"
```

**Eight-task training:**

```bash
bash "$CHUNKTRUST_ROOT/scripts/train_qha_eight_task.sh" --exp-name qha-eight-task
```

**Six-task training with two held-out tasks:**

```bash
cd "$ROBOTWIN_ROOT/policy/pi05_horizon"
bash scripts/train_qha_heldout_8gpu.sh --mode formal --exp-name qha-heldout
```

Both launchers accept `--dry-run`. The held-out split trains on Handover Block, Handover Mic, Hanging Mug, Place A2B Left, Place Bread Skillet and Place Can Basket. Blocks Ranking RGB and Place Bread Basket are held out.

## 🧪 Evaluation

Recompute the released outcomes from the repository root:

```bash
python scripts/summarize_results.py --output outputs/recomputed_summary.json
```

Inspect an archived RoboTwin-50 evaluation command:

```bash
python scripts/prepare_backend.py robotwin --destination workspaces/robotwin
python scripts/eval_robotwin50.py --task place_a2b_left --setting demo_clean \
  --method ahs --backend-root workspaces/robotwin --output outputs/robotwin50 --dry-run
```

Remove `--dry-run` after setting up the backend, simulation assets and matching checkpoints.

The [evaluation guide](docs/evaluation.md) also covers held-out QHA, RoboCasa GR1, and the RTC panels with four tasks, four methods, three delays and 100 episodes per cell in both Easy and Hard. Latency reports include success rate, waiting time, policy calls, inference time and mean observation age.

## 🗂️ Repository Structure

```text
src/chunktrust/  AHS, learned-prior fusion and backend selectors
examples/       Minimal API example and recorded decision replay
configs/        Frozen task, seed and protocol manifests
scripts/        Backend setup, model download, training smoke checks and evaluation
third_party/    Pinned upstreams, source overlays, provenance and licenses
environments/   Backend environment specifications
results/        Recorded outcomes, source hashes and recomputed summaries
tests/          Numerical equivalence, reset and resume checks
docs/           Training, evaluation, integration and reproduction status
```

## 📝 Citation

```bibtex
@misc{huang2026chunktrust,
  title={ChunkTrust: Adapting Execution Horizons for Robot Policies with Action-Expert Evidence},
  author={Huang, Fanding and Jiang, Jingyan and Bao, Shifeng and Pu, Mingkang and Li, Shiwei and Xu, Jing and Xu, Shijia and Huang, Guanbo and Gu, Chenghao and Huang, Yuzhi and Li, Chenxin and Khan, Faisal Nadeem and Yang, Huan and Wang, Yan and Chi, Cheng and Wang, Zhi},
  year={2026},
  url={https://hf618.github.io/ChunkTrust.github.io/}
}
```

## 🙏 Acknowledgements

Built on OpenPI, RoboTwin, RoboCasa, Isaac-GR00T, StarVLA and X-VLA. Please cite the corresponding upstream projects when using their models or benchmarks.

Original ChunkTrust code uses the repository's MIT license. Upstream and adapted files retain their applicable licenses and attribution, listed in [third-party notices](third_party/NOTICE.md). Model and dataset terms are separate from the code license.
