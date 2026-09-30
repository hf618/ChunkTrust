<div align="center">

# ChunkTrust
### Adapting Execution Horizons for Robot Policies with Action-Expert Evidence

**Learn when to observe again, from the policy's own action predictions.**

[Paper](https://hf618.github.io/ChunkTrust.github.io/assets/paper/ChunkTrust.pdf) · [Project Page](https://hf618.github.io/ChunkTrust.github.io/) · [4K Video](https://huggingface.co/Niugan/ChunkTrust/resolve/main/media/ChunkTrust_work_intro_4k.mp4) · [Models](https://huggingface.co/Niugan/ChunkTrust) · [Reproduction status](docs/reproducibility.md)

[![Watch the full ChunkTrust introduction in 4K](assets/video-cover.jpg)](https://huggingface.co/Niugan/ChunkTrust/resolve/main/media/ChunkTrust_work_intro_4k.mp4)

**Watch the complete introduction · 4K · 2 min 57 s · English narration and captions**

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

## What ChunkTrust does

A robot policy predicts an action chunk. ChunkTrust chooses how much of that
chunk to execute before collecting a new observation, keeping the base policy
frozen.

- **Complementary evidence:** generation stability within the predicted prefix
  and motion continuity with the executed history.
- **AHS:** episode-local memory accumulates horizon reliability and adapts to the
  current phase.
- **QHA:** a query-based adapter learns a horizon prior across episodes and joins
  the same decision rule at inference.

![ChunkTrust method](assets/method.png)

## Start here

The core example needs **Python 3.11 and NumPy**, without a GPU or simulator.

```bash
git clone https://github.com/hf618/ChunkTrust.git
cd ChunkTrust
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python examples/quickstart.py
python examples/replay_recorded_decisions.py
pytest -q
```

The first example illustrates the API with synthetic inputs. The second checks
23 recorded real-robot horizon decisions. Neither is a success-rate evaluation.

```python
from chunktrust import AHS, AHSConfig

selector = AHS(AHSConfig(horizon=50, candidates=(10, 20, 30, 40, 50)), seed=0)
k, evidence = selector.select(velocity, predicted_actions, executed_history)
# Execute at most the first k actions, then acquire the next observation.
# Reset the selector for each new episode. Preserve its state when resuming.
```

`velocity` has shape `[denoising_steps, horizon, action_dim]`.
`predicted_actions` and `executed_history` use the same action coordinates.
The Fourier transform runs along the action horizon at each denoising step.

## Run with a policy

| Integration | Released code | Start point |
|---|---|---|
| RoboTwin π0 / π0.5 | AHS, QHA, policy traces, evaluation hooks | [Installation](docs/backends.md), [Evaluation](docs/evaluation.md) |
| RoboCasa π0.5 | GR1 observation/action adapter and runtime | [Models](docs/models.md) |
| GR00T N1.5 / N1.6 | Model and simulation integrations | [Backends](docs/backends.md) |
| Qwen3GR00T | StarVLA RoboCasa interface and AHS | [Backends](docs/backends.md) |
| X-VLA | RoboTwin horizon client | [Backends](docs/backends.md) |
| RTC + AHS | Controlled latency evaluation, Hard and Easy | [Latency protocol](docs/evaluation.md#latency-and-rtc) |
| Real robot | Recorded decision example and interface contract | [Real robot](docs/real_robot.md) |

Model and simulator dependencies are installed separately from the core.
Each upstream is pinned to a commit; source overlays preserve backend-specific
behavior. **Source availability does not mean every backend has passed a fresh
end-to-end rollout.** Exact checks and remaining dependencies are recorded in
[verification](docs/verification.json).

## Training and checkpoints

- [Download released weights and check SHA-256](docs/models.md).
- [Train QHA with a frozen generator](docs/training.md).
- [Prepare the six-task held-out protocol](configs/qha_heldout/qha_heldout_protocol.json).
- [Adapt another policy](docs/integration.md).

Large checkpoints and the complete 4K video are hosted on
[Hugging Face](https://huggingface.co/Niugan/ChunkTrust), outside Git history.

## Reproduce result tables

```bash
python scripts/summarize_results.py --output outputs/recomputed_summary.json
```

The release includes **17,200 recorded episode outcomes** from RoboTwin-50,
RoboCasa π0.5, QHA held-out, and the two RTC panels. Aggregation checks duplicate
identities and applies equal task weighting.

The verified RoboTwin-50 π0.5 cohort improves from **56.70% to 63.50%**.
Two other located cohorts differ from the manuscript: RoboCasa Base is 42.08%
in the records versus 40.08% in the paper, and held-out AHS+QHA is 41.25% versus
42.75%. See the [source audit](docs/reproducibility.md) before using these numbers.
Original outcomes are retained, including regressions.

## Repository guide

```text
src/chunktrust/  Model-independent AHS, prior fusion and backend selectors
examples/       Minimal API example and recorded decision replay
configs/        Frozen task, seed and protocol manifests
scripts/        Backend preparation, model download, smoke checks and reports
third_party/    Pinned upstreams, source overlays, provenance and licenses
environments/   Backend environment specifications
results/        Recorded outcomes, source hashes and recomputed summaries
tests/          Numerical equivalence, reset and resume checks
docs/           Training, evaluation, integration and verification status
```

## Citation

```bibtex
@misc{huang2026chunktrust,
  title={ChunkTrust: Adapting Execution Horizons for Robot Policies with Action-Expert Evidence},
  author={Huang, Fanding and Jiang, Jingyan and Bao, Shifeng and Pu, Mingkang and Li, Shiwei and Xu, Jing and Xu, Shijia and Huang, Guanbo and Gu, Chenghao and Huang, Yuzhi and Li, Chenxin and Khan, Faisal Nadeem and Yang, Huan and Wang, Yan and Chi, Cheng and Wang, Zhi},
  year={2026},
  url={https://hf618.github.io/ChunkTrust.github.io/}
}
```

## Acknowledgments and licenses

Built on OpenPI, RoboTwin, RoboCasa, Isaac-GR00T, StarVLA and X-VLA.
Original ChunkTrust code uses the repository's MIT license. Upstream and adapted
files retain their applicable licenses and attribution; see
[third-party notices](third_party/NOTICE.md). Model and dataset terms are separate
from the code license.
