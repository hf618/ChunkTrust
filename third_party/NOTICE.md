# Third-party notices

The root MIT license applies to original ChunkTrust release tooling and authored
code. It does not replace the licenses of upstream source, adapted source,
checkpoints or datasets.

| Upstream | Source | Notice |
|---|---|---|
| OpenPI | https://github.com/Physical-Intelligence/openpi | `licenses/openpi-LICENSE` and the PI overlay LICENSE files |
| RoboTwin | https://github.com/RoboTwin-Platform/RoboTwin | `licenses/robotwin-LICENSE` |
| Isaac-GR00T | https://github.com/NVIDIA/Isaac-GR00T | `licenses/gr00t_n15-LICENSE`, `licenses/gr00t_n16-LICENSE` |
| StarVLA | https://github.com/starVLA/starVLA | `licenses/starvla-LICENSE` |
| X-VLA | https://github.com/2toinf/X-VLA | `licenses/xvla-LICENSE` |

Exact upstream revisions are in `upstreams.json`. `source_manifest.json` maps
exported files to original source paths and SHA-256 hashes. Machine-path changes
and the optional QHA feature-flag compatibility adapter are documented in
`portability_changes.json`. The source snapshots include research modifications,
so an upstream commit alone does not identify the exported working-tree content.

The standalone `_pi_ahs.py`, `prior.py`, `starvla_ahs.py` and `gr00t_ahs.py` are
extracted from those integrations. Their source provenance and applicable upstream
notices are retained here; extraction does not relicense third-party code.

Weights remain subject to applicable base-model and dataset terms. The code
license is not a license grant for upstream model weights or benchmark assets.
