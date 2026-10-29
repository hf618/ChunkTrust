# Third-party notices

The root MIT license applies to original ChunkTrust release tooling and authored
code. It does not replace the licenses of upstream source, adapted source,
checkpoints or datasets.

Installed wheels include this notice, the license files and the provenance JSON
files referenced below under `share/chunktrust/third_party/` in the installation
prefix. The full upstream overlays remain in the source repository; they are not
copied into the lightweight wheel.

| Upstream | Source | Notice |
|---|---|---|
| OpenPI | https://github.com/Physical-Intelligence/openpi | `licenses/openpi-LICENSE` and the PI overlay LICENSE files |
| RoboTwin | https://github.com/RoboTwin-Platform/RoboTwin | `licenses/robotwin-LICENSE` |
| Isaac-GR00T | https://github.com/NVIDIA/Isaac-GR00T | `licenses/gr00t_n15-LICENSE`, `licenses/gr00t_n16-LICENSE` |
| StarVLA | https://github.com/starVLA/starVLA | `licenses/starvla-LICENSE` |
| X-VLA | https://github.com/2toinf/X-VLA | `licenses/xvla-LICENSE` |
| RoboDriver | https://github.com/FlagOpen/RoboDriver | `licenses/robodriver-LICENSE` (Apache-2.0) |

Exact upstream revisions are in `upstreams.json`. `source_manifest.json` maps
exported files to original source paths and SHA-256 hashes. Machine-path changes
and the optional QHA feature-flag compatibility adapter are documented in
`portability_changes.json`. The source snapshots include research modifications,
so an upstream commit alone does not identify the exported working-tree content.

The standalone `_pi_ahs.py`, `prior.py`, `starvla_ahs.py` and `gr00t_ahs.py` are
extracted from those integrations. Their source provenance and applicable upstream
notices are retained here; extraction does not relicense third-party code.

The workstation export adds RoboDriver extensions and narrow AgileX OpenPI
overrides. Original and released hashes, source state and portability changes
are recorded separately in `robodriver_sources.json` and
`agilex_openpi_sources.json`. RoboDriver-derived files retain its Apache-2.0
notice, and OpenPI-derived files retain OpenPI's license. The root license does
not replace either upstream's terms.

Weights remain subject to applicable base-model and dataset terms. The code
license is not a license grant for upstream model weights or benchmark assets.
