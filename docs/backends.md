# Backends and environments

Install the lightweight package first, then prepare a separate checkout for each
policy family. Python 3.11 and Linux are the initial supported platform.

```bash
python scripts/prepare_backend.py robotwin --destination workspaces/robotwin
```

The installer checks out the exact commit in `third_party/upstreams.json` and
copies the released overlay. It refuses existing destinations. `--local-source`
can use committed Git objects from an existing clone without copying its working
tree. Use a new checkout with `--heldout` for the separate held-out QHA recipe.

| Backend | Prepare key | Main integration |
|---|---|---|
| RoboTwin pi0 / pi0.5 | `robotwin` | policy-local hooks, model traces, OpenPI QHA, shared eval hooks |
| GR00T N1.5 | `gr00t_n15` | GR00T policy / simulation service |
| GR00T N1.6 | `gr00t_n16` | GR00T evaluation policy and horizon selector |
| Qwen3GR00T | `starvla` | RoboCasa model interface and horizon selector |
| X-VLA | `xvla` | RoboTwin horizon client |
| RoboCasa pi0.5 | OpenPI GR1 export | `third_party/overlays/pi05_gr1` and exported model package |
| RTC pi0.5 | `robotwin` | Separate Hard/Easy runtime modules under `eval_rebuttal` |

Use each upstream's installation instructions at the pinned commit. OpenPI's
`pyproject.toml` and `uv.lock` are included in the PI overlay; other environment
files are under `environments/`. Do not install mutually incompatible policy
stacks into the lightweight NumPy environment.

For OpenPI, after obtaining the simulator dependencies/assets required by the
pinned RoboTwin version:

```bash
export CHUNKTRUST_ROOT="$PWD"
export ROBOTWIN_ROOT="$CHUNKTRUST_ROOT/workspaces/robotwin"
cd "$ROBOTWIN_ROOT/policy/pi05_horizon"
uv sync --frozen
uv pip install --no-deps -e "$CHUNKTRUST_ROOT"
```

The recorded OpenPI version uses JAX 0.5.0, Flax 0.10.2 and CUDA 12. GPU driver,
EGL/Vulkan, robot assets, and simulator libraries must also be compatible. An
import-only check cannot validate graphics initialization.

## Portable paths

Set these for your own installation; no author home directory is required:

- `ROBOTWIN_ROOT`: prepared RoboTwin checkout.
- `CHUNKTRUST_WORKSPACE_ROOT`: parent of `starVLA` and `robocasa-gr1-tabletop-tasks`.
- `CHUNKTRUST_MODELS_ROOT`: parent of the exported GR1 package.
- `CHUNKTRUST_DATA_ROOT`: downloaded training data.
- `CHUNKTRUST_CACHE_ROOT`: reusable model/data cache.
- `CHUNKTRUST_RESULTS_ROOT`: output and experiment area.

Path substitutions are recorded in `third_party/portability_changes.json`.
`third_party/source_manifest.json` records hashes before those substitutions.
Sources retain original backend-specific selection behavior. The standalone
`chunktrust.AHS` API is the PI-family implementation, not a silent replacement
for every backend selector.

## Integration status

Core numerical regression is tested. A real pi0.5+QHA prediction check and one
Place Bread Basket Easy simulator episode passed in a newly prepared source
checkout using the existing backend environment and simulator assets. Source overlays for other backends are
included with pinned upstream commits, but an overlay alone is not a guarantee
of end-to-end simulator reproduction. See `verification.json` for the exact
checks performed and `reproducibility.md` for unresolved result provenance.
