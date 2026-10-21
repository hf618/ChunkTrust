# AgileX real-robot source and deployment

The paper uses an AgileX COBOT Magic ALOHA-style bimanual platform with pi0.5.
Four household tasks compare fixed K=25 with AHS over 15 rollouts per task.
The reported metric averages predefined sub-step process scores; it is not
binary success rate.

The device-side source has now been located on the robot workstation and is
released as a [RoboDriver overlay](../third_party/overlays/robodriver/scripts/).
A separate [OpenPI overlay](../third_party/overlays/agilex_openpi/) supplies the
trained AgileX state layout and websocket denoising traces. The two deployment
entrypoints are the fixed-prefix baseline and synchronous AHS. Their shared
mapping and optional recording helpers are included as supporting code. These are source
exports with offline regression checks, not a new hardware evaluation or a
reconstruction of all paper rollouts. Calibration, task-specific checkpoints
and complete scoring records are still needed for hardware reproduction.

## Core source layout

| Entry in prepared RoboDriver `scripts/` | Purpose |
|---|---|
| `openpi_bridge_rollout_safe.py` | Fixed-prefix baseline |
| `openpi_bridge_rollout_ahs.py` | Historical synchronous AHS device integration |
| `openpi_bridge_common.py` | Shared robot construction, raw26/camera mapping and action-dictionary helpers |
| `eval_video_recorder.py` | Optional three robot videos plus overview-camera recording |
| `eval_episode_npz_logger.py` | Optional commands, observations, timing, AHS evidence and Beta states |

Only the first two files are rollout commands. Async prefetch, standalone reset,
manual motion and free-mode utilities are outside this core export. The original
workstation files remain unchanged. `make_ahs_selectors(args, candidates)` in the
synchronous module exposes the original selector construction without importing
hardware or model libraries.

Original and exported hashes, source state and portability changes are in
[robodriver_sources.json](../third_party/robodriver_sources.json) and
[agilex_openpi_sources.json](../third_party/agilex_openpi_sources.json).
The [upstream catalog](../third_party/upstreams.json) pins the Git revisions.
Device extensions were untracked in the local checkout, so the upstream revision
alone does not identify them; the per-file hashes do.

## Prepare separate source checkouts

Start in the ChunkTrust repository. These commands only prepare source files:

```bash
export CHUNKTRUST_ROOT="$PWD"
python scripts/prepare_backend.py robodriver --destination workspaces/robodriver
python scripts/prepare_backend.py agilex_openpi --destination workspaces/agilex-openpi
export ROBODRIVER_ROOT="$CHUNKTRUST_ROOT/workspaces/robodriver"
export OPENPI_ROOT="$CHUNKTRUST_ROOT/workspaces/agilex-openpi"
```

The preparer refuses existing destinations. `--local-source /path/to/clone`
uses committed Git objects without importing local working-tree changes.
`agilex_openpi` layers the released PI0.5 backend first, then the narrow AgileX
overrides, retaining the released QHA support.

Use separate environments for the policy server (Python 3.11) and RoboDriver
(Python 3.10+). Install the released OpenPI environment, for example:

```bash
cd "$OPENPI_ROOT"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
uv pip install --no-deps -e "$CHUNKTRUST_ROOT"
```

Install RoboDriver and its AgileX plugin following the
[pinned upstream instructions](https://github.com/FlagOpen/RoboDriver/blob/5beceeabb92291ed30c2d58c11d68cb1b8bd63e3/robodriver/robots/robodriver-robot-agilex-aloha-aio-dora/README.md).
Install only the lightweight client into the RoboDriver environment:

```bash
cd "$ROBODRIVER_ROOT"
source .venv/bin/activate
python -m pip install -e "$OPENPI_ROOT/packages/openpi-client"
```

Dora/camera/arm node environments are separate from this bridge environment.
The pinned upstream dataflow needs the actual three camera serials and verified
left/right CAN assignment. Saved poses, CAN USB-port maps, camera serials,
private training paths, videos and checkpoints are not copied into the release.
Local camera startup/timing and USB/CAN setup changes are installation-specific
and are not applied by the preparer. Use a matching configuration from an
established robot installation, or configure the pinned upstream dataflow for
the new machine.

## Policy server

Select a full pi0.5 real-robot checkpoint with its own `params/` and
`assets/<asset_id>/norm_stats.json`. Set `CHECKPOINT` to that directory. The
historical configuration is `pi05_base_aloha_agilex_real_full`.

```bash
cd "$OPENPI_ROOT"
source .venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/serve_policy.py \
  --port 8001 --return-denoise-trace \
  policy:checkpoint \
  --policy.config=pi05_base_aloha_agilex_real_full \
  --policy.dir="$CHECKPOINT"
```

The flag passes `return_denoise_trace=True` to the existing model trace path.
Trace arrays are serialized as ordinary NumPy floating-point arrays, including
when model inference uses bfloat16. If the configured normalization asset is
absent, loading falls back only when the checkpoint has exactly one norm-stat
asset, and reports its name; ambiguous assets still fail. For the fixed-prefix
comparison, the same server/checkpoint can run without `--return-denoise-trace`.

## Observation and action contract

| Data | Layout / mapping |
|---|---|
| Raw state, 26 values | right joints+gripper `[0:7]`, right TCP `[7:13]`, left joints+gripper `[13:20]`, left TCP `[20:26]` |
| Model state, 14 values | raw `[13:20]` followed by raw `[0:7]`: left7, right7 |
| Dataset action input | right7,left7 is reordered to left7,right7 during training |
| Returned policy action | absolute left7,right7; do not reorder it a second time |
| Top camera | `image_top` -> `cam_high` |
| Wrist cameras | `image_left` -> `cam_left_wrist`; `image_right` -> `cam_right_wrist` |

The alignment disables Trossen-to-PI remapping (`adapt_to_pi=False`), uses delta
joint training targets except for grippers, restores absolute targets on output,
and preserves pi0.5 quantile normalization. Gripper scales must match the
checkpoint and driver; raw numerical limits are not a universal unit conversion.

The exact inspected driver packages, public wheel hashes and wire formulas are
retained in [runtime_reference.json](../configs/real_robot/runtime_reference.json).
The installed Python sources match those public wheels: `dora-arm-piper==0.0.5`,
`dora-camera-realsense==0.0.2` and `piper-sdk==0.6.2`. These are observed reference
versions, not a complete environment lock. Pin the intended node packages when
configuring a new dataflow; an unversioned package install can select another
implementation in the future.

In that Piper node, a joint command is encoded as
`round(q * 57324.840764)` and feedback divides by the same factor. The gripper
command is `int(abs(g * 100000))`, while feedback divides the SDK value by
100000. The SDK documents 0.001 mm per gripper unit, so nonnegative raw `g`
corresponds nominally to `g / 10` metres. This does not establish physical
closed/open calibration. RoboDriver's degree/range metadata labels do not
convert these raw node values. A ManiMux adapter must map the calibrated physical
width into its normalized opening explicitly; forwarding raw `g` as metres or
as `[0,1]` aperture would change the checkpoint's action meaning.

`denoise_trace['v']` has shape `[denoising_steps,H,model_action_dim]` (or one
leading singleton batch dimension). It stays in normalized model coordinates;
the historical bridge uses its first 14 dimensions. Returned `actions[H,14]`
are postprocessed absolute robot targets. This historical mixed trace/action
contract is retained rather than presented as a new coordinate conversion.

## Bridge commands

Run bridge commands in the prepared RoboDriver environment. The `--help` paths
work without a robot, policy server, OpenPI client or video dependencies:

```bash
cd "$ROBODRIVER_ROOT"
python scripts/openpi_bridge_rollout_ahs.py --help
python scripts/openpi_bridge_rollout_safe.py --help
```

After CAN and the installation's Dora dataflow are configured and running, the
bridge itself provides the dynamic RoboDriver node. Do not run another process
using the same robot node concurrently. `--execute 0` suppresses bridge action
sends but still connects to the live robot and policy server; it is not an
offline simulation.

Use a task prompt matching the checkpoint. A short single-arm preview:

```bash
python scripts/openpi_bridge_rollout_ahs.py \
  --host 127.0.0.1 --port 8001 \
  --prompt "Move the bread from the left side of the table to the right side of the table using both arms." \
  --ahs-candidates 10,20,30,40,50 --ahs-exec-mode expected_round \
  --ahs-expected-temp 1.0 --ahs-ts-seed 0 --ahs-metadata-policy require \
  --single-arm-only right --episode-max-action-steps 50 --execute 0
```

The fixed K=25 entrypoint is:

```bash
python scripts/openpi_bridge_rollout_safe.py \
  --host 127.0.0.1 --port 8001 \
  --prompt "Move the bread from the left side of the table to the right side of the table using both arms." \
  --chunk-prefix-steps 25 --episode-max-action-steps 50 --execute 0
```

These examples verify command construction, not the paper's complete deployment
settings. Execution is opted into with `--execute 1`; dual-arm and gripper
control require `--allow-dual-arm` and `--allow-gripper`. Set the task's measured
control rate, limits, filtering and selector parameters explicitly. The CLI
defaults to temperature 1.0; portable `AHSConfig` defaults to 0.8, and the
released Duck replay records 0.6.

The two rollout commands retain their existing `--ctrl-c-soft-reset` option:
after Ctrl-C during execution it holds the current pose. It does not return to a
saved initial pose. No standalone reset command or recorded reset target is
distributed.

## Historical behavior and recording

The exported bridge keeps its original selector/control behavior, rather than
silently substituting `chunktrust.AHS`:

- Near the episode budget, it shortens the FFT length and feasible candidates;
  changing candidates resets Beta state. The portable PI selector keeps the
  configured FFT horizon and rejects insufficient suffixes.
- Its inter-chunk history records policy prefixes after completed chunks, not
  the clipped/blended commands actually sent. History can advance in preview
  mode. NPZ `executed_action_lr14` separately records filtered targets with
  `execute_enabled`; a preview target is not a sent command.
- `--ahs-metadata-policy require` prevents missing metadata silently producing
  a fixed-prefix run. Choice precedence is server `exec_length`, denoising
  trace, then QHA metadata.
- Robot/recorder/client startup precedes the rollout's `try/finally`; a startup
  failure can skip the normal disconnect path. Hardware teardown on failure
  has not been validated by this source export.

Recording is opt-in with `--record-video 1`. It writes to relative
`records/eval_videos`, or `--record-root <path>`, and also requires an overview
camera (auto-detected DJI Osmo, or `--record-overview-device <device>`), OpenCV,
FFmpeg and video-device access. These scripts have no NPZ-only recording mode.
Metadata includes CLI arguments, available source revisions, frame timestamps,
targets, evidence and posterior snapshots. Replan action ranges
describe intended prefixes and can overrun an interrupted recording; inspect
per-step rows and `execute_enabled` to determine what was logged as sent.

## Validation and result limits

CPU checks exercise observation/action mapping, command gating, selector
arithmetic, metadata handling, serialization helpers and source preparation.
They perform no CAN activation, Dora startup, arm enable, motion, camera capture
or checkpoint inference. See [verification.json](verification.json) for
checks actually completed.

`examples/replay_recorded_decisions.py` checks 23 actual Duck-to-Drawer replans.
Its source recording ended by interruption; this is a selection-arithmetic audit
and is not used to infer success. It retains evidence, selected horizons and
before/after Beta states. Source identity and hash accompany the example. The
project video does not replace calibration or complete scoring records.

## Developer handoff without the robot workstation

All source for these two entrypoints and their model-side trace path is either
in this repository or in the pinned upstream revisions. Source preparation uses
public remotes by default; no author checkout, workstation path, editable
installation on the robot, or saved pose is required for code development.

In a Python 3.11 environment, from a fresh ChunkTrust clone:

```bash
python -m pip install -e '.[test]'
python examples/real_robot_core_smoke.py
python -m pytest -q
python scripts/check_release.py
```

The smoke uses explicitly synthetic state, camera pixels, actions and denoising
velocities. It imports the released fixed/synchronous modules, runs the original
selector and command mapping, and prints a small JSON report. It opens no socket,
camera, robot or model checkpoint. The numerical replay tests retain original
source identities and golden decisions, including the historical end-of-budget
FFT/candidate behavior. These checks can be used when developing on another
machine without silently substituting a different selector.

For a future ManiMux integration, the model framework should own the AgileX
observation profile and opt-in trace sampler, while the runtime strategy owns
K selection, episode state, acceptance/reset and execution history. The exported
controller loop is a reference implementation; it does not implement ManiMux's
plugin interfaces. Preserve the explicit raw26-to-left/right mapping, camera
order, checkpoint normalization and gripper convention. Port the fixed-prefix
and synchronous AHS paths first. Define behavior for rejected plans and actually
sent commands before adding asynchronous scheduling.

Code development can continue from this repository. Running the trained policy
still requires a compatible real-robot checkpoint and its normalization assets;
physical execution additionally needs the new station's device configuration
and calibration. Neither weights nor station-specific addresses are embedded in
the source export.
