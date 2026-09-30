# Real-robot interface and traces

The paper uses an AgileX COBOT Magic ALOHA-style bimanual platform with pi0.5.
Four household tasks compare fixed K=25 with AHS over 15 rollouts per task.
The reported real-robot metric averages predefined sub-step process scores;
it is not binary success rate.

The portable AHS interface accepts 14-dimensional executed joint/action targets
and a denoising trace. A robot adapter must retain its trained state layout,
three-camera order, normalization, gripper convention and controller timing.
The original device-side RoboDriver process was on the robot workstation and
has not been located in this release workspace. Accordingly, this repository
does not claim a runnable hardware deployment entrypoint.

`examples/replay_recorded_decisions.py` checks 23 actual Duck-to-Drawer replans.
Its source recording ended by interruption; the example is a selection-arithmetic
audit and is not used to infer success. It retains evidence, selected horizons,
and before/after Beta states without copying video or machine paths into the code
package. Full source identity and its hash accompany the example.

The project page and 4K introduction provide visual demonstrations. A video
cannot replace the missing hardware driver, calibration and scoring records.
