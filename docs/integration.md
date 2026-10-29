# Integrating another action-chunk policy

AHS requires the internal velocity predictions from each flow/denoising step,
not only the final action chunk. Expose a trace with shape `[T, H, D]` and the
final actions `[H, D]`. Apply the same action-coordinate convention to recorded
executed history. Keep the model's observation transforms, normalizer, sampler,
noise schedule and action postprocessing unchanged.

```python
selector = AHS(config, seed=episode_seed)
while not done:
    actions, velocity = policy.predict_with_trace(observation)
    k, info = selector.select(velocity, actions, executed_history)
    actually_executed = controller.execute_prefix(actions, k)
    executed_history.extend(actually_executed)
    observation = environment.observe()
```

This pseudocode describes the synchronous interface. Real controllers can stop
before k on termination or an error; do not append unused actions to history.
The portable selector validates finite inputs and candidate feasibility.
Configuration dimensions and candidate horizons must be integers (not booleans
or integral-valued floats). Numeric selector parameters must be finite real
numbers; booleans and numeric strings are rejected. NumPy integer and floating
scalars are normalized to Python scalars for JSON state serialization.

Reproduce the local dependency and distribution checks with Python 3.11+ and `uv`,
from the repository root:

```bash
python scripts/check_runtime_matrix.py --output outputs/runtime-matrix
```

This runs the four documented Python/NumPy cases, source regression tests,
synthetic examples, wheel/sdist content checks and an installed-wheel check from
a temporary directory outside the checkout. Commands, versions and results are
recorded in the output directory; no model or device is constructed.
The four-case matrix is a local validation tool; the existing CI is unchanged.

For QHA, use the frozen model's context tokens and action-expert latents as
inputs to the released head. Match the head's candidate basis exactly. Call
`AHS.evidence()` to obtain evidence without updating memory, then the native
`HybridQHARuntimeSelector.select(prior, q_mix, max_exec_length=H)` to perform one
combined decision and one memory update. Dense deployment must use the original
interpolation or dense-evidence path specified for its experiment.

Keep one hybrid selector per episode and use its native reset convention.
Do not update an AHS posterior and then feed its sampled scores into another
posterior: that would count memory twice. The lightweight quickstart deliberately
uses the unsampled evidence values when showing prior fusion.
