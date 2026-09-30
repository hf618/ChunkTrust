# Reproducibility status

This release separates code availability, installation checks, inference tests,
and reproduction of reported numbers. A source export is not a fresh benchmark
rerun. Fast-WAM is outside this release.

## Verified result cohorts

Run `python scripts/summarize_results.py`. This reads every released episode,
rejects duplicate identities, averages settings within each task and then tasks,
and rounds only for presentation. Source files and SHA-256 hashes are indexed in
`results/source_ledger.json`.

| Cohort | Released episodes | Recomputed Base / AHS or AHS / AHS+QHA | Status |
|---|---:|---|---|
| RoboTwin 50 tasks, pi0.5 | 4,000 | 56.70 / 63.50 | Matches paper aggregate |
| RoboCasa 24 tasks, pi0.5 | 2,400 | 42.08 / 42.50 | Paper lists Base 40.08; discrepancy unresolved |
| QHA held-out, two tasks | 1,200 across three methods | 40.00 / 41.25 | Paper lists AHS+QHA 42.75; discrepancy unresolved |
| RTC pi0.5 Hard, four tasks | 4,800 | See generated summary | Final SR cells match paper |
| RTC pi0.5 Easy, four tasks | 4,800 | See generated summary | Final SR cells match paper |

For held-out Place Bread Basket Hard, the located records contain QHA-only 35/100
and AHS+QHA 29/100; the manuscript table contains 31/100 and 35/100. These are not
rounding differences. The release preserves recorded outcomes without changing
them to reproduce a reported aggregate. A corrected source cohort, if available,
must carry its own provenance before replacing this one.

Table 2A's eight-task QHA result-to-checkpoint association is unresolved. Local
step-5000 heads are not advertised as the weights for the reported table.
The six-task held-out step-10000 head has a separate, identified training protocol.

## Protocol preservation

- Keep prediction horizon H separate from execution budget K. Expected-round
  selection can produce integer horizons between sparse anchors.
- Fourier analysis runs along the action sequence at each denoising step, with
  the configured full prediction length used for zero padding.
- Episode memory is reset between episodes. Resume includes both Beta state and
  random-generator state. Executed history contains only targets actually sent.
- Keep QHA training targets, sparse/dense evidence, candidate basis, temperature,
  prior strength and checkpoint tied to each cohort.
- Held-out base policies remain task-specific. Only QHA is task-held-out.
- Original full-suite RoboTwin records include strict paired and documented
  fallback sources; the source column is retained. They are not all strictly paired.
- RTC uses the controlled clock and first legal waypoint boundary. The fixed
  comparator is K=40, H=50, with AHS anchors 10/20/30/40 and no QHA. Earlier K20
  fixed controls are excluded. Historical AHS storage labels are not execution K.
- Latency summaries include all outcomes. Wait percentage is total idle physics
  time divided by total physics time; mean observation age is action-weighted.

## Tests and their limits

`pytest` compares the portable selector against AST-extracted original numerical
methods for both PI backends, checks FFT invariance after suffix expiration,
checks QHA fusion against the original, and exercises episode reset and JSON
state restoration. The simple example uses synthetic inputs and is not a result.

The CPU QHA smoke checks three optimization steps on synthetic frozen features
and a parameter roundtrip. It does not establish full-policy training or a
simulator rollout. Existing experimental validation reports are historical;
new release validation is recorded separately in `docs/verification.json`.

A separate real-checkpoint smoke passed two predictions and one RoboTwin
Place Bread Basket Easy episode (seed 100005). This validates the prepared source
checkout against existing installed dependencies; it is not a fresh dependency
installation or a success-rate estimate.
