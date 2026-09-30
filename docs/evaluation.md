# Evaluation

## Evaluate the identified held-out QHA checkpoint

Use a dedicated `robotwin --heldout` checkout and its OpenPI environment. Download
the identified QHA head and the corresponding non-qnorm task-specific pi0.5 base.
The layout is `policy/pi05_horizon/checkpoints/<recipe>/<model>/<step>/`.
The released download script supports downloading the head directly to its step
directory; base model folders preserve the original recipe and model names.

From the prepared checkout's `policy/pi05_horizon` directory:

```bash
bash scripts/eval_qha_heldout_manifest.sh \
  --method ahs_qha --task place_bread_basket --setting demo_clean \
  --manifest "$CHUNKTRUST_ROOT/configs/qha_heldout/manifests/place_bread_basket_demo_clean_100.json" \
  --qha-model qha_heldout6_formal_20260727_05 --qha-step 10000 \
  --base-model pi05_place_bread_basket_clean50 --base-step 20000 \
  --result-root "$CHUNKTRUST_ROOT/outputs/heldout" --gpu 0 --dry-run
```

Remove `--dry-run` after installing simulator assets and checking the resolved
checkpoint paths. Use `ahs_only` and `qha_only` for the controls. Explicitly pass
the non-qnorm base name: the historical launcher has a qnorm default intended
for another validation stage. No task, seed, instruction or outcome filtering
is allowed when reproducing the frozen cohort.

The archived result discrepancy is documented separately. Running this command
again constitutes a new experiment, not a correction of recorded outcomes.

## RoboTwin 50-task evaluation

`configs/robotwin50` contains the frozen full-suite protocol and per-task/setting
manifests. Base and AHS share the multitask step-30000 checkpoint. Do not replace
it with a task-specific checkpoint or reuse the eight-task headline numbers.
The 164 located archived cell commands are available through a portable launcher
(the remaining fallback command sources are not yet reconstructed):

```bash
python scripts/eval_robotwin50.py --task place_a2b_left --setting demo_clean \
  --method ahs --backend-root workspaces/robotwin --output outputs/robotwin50 --dry-run
```

Remove `--dry-run` to execute. GPU selection is the only positional override;
the recipe retains its original candidates, temperature, seed and logging flags.
This reconstructs the archived command. Full-suite source-version equivalence
still requires its source-to-run audit and is not established by this launcher.

## RoboCasa GR1

The pi0.5 inference package includes 44D input-state to 29D absolute-action mapping,
H=16, its global quantile normalizer and one ego-view camera. Other camera slots
are masked. It must not use the ALOHA 14D adapter or delta-action postprocessing.
The frozen candidates are 4/8/12/16; Base K=16. The source runners are in the
RoboTwin overlay's `eval_rebuttal/scripts`, with the standalone GR1 OpenPI export
under `third_party/overlays/pi05_gr1`.

GR00T and Qwen3GR00T keep their own embodiment transforms and trace capture.
Consult their pinned upstream installation instructions and released rollout
interfaces. Fresh simulator smoke validation is tracked independently per backend.

## Latency and RTC

`results/rtc_pi05` contains Hard and Easy, each with four tasks, four methods,
three extra delays and 100 episodes per cell. The source runtimes are retained
separately under the RoboTwin overlay's experiment directories 12 and 13.

The experiment uses H=50, fixed K=40, AHS candidates 10/20/30/40, a frozen
142 ms base delay and +0/+100/+200 ms extra delay. Timing is a controlled physical
clock with variable-duration waypoints, not native deployment latency.

`python scripts/summarize_results.py` regenerates SR, Wait %, Wait s/ep,
Calls/ep, Infer s/ep and mean action observation age from all released outcomes.
The historical runtime's integrity gates need its original simulator assets and
source/checkpoint inventories. They are not bypassed by the portable package.
Do not launch the historical pipelines before resolving those inventories for a
new installation. The source export is available for inspection and integration;
full automatic rerunning remains an explicit release gap.
