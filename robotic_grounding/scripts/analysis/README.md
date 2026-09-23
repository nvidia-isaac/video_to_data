# Tissue-box CHORD reproduction — analysis scripts

One-off diagnostics written while investigating why an 8-hour PPO run on a
monocular ego-video sequence (`ego_recon / tissue_box_refined`) produced a
policy that never lifted the object, while every logged metric looked healthy.

Findings are in [`FINDINGS.md`](FINDINGS.md). The report is
`robotic_grounding/out/report_tissue_box_v2d.html` (gitignored; also published
as a Claude artifact).

| script | what it answers | where it runs |
|---|---|---|
| `lift_timing.py` | When does the reference actually lift the object, and for how long? | container, no sim |
| `objaway_margin.py` | What does a do-nothing policy feed into `object_away_from_trajectory`? | container, no sim |
| `rew_headroom.py` | What `object_keypoints_tracking_exp` score does a frozen object get? | container, no sim |
| `rew_sweep.py` | How does that score move with `var`, the keypoint lever, and motion amplitude? | container, no sim |
| `pen_per_hand.py` | Per-hand, per-frame hand-object penetration (capsule based) | container, no sim |
| `diag_voc.py` | With zero actions at VOC=1.0, which termination fires and how close do deviations get to threshold? | Isaac Lab, 512 envs |
| `../rsl_rl/diag_policy.py` | Instrumented rollout of a trained checkpoint: lift ratio, per-hand wrist error and wrench support | Isaac Lab, needs `--ckpt` |

`train_8h_metrics.csv` is the per-iteration metric table parsed out of the
10 MB raw training log (the log itself is not committed).

`qc_penetration.json` / `qc_penetration_real.json` are the quality-check
results with the default and relaxed `hull_ratio_max` respectively — see
FINDINGS.md B-2 for why the default run reports a misleading pass.

The scripts hardcode container-absolute paths under
`/workspace/video_to_data/robotic_grounding`; run them with
`docker exec <container> python scripts/analysis/<script>.py`.
