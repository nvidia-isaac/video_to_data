# Why the tissue-box run trained for 8 hours and learned nothing

An 8-hour PPO run (4079 iterations, 4096 envs, 401 M env steps) on a monocular
ego-video sequence produced a policy that raises the object **8.9 mm** where the
reference raises it **154 mm** — a lift ratio of **0.058**. Every metric in the
training log looked healthy: 95.8% of episodes ran to the end of the reference,
object position error fell from 11.3 cm to 3.7 cm, and mean reward rose from 50
to 354.

None of those numbers could distinguish success from failure on this clip. What
follows is the list of things in the default implementation that caused that,
each with the measurement that establishes it.

The root cause is not a bug. It is that several constants in the environment are
calibrated for mocap sequences where the object travels tens of centimetres,
while this clip's entire motion is 15.4 cm.

---

## A. Absolute constants fixed at mocap scale — primary cause

### A-1. `object_keypoints_tracking_exp.var = 0.1`

**Default** — `tasks/v2d/v2d_hand_env_cfg.py:219-226` hardcodes `"var": 0.1`.
The reward is `exp(-||Δkeypoint||² / var)`, so this declares an error resolution
of `sqrt(0.1) ≈ 0.32 m`.

**Why it is wrong here** — the object's entire travel is 0.154 m, so the
reward is coarser than the task. A policy that freezes the object at frame 0
scores **0.9326** where perfect tracking scores 1.0, leaving a **6.7% reward
band for the whole task**. In return terms that is 22 points, against the
~147 points of remaining reward stream forfeited by an early termination.
Standing still is the rational choice under this objective.

**Fix** — config only, no code change:

```
env.rewards.object_keypoints_tracking_exp.params.var=0.02
```

| var | do-nothing score | task reward band |
|---|---|---|
| 0.1 (default) | 0.9326 | 6.7% |
| **0.02** | 0.7548 | **24.5%** |
| 0.01 | 0.6377 | 36.2% |

Below 0.005 the reward goes too flat early in training; **0.02–0.05** is the
safe range. The principled fix is to derive `var` from the reference's own
excursion rather than ship one default for every sequence.

Measured by `rew_headroom.py`, `rew_sweep.py`.

### A-2. `object_away_from_trajectory` thresholds 0.2 m / 0.7 rad

**Default** — `tasks/v2d/v2d_hand_env_cfg.py:311-318`. The term compares the
object's actual pose against what the reference prescribes *at the current
timestep*.

**Why it is wrong here** — for a policy that leaves the object on the table,
the error it feeds the term is just the reference's own displacement:

```
position  max 0.1555 m   (threshold 0.20 m, 22% margin)
rotation  max 0.0965 rad (threshold 0.70 rad, 86% margin)
```

**Neither condition can ever fire.** Never lifting the object is, by this
criterion, following the trajectory. That is why `time_out` reached 95.8% and
was misread as a success rate.

**Fix** — derive the threshold from the reference's excursion. The repo already
contains a ratio-based variant for hands,
`tasks/v2d/mdp/terminations.py:hand_to_object_away_from_trajectory`, which is
not wired into the config. Applying the same idea, or computing
`position_threshold = max(0.05, 0.5 × reference_max_excursion)` at load time,
gives 0.078 m for this clip and terminates a non-lifting policy at frame 142.

**Blocked on B-1** — do not lower this first. Reset transients already reach
0.197 m under pure reference replay (see B-1), so a lower threshold would kill
legitimate episodes.

Measured by `objaway_margin.py`, `lift_timing.py`.

### A-3. `KEYPOINT_VECS` — verified, do not touch

`tasks/v2d/mdp/commands/hand_object_commands.py:290-304` places the six object
keypoints at **1 m** from the object centre regardless of object size. This
looks like the obvious culprit for a 20 cm box, but shrinking the lever makes
the reward band **worse**, not better (6.7% → 5.4% at 0.2 m). Most of the 6.7%
comes from the small orientation change being amplified by the long lever;
remove it and only the position error is left, which discriminates even less.
Do not spend effort here.

Measured by `rew_sweep.py`.

---

## B. Reference data does not pass the paper's quality gate — secondary cause

### B-1. Retargeted sequence penetrates the object by 2.19 cm

**Default** — the paper filters out sequences whose hand-object penetration
exceeds 2 cm before training. `scripts/data_quality_checks/` implements this,
but the pipeline does not enforce it ahead of training.

**Why it matters** — the sequence used for the 8-hour run scores **2.188 cm**
and would have been rejected (left hand, around frame 269; right hand peaks at
1.48 cm). This propagates: every reset teleports the hands into a pose that is
inside the box, the contact solver ejects them, and **`object_away` fires 0.36%
of the time even with zero actions**, with deviations reaching 0.197 m — 98% of
the termination threshold.

**Fix**
1. Run `scripts/data_assessor.py --checks hand_penetration,dummy_agent_success`
   right after retargeting and refuse to start training if it fails. Two
   minutes before spending eight hours.
2. Reduce the penetration itself: more gsplat refinement (which moved it
   2.11 → 1.25 cm on the right hand), a penetration penalty in the retargeting
   IK, or a watertight mesh so the refinement loss sees the real surface.

Measured by `pen_per_hand.py`, `diag_voc.py`.

### B-2. `hull_volume_ratio` gate misclassifies non-watertight meshes, and reports the skip as a pass

**Default** — `scripts/filter_penetrations.py:752` sets `hull_ratio_max = 3.0`.
When `hull.volume / mesh.volume` exceeds it, the object is treated as concave
(AR glasses, mugs, open vases) and the hand-object check is skipped entirely.

**Why it is wrong here** — the SAM3D reconstructed mesh is not watertight, so
`trimesh` computes a volume about 1/6 of the true one (0.000367 vs
0.00234 m³). The ratio comes out at 6.284 and **a convex rectangular tissue box
is classified as concave**. The gate assumes the clean scanned meshes of a mocap
dataset; a monocular reconstruction does not satisfy that assumption.

**Worse** — the skipped check is reported as `{"pass": True, "score": 0.0}`.
The `"no frames"` path at `filter_penetrations.py:770` does the same. **"Not
measured" is indistinguishable from "passed"**; the first run of this check
returned `pass 100%, score 0.0000` and was nearly taken at face value.

**Fix**
1. Surface the skip reason in the result, e.g.
   `{"pass": None, "reason": "skipped: hull_ratio 6.28 > 3.0"}`. This is the
   more urgent half — it lets a human catch the problem even without fix 2.
2. Normalise the mesh before judging convexity (`trimesh.repair`, or use the
   convex hull as the collision proxy directly).

---

## C. Nothing observes success

### C-1. No lift metric

**Default** — the environment has no termination term and no metric that asks
whether the object was lifted. The log carries tracking errors and termination
rates only.

**Why it matters** — there was no way to know the run had failed until the
video was watched eight hours later. The number that makes it obvious,
lift ratio 0.058, only exists because a diagnostic script was written after the
fact.

**Fix** — add it as a **metric, not a reward**, so the reproduction stays
faithful. One line in the command term's `_update_metrics`:

```python
self.metrics["object_lift_ratio"] = (
    (self.object_position_e[:, 0, 2] - reset_object_z)
    / (reference_z_range + 1e-6)
).clamp(0, 2)
```

### C-2. `Episode_Termination/time_out` reads as a success rate

**Default** — Isaac Lab's `TerminationManager.reset()` logs
`_last_episode_dones.float().mean(dim=0)`: the fraction of **environments**
whose most recent episode ended with each cause. The statistic itself is
correct and, being per-environment rather than per-episode, is not biased by
short failing episodes resetting more often.

**Why it misleads** — in an environment with no success criterion, `time_out`
is the only number that looks like things are working, so a reader takes it for
a success rate. Its actual meaning is "was not terminated early".

**Fix** — C-1 resolves this. Separately, do not label it "completion rate" in
reports.

---

## D. Training configuration

### D-1. `wrist_position_clip` equals the wrist termination threshold

`tasks/v2d/mdp/actions/actions_cfg.py:48` sets `wrist_position_clip = 0.2`, and
`v2d_hand_env_cfg.py:303-309` sets
`hand_wrist_away_from_trajectory.threshold = 0.2`. The policy's maximum
commandable wrist residual is exactly the termination boundary, and it used it:
the final right-wrist error sat at **0.171 m**, just inside. Keep the clip
below the threshold (e.g. 0.12) so that "died from its own command" and "died
from a physics accident" stay distinguishable.

### D-2. `force_closure_reward` has no per-hand floor

`tasks/v2d/mdp/rewards.py:627-676` averages over reference-active hands, so
abandoning one hand still collects half the reward. The run split: right-hand
wrench support **0.493**, left **0.743**, with the right wrist hovering 4.5 cm
further from the box than it should.

This is emergent symmetry breaking, not a structural bias. Ruled out:
right/left robot configs are identical except the URDF path; the reference is
symmetric (wrist-object distance 0.209 vs 0.210 m, path length 0.604 vs
0.639 m, contact frames 218 vs 230); and the **left** hand is the one with
deeper penetration (2.19 vs 1.48 cm), so that cannot explain a worse right
hand. The asymmetry also flips mid-training — at iteration 1000 the left finger
error was 1.035 against the right's 0.220.

**Fix** — require a minimum per-hand support, or combine the hands
multiplicatively (`sqrt(left × right)`). This changes a reward function, so
defer it behind A-1 and B-1, and first confirm the symmetry break reproduces
under a different seed.

### D-3. `reset_to_first_frame_prob` is gated on VOC — low priority, NOT the cause

`hand_object_commands.py:1907-1920` only applies the frame-0 reset once
`virtual_object_controller_scale_factor < 0.1`, which in this run meant from
iteration 2105, giving 4.8% frame-0 exposure overall against an eval that
always starts at frame 0.

**This was initially blamed and then refuted.** Replaying the same checkpoint
from random start frames:

| | start at frame 0 | random start frame |
|---|---|---|
| mean z(actual) − z(reference) | −4.47 cm | −4.92 cm |
| steps more than 3 cm below reference | 42.9% | **99.4%** |

71% of training episodes began with the object already in the air, and the
policy still failed to hold it there. Failure is uniform across the state
distribution, which points at the objective, not at coverage. Ungating this is
a cheap and harmless change, but on its own it changes nothing.

Measured by `../rsl_rl/diag_policy.py --random_start`.

### D-4. Curriculum compression — secondary

The schedule was compressed by 0.2632 to fit 4079 iterations, leaving **395**
iterations after VOC reached 0 against **6000** in the stock schedule. But the
weight-normalised rewards plateau at iteration ~900 and stay flat within each
curriculum stage, so there is little evidence more iterations would have helped.
Re-evaluate after A and B.

### D-5. Do not raise the iteration budget

The stock schedule implies 20,000 iterations (≈39 h here) and there is no
evidence for it. `force_closure` peaks at 0.690 around iteration 900 and drifts
down to 0.650 over the following 3,100 iterations; the hand tracking terms start
at 0.985 and only decline. Fix A-1, B-1 and C-1, then run **1,000–1,500
iterations** and check whether the lift ratio moves at all.

To recover these numbers from the log, note that `Episode_Reward/<term>` is
`sum(func × weight × dt) / max_episode_length_s` with `dt = 0.05` and
`max_episode_length_s = 26.0`. Undo it with
`Episode_Reward × 26 / (weight × mean_episode_length × 0.05)` to get the raw
0..1 term value. `train_8h_metrics.csv` carries the raw columns.

---

## E. Repository bugs found along the way, unrelated to this experiment

| where | problem | fix |
|---|---|---|
| `scripts/rsl_rl/eval.py` | does not clamp `viewer.env_index = 6`, so `--num_envs < 7` always fails | port the two lines from `dummy_agent.py:204-207` |
| `reconstruction/modules/v2d_pipelines/run_ego_wilor.py` | `_step("Package result/", ...)` skips when `result/` exists, so **gsplat-refined poses never reach the bundle** | re-package when refinement ran |
| `reconstruction/modules/v2d_task_library_loader/lib/level_result_bundle.py` | axis evaluation and application each run the same 90k-rotation brute force (4 min × 2) | cache the search, or build `--up_rank` comparison in |

---

## Suggested order

1. **B-2** (surface the skip reason) — cheapest, and without it you cannot
   verify anything else.
2. **B-1** (penetration below 2 cm) — prerequisite for A-2.
3. **C-1** (log lift ratio) — without it you cannot tell whether a change helped.
4. **A-1** (`var` 0.1 → 0.02) — one config value, largest expected effect.
5. A 1,000-iteration run — does the lift ratio move?
6. **A-2** (thresholds), then **D-1**, **D-2**, **D-3**.

Steps 1–4 are configuration changes and added observability only, so the run
remains a faithful CHORD reproduction. D-2 is the sole reward-function change
and it can wait.
