---
name: robotic_grounding_whole_body
description: Pointer skill for the robotic_grounding G1 WHOLE-BODY humanoid pipeline (SONIC controller + RL residuals). Orients the user on the two reference-motion sources and the SonicG1 Isaac Lab tasks, then points at the authoritative source READMEs for exact commands. Use this skill whenever the user works with the G1 humanoid or whole-body motion: "retarget SOMA to G1", "run the whole-body planner", "SonicG1", "ReconBody / ReconHand", "train the G1 whole-body policy", "g1_planner", "whole-body eval", or mentions the G1 robot / SONIC controller. For the floating-hand (Sharpa Wave / Dex3) pipeline use robotic_grounding_run; for failures use robotic_grounding_doctor.
---

# robotic_grounding — Whole-Body (G1 / SONIC) Pointer

The G1 whole-body humanoid pipeline (SONIC controller + RL residuals) is separate from the
floating-hand (Sharpa/Dex3) pipeline and is **only lightly covered in this release**. This skill
orients you and hands off to the authoritative source docs for exact commands.

**Authoritative references (read these for commands):**
- Tasks / environments / training / eval:
  [`source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/README.md`](../../../robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/README.md)
- EE → whole-body planner:
  [`source/robotic_grounding/robotic_grounding/planner/README.md`](../../../robotic_grounding/source/robotic_grounding/robotic_grounding/planner/README.md)

For the floating-hand pipeline, use **`robotic_grounding_run`** instead.

## Two sources of G1 reference motion

Ask which the user has / wants — it determines the task and the docs to open:

| Source | Entry point | Feeds task | Notes |
|--------|-------------|------------|-------|
| **SOMA → G1** (body-accurate, mocap/video reconstruction) | `scripts/retarget/soma_to_g1.py` | `SonicG1-ReconBody-v0` | Direct whole-body retarget |
| **EE → whole-body planner** (from V2D floating-hand output) | `python -m robotic_grounding.planner.g1_planner` | `SonicG1-ReconHand-v0` | Learned motion model; **v0.2 has known foot-skating artifacts** |

## Registered G1 tasks

| Task ID | Reference source | Notes |
|---------|-----------------|-------|
| `SonicG1-v0` | — | Base env; reward weights zeroed. Scaffold for custom reward configs, not train-ready as-is |
| `SonicG1-ReconBody-v0` | SOMA→G1 | Body-accurate tracking |
| `SonicG1-ReconHand-v0` | planner | Hand-accurate tracking |
| `SonicG1-ReconHand-EpisodeTimeout-v0` | planner | `ReconHand` with a fixed episode-length timeout |

## How to help

- Confirm the reference source and matching task from the tables above.
- For exact retarget/plan/replay/support/train/eval commands, open the source README that matches the
  stage (task README for RL + eval; planner README for the EE-planner). Don't invent commands here —
  those READMEs are the source of truth.
- Same host/container split as the rest of the repo: whole-body scripts run **inside** the
  `robotic-grounding` container (wrap from the host with `./workflow/run.sh exec latest 0 -- <cmd>`).
- RL uses the shipped `scripts/rsl_rl/train.py` / `eval.py` with a `SonicG1-*` `--task`. The task
  README also shows an `experiments/run_experiment.py` runner, but `experiments/` is **not shipped in
  this release**.
- On failure, hand off to **`robotic_grounding_doctor`** (its Isaac/GPU/asset sections apply to G1 too).
