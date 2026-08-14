# Whole-body example sequences (retarget → plan → train)

End-to-end reproduction of the **ReconHand** whole-body pipeline on three
hand-object sequences. Starting from MANO reference motion, each sequence is
retargeted to the Dex3 hands, planned into a whole-body G1 trajectory, and used
to train a SONIC + RL-residual policy in three stages.

Pipeline:
```
MANO reference  ──(1) retarget──▶  Dex3 EE motion  ──(2) plan──▶  G1 whole-body (g1_dex3)
                                                                       │
                                              (3) train stage 1 — no-collision warm-up
                                                     │
                                              (4) train stage 2 — contact grounding (resume)
                                                     │
                                              (5) train stage 3 — full-sequence finetune (resume)
```

The planner writes each plan **under the dataset dir** (`<ds>/planner_processed/…`), so
it sits next to `<ds>/object_assets/` and `<ds>/reconstructed_stage/` and trains directly
— no copy/staging step.

For the floating-hand (Dex3-only, no whole body) example set, see
[`docs/EXAMPLE_SEQUENCES.md`](../../../../../docs/EXAMPLE_SEQUENCES.md).

## Sequences

| Key | Dataset | `sequence_id` | Object | Retarget |
|---|---|---|---|---|
| `espresso` | arctic | `dataset_s09_espressomachine_use_02` | espresso machine (articulated) | `arctic_to_dex3.py` |
| `taco_skim` | taco | `taco_skim_off__spoon__pan_20230926_011` | spoon + pan (rigid) | `taco_to_dex3.py` |
| `cup_bowl` | taco | `taco_empty__cup__bowl_20231006_280` | cup + bowl (rigid) | `taco_to_dex3.py` |

## Prerequisites

- MANO models and the **loaded** sources for these sequences, laid out under your
  host data root `$HMD` (see [`SETUP.md`](../../../../../docs/SETUP.md),
  [`ARCTIC_SETUP.md`](../../../../../docs/ARCTIC_SETUP.md),
  [`TACO_SETUP.md`](../../../../../docs/TACO_SETUP.md)):
  ```
  $HMD/arctic/arctic_loaded/sequence_id=dataset_s09_espressomachine_use_02/
  $HMD/taco/taco_loaded/sequence_id=taco_skim_off__spoon__pan_20230926_011/
  $HMD/taco/taco_loaded/sequence_id=taco_empty__cup__bowl_20231006_280/
  ```
  Stage 2's contact rewards need hand-object contact normals in the loaded source
  (`mano_{left,right}_link_contact_normals`); the loader populates these.
- The articulated espresso-machine URDF resolved from the dataset's
  `object_assets/` (arctic objects are articulated — URDFs come from ArtiGrasp,
  see [`ARCTIC_SETUP.md`](../../../../../docs/ARCTIC_SETUP.md)).

## Setup

Start the training container with your data root mounted — `run.sh` bind-mounts
host `$HUMAN_MOTION_DATA_DIR` at `…/assets/human_motion_data` inside the container
(build it once first with `./workflow/run.sh build`):
```bash
HMD=~/datasets/human_motion_data
HUMAN_MOTION_DATA_DIR=$HMD ./workflow/run.sh start   # re-enter a running one with: ./workflow/run.sh shell
```
All steps below run **inside** that container from `robotic_grounding/`, with paths
relative to the mounted data root:
```bash
DATA=source/robotic_grounding/robotic_grounding/assets/human_motion_data
```
So `$DATA/arctic/…` is your host `$HMD/arctic/…`, and a relative `--motion_file`
(e.g. `arctic/planner_processed/<seq>/g1_dex3`) resolves under the same root.

## 1. Retarget (MANO → Dex3)

Writes one parquet per sequence to `<output_dir>/sequence_id=<seq>/robot_name=dex3/`.
Retarget scale defaults to 1.0 for arctic and 1.2 for taco.

```bash
# espresso (arctic)
python scripts/retarget/arctic_to_dex3.py \
  --input_dir  $DATA/arctic/arctic_loaded \
  --output_dir $DATA/arctic/arctic_dex3 \
  --sequence_id dataset_s09_espressomachine_use_02 --device cuda:0 --save

# taco_skim + cup_bowl (taco)
python scripts/retarget/taco_to_dex3.py \
  --input_dir  $DATA/taco/taco_loaded \
  --output_dir $DATA/taco/taco_dex3 \
  --sequence_id taco_skim_off__spoon__pan_20230926_011 --device cuda:0 --save
python scripts/retarget/taco_to_dex3.py \
  --input_dir  $DATA/taco/taco_loaded \
  --output_dir $DATA/taco/taco_dex3 \
  --sequence_id taco_empty__cup__bowl_20231006_280 --device cuda:0 --save
```

## 2. Plan (Dex3 EE → G1 whole-body)

The planner starts the reference at first hand-object contact and generates a
100 fps whole-body G1 trajectory. Planner flags are tuned per sequence — the two
taco clips use the contact-gated defaults; espresso keeps a longer contact window
and a workspace offset with heading aligned at first contact.

```bash
# cup_bowl
python -m robotic_grounding.planner.g1_planner --robot dex3 \
  --v2d_parquet $DATA/taco/taco_dex3 --v2d_robot_name dex3 \
  --v2d_sequence taco_empty__cup__bowl_20231006_280 \
  --v2d_start_at_first_contact --v2d_pre_contact_frames 10 --target_fps 100 \
  --output $DATA/taco --no_viewer

# taco_skim
python -m robotic_grounding.planner.g1_planner --robot dex3 \
  --v2d_parquet $DATA/taco/taco_dex3 --v2d_robot_name dex3 \
  --v2d_sequence taco_skim_off__spoon__pan_20230926_011 \
  --v2d_start_at_first_contact --v2d_pre_contact_frames 10 --target_fps 100 \
  --output $DATA/taco --no_viewer

# espresso
python -m robotic_grounding.planner.g1_planner --robot dex3 \
  --v2d_parquet $DATA/arctic/arctic_dex3 --v2d_robot_name dex3 \
  --v2d_sequence dataset_s09_espressomachine_use_02 \
  --v2d_start_at_first_contact --v2d_pre_contact_frames 23 \
  --v2d_end_after_last_contact_frames 7 --target_fps 100 \
  --workspace_offset -0.10 0.0 -0.05 --heading_align_frame first_contact \
  --output $DATA/arctic --no_viewer
```

`--output $DATA/<ds>` puts each plan and its support under the dataset dir:
`<ds>/planner_processed/sequence_id=<seq>/robot_name=g1_dex3/<hash>.parquet` and
`<ds>/reconstructed_stage/<seq>_support.usda`. Because they sit under `<ds>/`, the env
resolves the object assets (`<ds>/object_assets/`) and the support USD directly — so
training reads the plan in place, with no copy/staging step.

`<ds>/reconstructed_stage/` is shared with the floating-hand pipeline's support USDs
(same `<seq>_support.usda` name, no robot qualifier), so a sequence planned by both
flows overwrites the other. The example set overlaps only on the taco clips.

Preview a plan before training (headless viser on `localhost:8080`):
```bash
python scripts/replay_motion_viser.py \
  --motion_file $DATA/arctic/planner_processed/sequence_id=dataset_s09_espressomachine_use_02/robot_name=g1_dex3 \
  --port 8080 --start-paused
```

## 3. Train — stage 1 (no-collision warm-up)

`SonicG1-ReconHand-Stage1-v0` bakes the warm-up recipe: robot↔object collisions
off (support surfaces stay solid), VOC always on, hand-keypoint + finger-joint
tracking rewards only, upper-body residuals. `--zero-actor` starts residuals at
zero so the policy departs cleanly from the SONIC prior.

```bash
python scripts/rsl_rl/train.py --headless \
  --task SonicG1-ReconHand-Stage1-v0 \
  --motion_file arctic/planner_processed/dataset_s09_espressomachine_use_02/g1_dex3 \
  --num_envs 4096 --zero-actor \
  --video --logger tensorboard --run_name espresso_stage1
```

Repeat with `--motion_file taco/planner_processed/taco_skim_off__spoon__pan_20230926_011/g1_dex3`
and `taco/planner_processed/taco_empty__cup__bowl_20231006_280/g1_dex3` (and matching `--run_name`).
Stage-1 checkpoints land in `logs/rsl_rl/<experiment>/<timestamp>_<run_name>/model_4999.pt`.

## 4. Train — stage 2 (contact grounding)

`SonicG1-ReconHand-Stage2-v0` turns robot↔object collisions **on**, decays the
virtual-object controller to zero over a fixed-timestep curriculum, and enables
the contact rewards (wrench-support + unintended/missed-contact). Resume from the
stage-1 checkpoint; **omit `--zero-actor`** — it would wipe the resumed actor.

```bash
python scripts/rsl_rl/train.py --headless \
  --task SonicG1-ReconHand-Stage2-v0 \
  --motion_file arctic/planner_processed/dataset_s09_espressomachine_use_02/g1_dex3 \
  --num_envs 4096 \
  --video --logger tensorboard --run_name espresso_stage2 \
  agent.resume=true agent.load_checkpoint=logs/rsl_rl/<experiment>/<stage1_run>/model_4999.pt
```

Point `agent.load_checkpoint` at each sequence's own stage-1 checkpoint.

## 5. Train — stage 3 (full-sequence finetune)

`SonicG1-ReconHand-Stage3-v0` finetunes on the full trajectory: full-length
episodes (trajectory-end timeout), VOC off, curriculum disabled, object-keypoint
tracking weighted 10×. Resume from the stage-2 checkpoint.

```bash
python scripts/rsl_rl/train.py --headless \
  --task SonicG1-ReconHand-Stage3-v0 \
  --motion_file arctic/planner_processed/dataset_s09_espressomachine_use_02/g1_dex3 \
  --num_envs 4096 \
  --video --logger tensorboard --run_name espresso_stage3 \
  agent.resume=true agent.load_checkpoint=logs/rsl_rl/<experiment>/<stage2_run>/model_<N>.pt
```

Each stage's iteration budget is baked into its task (stage 1/3: 5000,
stage 2: 50000); pass `--max_iterations` to override.

## Local smoke

To validate the pipeline end-to-end without a full run, shrink any train command:
`--num_envs 256 --max_iterations 8 --logger tensorboard` (drop `--video`). A pass
shows a non-zero `virtual_object_controller_scale_factor`, a climbing mean
reward, and no missing-asset or PhysX-buffer errors. Smoke both a taco (rigid) and the
espresso (articulated) sequence — they exercise different object-load paths.

## Eval

Eval with the base task `SonicG1-ReconHand-v0`. The `-Stage{1,2,3}-v0` tasks
bake training-time deltas that skew an eval — stage 1 disables robot↔object
collisions outright, and stage 2 leaves the virtual object controller partially
driving the object (its decay curriculum doesn't run under eval) — while the
base task shares their observation/action spaces, so any stage's checkpoint
loads there. Use a stage-2 or stage-3 checkpoint (stage-1 checkpoints were
trained without contact):

```bash
python scripts/rsl_rl/eval.py \
  --task SonicG1-ReconHand-v0 \
  --motion_file arctic/planner_processed/dataset_s09_espressomachine_use_02/g1_dex3 \
  --num_envs 1 --checkpoint logs/rsl_rl/<experiment>/<stage2_run>/model_<N>.pt
```

No checkpoint trained yet? Download the example checkpoints
(`checkpoints/recon_hand/*.pt` covers this espresso sequence and the taco
empty-cup-bowl sequence) — see
[SETUP.md §10](../../../../../docs/SETUP.md#10-pretrained-rl-checkpoints-optional-eval-without-training).
