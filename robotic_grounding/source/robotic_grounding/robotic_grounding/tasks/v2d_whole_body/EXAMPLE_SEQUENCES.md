# Whole-body example sequences (retarget → plan → train)

End-to-end reproduction of the **ReconHand** whole-body pipeline on three
hand-object sequences. Starting from MANO reference motion, each sequence is
retargeted to the Dex3 hands, planned into a whole-body G1 trajectory, and used
to train a SONIC + RL-residual policy in two stages.

Pipeline:
```
MANO reference  ──(1) retarget──▶  Dex3 EE motion  ──(2) plan──▶  G1 whole-body (g1_dex3)
                                                                       │
                                              (3) train stage 1 — no-collision warm-up
                                                     │
                                              (4) train stage 2 — contact grounding (resume)
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

Robot↔object collisions are **off** (support surfaces stay solid); the
virtual-object controller is always on; only hand-keypoint and finger-joint
tracking rewards are active. `--zero-actor` starts residuals at zero so the
policy departs cleanly from the SONIC prior.

```bash
STAGE1_OVERRIDES="\
env.episode_length_s=10.0 \
env.commands.motion.always_reset_to_first_frame=false \
env.events.reset_to_trajectory_frame.params.trajectory_time_index=[0,999999] \
env.commands.motion.initial_virtual_object_control_curriculum_scale=1.0 \
env.commands.motion.voc_reset_scale=1.0 \
env.commands.motion.voc_decay_steps=0 \
env.commands.motion.reset_freeze_steps=50 \
env.commands.motion.reset_shoulder_spread=0.5 \
env.events.setup_collision_groups.params.disable_robot_to_object_collisions=true \
env.events.setup_collision_groups.params.disable_robot_to_fixed_object_collisions=false \
env.actions.joint_pos.residual_scale=0.5 \
env.actions.joint_pos.use_tanh=false \
env.actions.joint_pos.finger_residual=true \
env.actions.joint_pos.finger_residual_scale=0.15 \
env.actions.joint_pos.residual_joint_names=[waist_yaw_joint,waist_roll_joint,waist_pitch_joint,left_shoulder_pitch_joint,left_shoulder_roll_joint,left_shoulder_yaw_joint,left_elbow_joint,left_wrist_roll_joint,left_wrist_pitch_joint,left_wrist_yaw_joint,right_shoulder_pitch_joint,right_shoulder_roll_joint,right_shoulder_yaw_joint,right_elbow_joint,right_wrist_roll_joint,right_wrist_pitch_joint,right_wrist_yaw_joint] \
env.rewards.termination_penalty.weight=-100.0 \
env.rewards.motion_hand_keypoints_gaussian_exp.weight=1.0 \
env.rewards.motion_hand_keypoints_gaussian_exp.params.std=0.1 \
env.rewards.motion_finger_joint_pos_gaussian_exp.weight=1.0 \
env.rewards.motion_finger_joint_pos_gaussian_exp.params.std=1.0 \
env.rewards.motion_object_keypoints_tracking_exp.weight=0.0 \
env.rewards.motion_contact_tracking_gaussian_exp.weight=0.0 \
env.rewards.contact_wrench_support_reward.weight=0.0 \
env.rewards.unintended_contact_penalty.weight=0.0 \
env.rewards.missed_contact_penalty.weight=0.0 \
env.rewards.action_rate.weight=-0.01 \
env.rewards.action_l2.weight=-0.001 \
env.rewards.joint_pos_limit.weight=-0.01 \
env.terminations.anchor_pos_error.params.threshold=0.7 \
env.terminations.anchor_quat_error.params.threshold=1.5 \
env.terminations.hand_wrist_away.params.threshold=0.15 \
agent.algorithm.max_grad_norm=0.1 \
agent.algorithm.entropy_coef=0.0001 \
agent.algorithm.schedule=adaptive \
agent.algorithm.desired_kl=0.02 \
agent.algorithm.learning_rate=5e-4 \
agent.policy.init_noise_std=0.5 \
agent.policy.actor_hidden_dims=[1024,512,256,128] \
agent.policy.critic_hidden_dims=[1024,512,256,128]"

python scripts/rsl_rl/train.py --headless \
  --task SonicG1-ReconHand-EpisodeTimeout-v0 \
  --motion_file arctic/planner_processed/dataset_s09_espressomachine_use_02/g1_dex3 \
  --num_envs 4096 --max_iterations 5000 --zero-actor \
  --video --logger wandb --run_name espresso_stage1 \
  $STAGE1_OVERRIDES
```

Repeat with `--motion_file taco/planner_processed/taco_skim_off__spoon__pan_20230926_011/g1_dex3`
and `taco/planner_processed/taco_empty__cup__bowl_20231006_280/g1_dex3` (and matching `--run_name`).
Stage-1 checkpoints land in `logs/rsl_rl/<experiment>/<timestamp>_<run_name>/model_4999.pt`.

## 4. Train — stage 2 (contact grounding)

Resume from the stage-1 checkpoint, turn robot↔object collisions **on**, decay
the virtual-object controller to zero over a fixed schedule, and enable the
contact rewards (wrench-support + unintended/missed-contact penalties). Stage 2
is stage 1 plus the deltas below (later overrides win, so append them after
`$STAGE1_OVERRIDES`). **`--zero-actor` is omitted** — it would wipe the resumed
actor weights.

```bash
STAGE2_DELTAS="\
env.events.setup_collision_groups.params.disable_robot_to_object_collisions=false \
env.commands.motion.initial_virtual_object_control_curriculum_scale=0.5 \
env.commands.motion.voc_decay_steps=10 \
env.curriculum.fixed_timestep_curriculum.params.num_steps_per_env=24 \
env.curriculum.fixed_timestep_curriculum.params.timestep_schedule=[5000,10000,15000,20000] \
env.curriculum.fixed_timestep_curriculum.params.virtual_object_control_scale_factor=[0.5,0.1,0.02,0.004,0.0] \
env.curriculum.fixed_timestep_curriculum.params.reward_weight_schedules.motion_object_keypoints_tracking_exp=[0.0,0.4,0.48,0.496,0.5] \
env.curriculum.fixed_timestep_curriculum.params.reward_weight_schedules.motion_hand_keypoints_gaussian_exp=[0.10,0.10,0.10,0.10,0.10] \
env.curriculum.fixed_timestep_curriculum.params.reward_weight_schedules.motion_finger_joint_pos_gaussian_exp=[0.10,0.10,0.10,0.10,0.10] \
env.curriculum.fixed_timestep_curriculum.params.reward_weight_schedules.motion_contact_tracking_gaussian_exp=[0.10,0.10,0.10,0.10,0.10] \
env.curriculum.fixed_timestep_curriculum.params.reward_weight_schedules.contact_wrench_support_reward=[5.0,5.0,5.0,5.0,5.0] \
env.curriculum.fixed_timestep_curriculum.params.reward_weight_schedules.unintended_contact_penalty=[-2.5,-2.5,-2.5,-2.5,-2.5] \
env.curriculum.fixed_timestep_curriculum.params.reward_weight_schedules.missed_contact_penalty=[-5.0,-5.0,-5.0,-5.0,-5.0] \
env.rewards.motion_hand_keypoints_gaussian_exp.weight=0.10 \
env.rewards.motion_finger_joint_pos_gaussian_exp.weight=0.10 \
env.rewards.motion_contact_tracking_gaussian_exp.weight=0.10 \
env.rewards.contact_wrench_support_reward.weight=5.0 \
env.rewards.unintended_contact_penalty.weight=-2.5 \
env.rewards.missed_contact_penalty.weight=-5.0 \
env.terminations.object_pos_error.params.threshold=0.15 \
env.terminations.object_quat_error.params.threshold=1.0"

python scripts/rsl_rl/train.py --headless \
  --task SonicG1-ReconHand-EpisodeTimeout-v0 \
  --motion_file arctic/planner_processed/dataset_s09_espressomachine_use_02/g1_dex3 \
  --num_envs 4096 --max_iterations 50000 \
  --video --logger wandb --run_name espresso_stage2 \
  agent.resume=true agent.load_checkpoint=logs/rsl_rl/<experiment>/<stage1_run>/model_4999.pt \
  $STAGE1_OVERRIDES $STAGE2_DELTAS
```

Point `agent.load_checkpoint` at each sequence's own stage-1 checkpoint. The
curriculum's `num_steps_per_env` must match the PPO runner's (24).

## Local smoke

To validate the pipeline end-to-end without a full run, shrink any train command:
`--num_envs 256 --max_iterations 8 --logger tensorboard` (drop `--video`). A pass
shows a non-zero `virtual_object_controller_scale_factor`, a climbing mean
reward, and no override or PhysX-buffer errors. Smoke both a taco (rigid) and the
espresso (articulated) sequence — they exercise different object-load paths.

## Eval

```bash
python scripts/rsl_rl/eval.py --headless \
  --task SonicG1-ReconHand-v0 \
  --motion_file arctic/planner_processed/dataset_s09_espressomachine_use_02/g1_dex3 \
  --num_envs 1 --checkpoint logs/rsl_rl/<experiment>/<stage2_run>/model_<N>.pt
```
