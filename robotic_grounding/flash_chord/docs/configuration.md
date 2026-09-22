# Configuration guide

FlashCHORD separates reusable Hydra groups from complete, validated experiment recipes. Start from a recipe and
provide the converted reference sequence explicitly.

The [configuration map](../README.md#configuration) links each YAML group to its implementation.

## Public recipes

| Recipe | Root config | Notes |
| --- | --- | --- |
| `sharpa_ppo` | `train` | floating-hand PPO baseline |
| `sharpa_flash_sac` | `train_flash_sac` | 4,096 worlds, validated 250M-step schedule |
| `dexmate_sharpa_flash_sac` | `train_flash_sac` | 2,048 worlds, calibrated Vega robot model and controller |
| `g1_recon_body_ppo` | `train` | G1+Dex3/SONIC ReconBody PPO |
| `g1_recon_body_flash_sac` | `train_flash_sac` | G1+Dex3/SONIC ReconBody FlashSAC |

```bash
python scripts/train_flash_sac.py \
  experiment=dexmate_sharpa_flash_sac \
  "task.parquet=/data/reference.parquet"
```

The Dexmate recipe uses the current `embodiment=dexmate_sharpa` and `action=residual_joint_position` groups. Its
asset metadata retains the `vega_sharpa_v2` identifier and SHA-256 as provenance for the released 58-DOF robot model.
The released controller includes the calibrated joint parameters and two-physics-step action delay; no system
identification pipeline or raw calibration data is required at runtime.

## Curriculum

`tracking_250m_mixed_reset` preserves the validated 10-stage reduction in Virtual Object Control (VOC) assistance
and adds a terminal stage:

- final-stage boundary: 200,007,680 environment steps;
- total budget: 250,003,456 environment steps;
- first-frame reset probability during the final stage: 0.5.

The remaining probability uses ordinary reference-frame reset sampling.

## Domain randomization

Object scale randomization is configured on the scene:

```bash
scene.object_scale_min=0.8 scene.object_scale_max=1.2 scene.object_scale_seed=7
```

The scale affects object geometry and inertia while preserving configured mass.

## Checkpoints

Every new checkpoint embeds:

- the resolved training configuration;
- the exact policy action and observation layout;
- the object joint and drive settings resolved from the reference;
- the exact FlashSAC critic input layout.

Evaluation and resume reject missing metadata, incomplete or incompatible policy/critic descriptions, and
incompatible object joint or drive settings. This release does not migrate older checkpoint configurations.

## Viewer

Use `viewer.backend=gl` or `viser` interactively, `mp4` for bounded video capture, and `null`/`none` for headless
stepping. For MP4 capture, set `viewer.output_path`, `viewer.num_frames`, `viewer.video_fps`, and optionally the
camera and render-style fields.

## Additional replay and training commands

Use the example paths from the [workflow](../README.md#example-data), inside the FlashCHORD container.

```bash
python scripts/train_rl.py experiment=sharpa_ppo "task.parquet='${SHARPA_PARQUET}'"
python scripts/train_rl.py experiment=g1_recon_body_ppo "task.parquet='${G1_PARQUET}'"

# Kinematic Sharpa replay.
python scripts/debug/replay.py "task.parquet='${SHARPA_PARQUET}'" \
  replay=kinematic action=residual_hand_pose_stiff \
  collision=kinematic embodiment=sharpa_hands_stiff

# Dexmate/Vega physical replay.
PARQUET=/v2d/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=vega_sharpa/data.parquet
python scripts/debug/replay.py "task.parquet='${PARQUET}'" \
  embodiment=dexmate_sharpa action=residual_joint_position \
  collision=manipulation collision.robot_support_collision=true

# Bounded capture: 800 G1 control frames at 50 fps.
python scripts/view_policy.py \
  evaluation.checkpoint=outputs/g1_full/policy_349999104.safetensors \
  viewer.backend=mp4 viewer.output_path=outputs/g1_policy.mp4 \
  viewer.num_frames=800 viewer.video_fps=50
```

- Viewers: `viser`, `gl`, `mp4`, or `null`/`none`; robot/scene debug tools support `viser`, `gl`, and `null`.
- G1 replay with `mp4`/`gl`/`null`: also set `start_paused=false`.
- Overlays: `markers.axes`, `markers.keypoints`, `markers.contacts`, `markers.raw_contacts`; disable with `markers.enabled=false`.
- Object scale: `scene.object_scale_min`, `scene.object_scale_max`, `scene.object_scale_seed`.

## Policy resets

- Default: `evaluation.reset_mode=explicit`. Completed or failed episodes restart at `evaluation.start_frame` (zero), using the same initialization as startup.
- `evaluation.reset_mode=sampled_settled` uses the checkpoint's sampled training resets.
