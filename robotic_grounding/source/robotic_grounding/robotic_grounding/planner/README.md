# Whole-Body Planner

> [!NOTE]
> **End-effector → whole-body planner (v0.2).** This version is known to produce foot-skating artifacts. The team is actively working on an updated release that reduces these artifacts and improves end-effector tracking accuracy.

Generates planned whole-body motion from V2D retargeted hand/object trajectories. Takes EE targets as input, runs a learned motion model to produce full-body joint trajectories, and outputs a single Hive-partitioned parquet containing everything needed for RL training.

## Quick Start

```bash
cd /path/to/video_to_data
PYTHONPATH=robotic_grounding/source/robotic_grounding:$PYTHONPATH \
python -m robotic_grounding.planner.g1_planner \
    --v2d_parquet robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic/arctic_processed \
    --v2d_sequence box_grab \
    --robot dex3 \
    --workspace_offset -0.30 0.0 0.07 \
    --output planner_processed
```

This opens a MuJoCo viewer showing the planned motion with EE axes, object mesh, and support surface.

## Pipeline

```
V2D Retargeted Parquet (arctic_processed/)
│
├── Step 1: Nominal FK
│     Compute G1 nominal wrist positions from standing pose
│
├── Step 2: Load V2D Reference
│     Load hand/object trajectories, interpolate to target FPS
│
├── Step 3-4: Transform to G1 Frame
│     Reference frame alignment → yaw correction → position offset
│
├── Step 5: Build Trajectory
│     Hold nominal (5s) → interpolate (5s) → hold start (5s) → reference
│
├── Step 6: Inference
│     MotionInferenceAgent: EE targets → chunked autoregressive → full-body qpos
│
├── Step 7: Build Full Qpos
│     Combine planner body (29 DOF) + reference fingers; legs are
│     planner-predicted unless --fix_lower_body pins them to a crouch
│
├── Step 8: Save Parquet
│     Hive-partitioned: planner_processed/sequence_id=.../robot_name=.../*.parquet
│     Post-write invariants (utils/validation.py) hard-fail on any contract
│     break before the parquet leaves planning.
│
├── Step 8b: Reconstruct Support Surface
│     support_recon writes <output>/reconstructed_stage/<seq>_support.usda.
│     Disks that sit on another body's trajectory (a tool resting on the
│     target) are filtered so they don't spawn intersecting the target body.
│
└── Step 9: Viewer
      MuJoCo playback with EE axes, object mesh, support surface
```

Pre-plan, `utils/validation.warn_reference_issues` runs over the loaded V2D
motion and prints warnings for reference-owned gaps (missing required fields,
unresolvable asset paths, missing URDF mesh dependencies, off-FPS source).
These are informational — the upstream retargeter / asset pipeline owns them.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--v2d_parquet` | required | Path to V2D retargeted parquet folder |
| `--v2d_sequence` | `box_grab` | Sequence ID substring filter |
| `--v2d_robot_name` | `dex3` | Robot name filter |
| `--v2d_trajectory_id` | `0` | Trajectory index within filtered results |
| `--robot` | `dex3` | Robot type. Only `dex3` is supported. |
| `--target_fps` | `150.0` | Resample V2D data to this FPS |
| `--workspace_offset` | `-0.10 0.0 -0.15` | XYZ offset applied to EE targets |
| `--ref_seconds` | `-1` | Seconds of reference to include (`-1` = all) |
| `--output` | `./planner_output` | Output root; parquet lands under `<output>/planner_processed/` |
| `--no_viewer` | false | Skip the MuJoCo viewer |
| `--ik_verify` | false | Run an IK reachability check |
| `--ik_plan` | false | Use the IK solution instead of the learned model |
| *Reference trimming* | | |
| `--v2d_start_frame` | `0` | Drop this many frames from the start of the reference |
| `--v2d_start_at_first_contact` | false | Start the reference at first hand-object contact |
| `--v2d_pre_contact_frames` | `10` | Frames to keep before first contact |
| `--v2d_end_after_last_contact_frames` | `-1` | If `>= 0`, truncate this many frames after last contact |
| *Warmup approach* | | |
| `--hold_start_s` | `5.0` | Seconds holding the nominal pose |
| `--interp_s` | `5.0` | Seconds interpolating from nominal to the reference start |
| `--hold_end_s` | `5.0` | Seconds holding the reference start before playback |
| `--no_approach` | false | Skip the warmup; start directly at the reference |
| *Model behavior* | | |
| `--fix_lower_body` | false | Pin hips/knees/ankles to a static crouch |
| `--fix_root` | `()` | Pin root components, e.g. `z roll pitch` (`x y z roll pitch yaw`) |
| `--no_smooth_qpos` | false | Disable post-inference qpos smoothing |
| `--search_heading_deg` | `0.0` | If `> 0`, sweep heading offsets and pick the lowest wrist error |
| `--heading_align_frame` | `start` | Heading-correction frame (`start` or `first_contact`) |

Run `g1_planner --help` for the full list, including the legacy `--fix_root_*` aliases.

## Output Parquet Schema

The planner writes the unified `motion_v1` schema (see
[../motion_schema/README.md](../motion_schema/README.md) and
[../motion_schema/motion_schema.md](../motion_schema/motion_schema.md)). Hive
layout: `<output>/planner_processed/sequence_id=<seq>/robot_name=<robot>/*.parquet`.

The planner populates:

- Robot state (`robot_root_position`, `robot_root_wxyz`, `robot_joint_positions`,
  `robot_joint_names`) decomposed from the planner's mujoco qpos.
- `ee_link_names` set to `["left_hand_palm_link", "right_hand_palm_link"]`
  for dex3, where the palm is the free-flyer URDF root. `ee_pose_w (T, 2, 7)`
  is built from the reference wrist trajectories.
- Object metadata + trajectory (`object_body_position`, `object_body_wxyz`,
  `object_body_names`, `object_articulation`, mesh/URDF paths copied from the
  upstream ManoSharpaData retarget file).
- `object_root_position` / `object_root_axis_angle` derived from body 0 of the
  planner-frame object pose so the env's articulated scene init lands where the
  trajectory starts.
- `robot_joint_names` / `robot_joint_positions` cover every actuated joint
  (body + fingers) in MuJoCo joint order; the per-side `hand_finger_joints` /
  `hand_finger_joint_names` lists stay populated for callers that want the
  side-segregated view.
- Per-side hand frames + contact groups are transformed by the same per-frame
  rigid transform applied to `ee_pose_w` / `object_body_position`, so every
  field of the output parquet lives in a single coherent planner frame.

Support surfaces are discovered by `SceneConfig.from_motion_file()` from the
sibling `reconstructed_stage/` directory; they are not embedded in the parquet
(previously stored as `support_position` / `support_size`).

## Module Layout

```
planner/
├── g1_planner.py             CLI orchestration; sole consumer of utils/
├── trajectory.py             warmup builder: hold nominal → interp → hold start
│                             → reference
├── visualization.py          MuJoCo viewer with EE axes, object mesh, support
├── motionbricks/             whole-body model backend (ONNX)
│   ├── inference.py          MotionInferenceAgent: EE targets → full-body qpos
│   ├── runtime.py            ONNX engine + chunked autoregressive loop + blend
│   ├── kinematics.py         seed FK, hand-root → wrist offset, transforms → qpos
│   ├── canonicalization.py   frame canonicalization + per-chunk re-framing
│   ├── rotations.py          shared quaternion / 6D rotation conversions
│   ├── smoothing.py          qpos + chunk-seam smoothing
│   └── assets/models/        ONNX graphs (root, pose, decoder) + meta.json,
│                             arrays.npz
└── utils/                    Pure helpers, no planner state
    ├── transforms.py         Quaternion conversions, low-level rigid
    │                         primitives (quat_*, transform_primary_*,
    │                         transform_contact_*_by_part), and the high-level
    │                         transform_reference pipeline (local frame fix →
    │                         heading → position offset → workspace shift)
    ├── motion.py             Resample V2D motion fields to target FPS (linear
    │                         for positions, SLERP for quats, masked interp for
    │                         contacts) + assemble the output motion fields
    ├── qpos.py               MuJoCo qpos assembly (planner body + reference
    │                         fingers, root / lower-body pinning) + wrist EE
    │                         scoring
    └── validation.py         Pre-plan warn_reference_issues + post-plan
                              assert_motion_parquet_invariants, so output-contract
                              regressions surface at planning time, before
                              training sees them.
```

Support-surface reconstruction (still-frame detection → mesh-projected
disks → USDA output, with a phantom-tool cross-body filter) lives in
`retarget/support_recon.py` — shared with the standalone CLI
`scripts/reconstruct_support_surfaces.py` and retargeting viz tools.

The active model backend is `motionbricks/`. `MotionInferenceAgent` in
`motionbricks/inference.py` drives three ONNX graphs (`root`, `pose`,
`decoder`) bundled under `assets/models/`, using only numpy, scipy, and
onnxruntime — the planner has no PyTorch or training-codebase dependency.

## Acknowledgements

The whole-body motion model is built on **MotionBricks**. If you use this
planner, please cite MotionBricks in addition to this project:

```bibtex
@misc{wang2026motionbricksscalablerealtimemotions,
      title={MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives},
      author={Tingwu Wang and Olivier Dionne and Michael De Ruyter and David Minor and Davis Rempe and Kaifeng Zhao and Mathis Petrovich and Ye Yuan and Chenran Li and Zhengyi Luo and Brian Robison and Xavier Blackwell and Bernardo Antoniazzi and Xue Bin Peng and Yuke Zhu and Simon Yuen},
      year={2026},
      eprint={2604.24833},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.24833},
}
```
