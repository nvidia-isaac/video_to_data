# Visual inpainting investigation

This directory contains the reproducible experiment harness for replacing the
human arms in TACO egocentric RGB videos with a kinematically valid Dexmate
Vega robot and Sharpa hands.

The first experiment keeps the human-removal backend fixed to E2FGVI and varies
only the source of the hand trajectory:

1. Phantom HaMeR tracking
2. Video2Data WiLoR reconstruction
3. TACO motion-capture ground truth

Each tracker must emit the same versioned tracking and robot-trajectory
contracts. The initial controlled baseline uses one visually reviewed SAM2 arm
mask per clip for every tracker, then keeps E2FGVI and robot rendering
identical. This makes the comparison attributable to tracking rather than
hidden stage changes. Tracker-specific mask generation can be evaluated as a
separate ablation. See
[`CONTRACTS.md`](CONTRACTS.md) for the on-disk schemas.

A separate MECKA-to-Panda pipeline lives alongside this harness and carries its
own schemas under `inpainting/mecka_panda/`. See
[`MECKA_PANDA.md`](MECKA_PANDA.md).

## Prepare the demo set

Run from the repository root:

```bash
python3 -m inpainting.prepare_experiment \
  --config inpainting/configs/taco_demo_set.json \
  --output inpainting/artifacts/runs/taco_hand_tracking_v1/manifest.resolved.json
```

This is a read-only inventory step. It verifies the RGB/motion/camera frame
counts, resolves the exact source paths, reports any missing inputs, and writes
a resolved manifest for later stages. It does not copy the TACO RGB tree.

## Run layout

The current controlled-baseline layout is:

```text
artifacts/runs/<experiment>/<sequence>/
  shared_arm_mask/arm_mask.npy
  shared_arm_mask/mask_preview.mp4
  shared_inpaint/e2fgvi_960.mp4
  <tracker>/tracking/tracking.npz
  <tracker>/tracking/robot_trajectory.npz
  <tracker>/robot_render/robot_rgb.mp4
  <tracker>/robot_render/robot_mask.npy
  <tracker>/robot_render/robot_depth.npy
  ground_truth/object_render/object_mask.npy
  ground_truth/object_render/object_depth.npy
  ground_truth/object_render/object_render_metadata.json
  <tracker>/final_overlay.mp4
  <tracker>/final_comparison_grid.mp4
```

Generated artifacts are ignored by git. Source code, configs, tests, and the
investigation tracker are versioned.

## External Phantom repository boundary

The sibling Phantom checkout is a read-only reference, not a Video2Data
runtime dependency. This branch does not add Phantom as a git submodule,
symlink to it, import Python from its host path, or copy its model weights into
git. The versioned boundary is:

- `inpainting/phantom_tracker/`: Video2Data-owned acquisition, identity,
  geometry, provenance, and common-contract adapter code.
- `inpainting/phantom_tracker/Dockerfile`: fetches the official
  `MarionLepert/phantom-hamer` repository at the exact HaMeR submodule commit
  recorded by the reviewed parent Phantom revision.
- `inpainting/e2fgvi/docker/Dockerfile`: independently fetches
  `MarionLepert/phantom-E2FGVI` at its pinned release commit.
- `inpainting/artifacts/`: ignored checkpoints, model caches, tracking outputs,
  and videos.
- Licensed MANO files: supplied from a separate sibling directory and mounted
  read-only; they are never copied under this repository.

The parent Phantom commit is retained as provenance. Of the upstream Phantom
repositories, only the executed HaMeR and E2FGVI revisions enter their
respective images. Image builds and the explicit model-acquisition step may use
the network; inference runs with `--network none`. Both stages emit Video2Data
contracts and hashed sidecars, so every downstream retarget, render, and
evaluation consumes local `tracking.npz` or E2FGVI video artifacts without
knowing where Phantom was checked out on the host. See
[`phantom_tracker/README.md`](phantom_tracker/README.md) for the exact commit
mapping and reproduction commands.

## Camera calibration

TACO motion capture and the retargeted Sharpa trajectories are expressed in the
TACO world frame. Exact projection into the head-mounted RGB stream requires
the matching per-sequence `egocentric_intrinsic.txt` and
`egocentric_frame_extrinsic.npy` files. The official files for the selected
demo set are stored under the ignored
`inpainting/artifacts/source_data/Egocentric_Camera_Parameters/` tree and are
resolved by the experiment manifest. Projected GT hand skeleton previews have
been visually checked against all three RGB clips. The preparation and render
stages fail when calibration is absent or inconsistent; they never substitute
Phantom's unrelated fixed camera calibration.

## Object-aware robot compositing

The current controlled comparison uses an oracle object-occlusion pass. Learned
hand tracking does not consume GT hand tracks, but compositing reuses TACO GT
tool/target meshes, per-frame object poses, and official camera metadata.
Consequently, the tracker comparison is controlled but is not a pure-RGB
end-to-end system.

The processed TACO rows also contain frame-aligned world poses for the tool and
target object. Their `_cm.obj` meshes are rendered at a fixed 0.01 scale into a
metric depth pass using the same official camera. The container wrapper is
offline and prints its Docker command by default; `--execute` starts it and a
full EGL render additionally requires an explicit `--gpu`:

```bash
python3 -m inpainting.taco_object_depth_container \
  --sequence-id <taco_sequence_id> \
  --parquet /path/to/sequence_id=<taco_sequence_id>/robot_name=sharpa_wave/data.parquet \
  --source-video /path/to/color.mp4 \
  --intrinsics /path/to/egocentric_intrinsic.txt \
  --world-to-camera /path/to/egocentric_frame_extrinsic.npy \
  --mesh-root /path/to/assets/meshes/taco \
  --output-dir /path/to/object_render \
  --gpu 0 --execute
```

The explicit sequence ID must match the parquet's host
`sequence_id=<id>` Hive partition. The wrapper verifies that match before the
single file is bind-mounted, because the flattened container path no longer
carries PyArrow's virtual partition column.

The wrapper resolves the requested renderer image once and executes its
immutable image ID. A completed object-depth sidecar records both that ID and
the requested reference, canonical host and container paths plus SHA-256 for
the motion parquet, source video, and camera files, and the exact mounted
implementation sources. Batch resume rehashes those records before accepting
the object bundle as current.

The compositor always requires the completed robot-render sidecar. Supplying
the complete object trio enables depth-aware object occlusion; omitting it
preserves hard-mask behavior. Existing outputs are replaced only with an
explicit `--overwrite`:

```bash
python3 -m inpainting.composite_robot \
  --base-video /path/to/e2fgvi.mp4 \
  --robot-video /path/to/robot_rgb.mp4 \
  --robot-mask /path/to/robot_mask.npy \
  --robot-metadata /path/to/render_metadata.json \
  --object-mask /path/to/object_mask.npy \
  --object-depth /path/to/object_depth.npy \
  --object-metadata /path/to/object_render_metadata.json \
  --output-video /path/to/final_overlay.mp4
```

The production batch requires a valid object-depth bundle by default. Its
`--allow-hard-composite` option is an explicitly degraded/debug fallback and
labels the resulting plan accordingly.

The completed sequence-`105` ablation also exercises two RGB-only upstream
replacements. One restricts MoGe metric camera-z to SAM2 tool/target masks. The
other reconstructs SAM3D meshes, estimates their FoundationPose trajectories,
and renders those meshes into metric camera-z depth so surfaces hidden by the
original hand can still participate in occlusion. Raw dense source depth is
not used unchanged because the original human arm would then occlude the
replacement robot.

### Sequence-105 object-compositing comparison

The ablation fixes the same E2FGVI base, Video2Data robot render, 3 mm depth
guard, and both the knife and cutting-board/plate objects across all three
conditions. Conditions 2 and 3 use only RGB upstream: two human-provided boxes
on frame 0 initialize the two SAM2 object tracks, MoGe estimates camera
intrinsics/depth, and the SAM3D + FoundationPose condition uses the canonical
smoothed poses. These boxes are prompts, not GT object masks. GT object
meshes, poses, masks, depth, and camera calibration are absent from both
estimated pipelines; GT is used only for condition 1 and post-hoc evaluation.

| Occluder condition | Visible IoU | Decision errors | Mask IoU | Depth MAE | False visible | False occluded | Temporal disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT mesh + GT pose | 1.0 | 0 (0%) | 1.0 | 0 m | 0 | 0 | 0% |
| RGB SAM2 + MoGe | 0.9759203532 | 1,325,469 (2.38893297%) | 0.6858006999 | 0.1762251283 m | 1,190,276 (73.0769% of GT-occluded pixels) | 135,193 | 0.998360% |
| SAM3D + smoothed FoundationPose | 0.9798924473 | 1,100,569 (1.98358888%) | 0.7109262537 | 0.1752361543 m | 879,185 (53.9775% of GT-occluded pixels) | 221,384 | 1.089252% |

The artifact root is:

```text
/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/runs/taco_hand_tracking_v1/taco_cut__knife__plate_20231013_105/object_compositing_v1
```

The synchronized review video is `object_compositing_3way_105.mp4`; its review
sheets are `object_compositing_3way_105_contact.jpg` and
`object_compositing_3way_105_error_frames.jpg`. The verified metric reports are
`ground_truth/evaluation_vs_gt.json`,
`rgb_estimated_depth_verified/evaluation_vs_gt.json`, and
`v2d_estimated_object/evaluation_vs_gt.json` under that root.

## Reproduce or resume the GT batch

The default invocation is plan-only and does not start containers or write
artifacts:

```bash
python3 -m inpainting.run_ground_truth_batch \
  --manifest inpainting/artifacts/runs/taco_hand_tracking_v1/manifest.resolved.json
```

Review that JSON plan, then explicitly provide both execution authorization
and a GPU selector to run pending stages:

```bash
python3 -m inpainting.run_ground_truth_batch \
  --manifest inpainting/artifacts/runs/taco_hand_tracking_v1/manifest.resolved.json \
  --execute --gpu 0
```

By default the batch plans robot rendering, object depth, depth-aware
compositing, and the comparison grid for every manifest sequence. Repeat
`--sequence <id>` or `--stage <name>` to select a subset. Complete bundles are
strictly validated and resumed without rerunning their stages. Incomplete or
conflicting outputs block execution; use `--overwrite` only when replacing
those exact selected-stage outputs is intentional.

## Reproduce or resume a learned condition

Video2Data and Phantom use their condition-specific
`tracking/robot_trajectory.npz`, the same shared E2FGVI video, and the exact
validated GT object-depth bundle. Planning is read-only; execution and one
physical GPU selector are explicit:

```bash
python3 -m inpainting.run_learned_condition_batch \
  --manifest inpainting/artifacts/runs/taco_hand_tracking_v1/manifest.resolved.json \
  --condition v2d \
  --sequence taco_dust__brush__cup_20231005_253 \
  --max-ik-residual-m 0.012 \
  --max-joint-step-rad 0.4 \
  --gpu 0

# After reviewing the plan:
python3 -m inpainting.run_learned_condition_batch \
  --manifest inpainting/artifacts/runs/taco_hand_tracking_v1/manifest.resolved.json \
  --condition v2d \
  --sequence taco_dust__brush__cup_20231005_253 \
  --max-ik-residual-m 0.012 \
  --max-joint-step-rad 0.4 \
  --gpu 0 --execute
```

Resume validation binds the trajectory, camera files, immutable renderer image,
renderer sources, external IK sources, kinematics policy, and every robot
asset fingerprint. Existing stale/incomplete generations block unless their
exact selected stage is deliberately replaced with `--overwrite`. A trajectory
with a missing hand row is rejected at plan time because the renderer has no
implicit hold or interpolation policy.

## Reproduce or resume a parallel-jaw comparison

The parallel-jaw extension converts each tracker's 21-joint hand tracks into
one robot-neutral world-frame pose and aperture contract, then maps that
contract into either the Galbot Golf or YAM bundle. Planning is read-only by
default:

```bash
python3 -m inpainting.run_parallel_jaw_comparison \
  --manifest inpainting/artifacts/runs/taco_hand_tracking_v1/manifest.resolved.json \
  --sequence taco_dust__brush__cup_20231005_253 \
  --bundle inpainting/artifacts/parallel_jaw_assets/galbot_one_golf/bundle_manifest.json \
  --robot-asset-root /path/to/galbot_one_golf_description

# After reviewing the seven-stage plan:
python3 -m inpainting.run_parallel_jaw_comparison \
  --manifest inpainting/artifacts/runs/taco_hand_tracking_v1/manifest.resolved.json \
  --sequence taco_dust__brush__cup_20231005_253 \
  --bundle inpainting/artifacts/parallel_jaw_assets/galbot_one_golf/bundle_manifest.json \
  --robot-asset-root /path/to/galbot_one_golf_description \
  --gpu 0 --execute
```

For YAM, use
`inpainting/artifacts/parallel_jaw_assets/yam_bimanual/bundle_manifest.json`
as the bundle and its parent directory as `--robot-asset-root`. The
preview-reviewed Galbot `105` result also requires
`--max-orientation-residual-deg 55 --max-joint-step-rad 0.55`; all other
production comparisons retain the strict defaults of 20 degrees and
0.4 rad/frame.

Each run reuses one GT-derived hub transform across all three tracker
conditions and writes:

```text
<sequence>/parallel_jaw/<robot_id>/<condition>/robot_render/*
<sequence>/parallel_jaw/<robot_id>/<condition>/final_overlay.{mp4,json}
<sequence>/parallel_jaw/<robot_id>/final_5panel_comparison.{mp4,json}
```

Resume validation binds the target, bundle, immutable render image, camera,
shared mount, renderer policy, GT object-depth input, composites, and final
grid lineage. Existing partial or stale artifacts block until their exact
selected stage is deliberately replaced with `--overwrite`.

The derived Galbot/YAM bundles are ignored outputs and therefore are not part
of the pushed branch. Fresh-clone source pins, bundle-build commands, and the
role of RoboLab MRs 62 and 68 are documented in
[`PARALLEL_JAW_REPRODUCTION.md`](PARALLEL_JAW_REPRODUCTION.md).

## Refine parallel-jaw contacts with GraspGenX

The refinement is deliberately split into two auditable commands. First,
`inpainting.graspgenx_candidates` runs the official sweep-volume-conditioned
GraspGenX inference on one metric SAM3D mesh and writes a ranked candidate NPZ
plus hashed JSON provenance. Then `inpainting.run_graspgenx_refinement` selects
one grasp against V2D thumb/index contacts and writes the same 12-key
robot-neutral target contract consumed by the parallel-jaw renderer.

```bash
python3 -m inpainting.graspgenx_candidates --help
python3 -m inpainting.run_graspgenx_refinement --help
```

Run the second command once per interaction, feeding its output target into
the next interaction so the other side is preserved. Production uses
`--propagation-mode base_local_offset`,
`--score-registration-weight 0`,
`--score-pose-rotation-weight 0.04`, and
`--score-approach-weight 0.015`. The reviewed start/anchor/end windows and
their RGB/Phantom evidence live in
`inpainting/configs/graspgenx_refinement_events.json`. Final combined targets
are stored under:

```text
<sequence>/parallel_jaw/graspgenx_targets/<robot_id>/v2d_graspgenx_aligned/
```

Candidate generation and grasp selection use only V2D hand tracks and
MoGe/SAM2/SAM3D/FoundationPose object reconstruction. The review overlays keep
the existing GT object-depth occluder fixed to isolate the trajectory change,
so the rendered ablation is not a pure-RGB end-to-end compositing result.
The complete source setup, production sweep volumes, `600/150` candidate
policy, chained interaction command, and refined render gates are in
[`PARALLEL_JAW_REPRODUCTION.md`](PARALLEL_JAW_REPRODUCTION.md).

## Current model status

E2FGVI-HQ, SAM2, WiLoR, Grounding DINO, HaMeR, MoGe, SAM3D,
FoundationPose, and the licensed MANO v1.2 pair are local and fingerprinted.
GT is complete on clips 060, 105, and 253;
Video2Data and Phantom are both complete end to end on 105 and 253. Learned
tracking and Sharpa retargeting are complete for all three clips. The learned
060 renders are intentionally blocked because their source trackers contain
real right-hand gaps and no pose filling was authorized. The common `253`
comparison is the definitive current generation: both learned conditions have
tracking/Sharpa v2 provenance, and their strict render/composite/grid plans
report `skipped_complete`. The `105` render/composite/grid outputs also pass
strict current validation, but their learned tracking/Sharpa sidecars retain
legacy-generation provenance. The sequence-105 three-way object-compositing
ablation is complete, including RGB-only SAM2 + MoGe and SAM3D +
FoundationPose conditions. Galbot Golf and YAM parallel-jaw five-panel
comparisons are also complete on clips `105` and `253`; all four full pipeline
plans validate as seven `skipped_complete` stages. GraspGenX-refined V2D
targets and synchronized four-panel baseline/refined comparisons are complete
for both embodiments and both clips. Their local registered contact residuals
are `2.4--20.7` mm, but the required `172.6--305.5` mm hand/object
registration shifts remain an explicit metric-alignment limitation.

The final synchronized source/E2FGVI/GT/Video2Data/Phantom review video is:

```text
/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_253_tracker_comparison.mp4
/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_105_tracker_comparison.mp4
```

## Tests

The contract and orchestration tests have no model or GPU dependency. The
repository test suite uses pytest because several strict provenance tests use
fixtures and monkeypatching:

```bash
pytest -q inpainting/tests inpainting/e2fgvi/tests \
  inpainting/robot_renderer/tests inpainting/phantom_tracker/tests \
  inpainting/parallel_jaw_renderer/tests inpainting/robot_assets/tests \
  reconstruction/modules/v2d_docker/tests \
  reconstruction/modules/v2d_depth/tests \
  reconstruction/modules/v2d_foundation_pose/docker/tests \
  reconstruction/modules/v2d_foundation_pose/lib/tests \
  reconstruction/modules/v2d_moge/docker/tests \
  reconstruction/modules/v2d_pipelines/tests \
  reconstruction/modules/v2d_sam3d/docker/tests
```
