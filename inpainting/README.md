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

## Current model status

E2FGVI-HQ, SAM2, WiLoR, Grounding DINO, HaMeR, and the licensed MANO v1.2 pair
are local and fingerprinted. GT is complete on clips 060, 105, and 253;
Video2Data is complete end to end on 105 and 253; Phantom is complete end to
end on 253. Learned tracking and Sharpa retargeting are complete for all three
clips. The learned 060 renders are intentionally blocked because their source
trackers contain real right-hand gaps and no pose filling was authorized. The
common `253` comparison is the definitive current generation: both learned
conditions have tracking/Sharpa v2 provenance, and their strict
render/composite/grid plans report `skipped_complete`. The earlier `060` and
`105` learned sidecars are retained as legacy-generation supporting artifacts.

The final synchronized source/E2FGVI/GT/Video2Data/Phantom review video is:

```text
/home/mverghese/visual_inpainting/video_to_data_internal/inpainting/artifacts/audit/taco_253_tracker_comparison.mp4
```

## Tests

The contract and orchestration tests have no model or GPU dependency. The
repository test suite uses pytest because several strict provenance tests use
fixtures and monkeypatching:

```bash
pytest -q inpainting/tests inpainting/e2fgvi/tests \
  inpainting/robot_renderer/tests inpainting/phantom_tracker/tests \
  reconstruction/modules/v2d_docker/tests
```
