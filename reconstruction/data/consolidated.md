# `run_ego_reconstruction.py` Consolidation Plan

## Goal

Add `modules/v2d_pipelines/run_ego_reconstruction.py` as the new public
egocentric reconstruction entrypoint.

Leave the old scripts untouched:

- `modules/v2d_pipelines/run_v2d_ego_e2e.py`
- `modules/v2d_pipelines/run_ego_wilor.py`
- `modules/v2d_pipelines/run_ego_wilor_with_obj.py`

The new script should preserve the old DynHaMR workflow behavior, fold in the
useful WiLoR pipeline features, and avoid a large new abstraction. Depth is
always MoGe. The new pipeline should support three commands.

## Required Commands

Old pipeline behavior through the new entrypoint:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video ... \
  --object_prompt "A toy airplane" \
  --output_dir ... \
  --reference_frame 0 \
  --undistort \
  --hand_tracking dynhamr \
  --dev
```

New full pipeline with prompt-based object reconstruction:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video ... \
  --object_prompt "A toy airplane" \
  --output_dir ... \
  --reference_frame 0 \
  --undistort \
  --hand_tracking hamer \
  --run_droid_slam \
  --run_gravity_alignment \
  --run_gsplat_refinement \
  --export_threejs_result \
  --dev
```

New full pipeline with an existing object mesh override:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video ... \
  --object_prompt "A toy airplane" \
  --object_mesh /path/to/object_mesh.obj \
  --skip_object_scale_estimation \
  --output_dir ... \
  --reference_frame 0 \
  --undistort \
  --hand_tracking hamer \
  --run_droid_slam \
  --run_gravity_alignment \
  --run_gsplat_refinement \
  --export_threejs_result \
  --dev
```

The existing docs command in `reconstruction/docs/ego_e2e_setup.md` should keep
working through the old script, since that script is untouched:

```bash
python modules/v2d_pipelines/run_v2d_ego_e2e.py \
  --video_path assets/airplane.mp4 \
  --prompt "airplane" \
  --output_dir data/outputs/airplane \
  --depth_source moge
```

The new script can also accept the old names as compatibility aliases:

- `--video_path` aliases to `--video`.
- `--prompt` aliases to `--object_prompt`.
- `--depth_source` is accepted but ignored with a warning. Depth is always MoGe.

## Public Flags

Core:

```bash
--video PATH
--output_dir DIR
--object_prompt TEXT
--object_mesh PATH
--skip_object_scale_estimation
--reference_frame INT
--undistort
--dev
```

Object behavior:

- `--object_prompt` is required for all object workflows. It drives object
  detection, masks, and FoundationPose tracking initialization.
- If `--object_mesh` is set, use that mesh and skip SAM3D reconstruction.
- If `--object_mesh --skip_object_scale_estimation` is set, trust the mesh scale
  exactly and pass it directly into object tracking.
- If `--object_mesh` is not set, require `--object_prompt` and reconstruct the
  object with the existing DINO/SAM2/SAM3D path.
- Object tracking still uses FoundationPose.
- Keep `--reregister_iou_thresh` for FoundationPose re-registration.
- Scale estimation should remain enabled for SAM3D meshes if that matches the
  current behavior. For provided meshes, `--skip_object_scale_estimation` is the
  explicit way to use the mesh directly.

Hand tracking:

```bash
--hand_tracking {dynhamr,hamer}
```

- `dynhamr`: old ViPE + DynHaMR hand path. This should reproduce the current
  `run_v2d_ego_e2e.py` behavior, except that downstream depth is MoGe.
- `hamer`: WiLoR detections + SAM2 hand masks + HaMeR hand reconstruction,
  equivalent to the current tissue box WiLoR workflow.

Optional stages:

```bash
--run_droid_slam
--run_gravity_alignment
--run_gsplat_refinement
--export_threejs_result
```

Naming cleanup:

- Use `--run_droid_slam`, not `--run_slam`.
- Use `--run_gravity_alignment`, not `--gravity_align`.
- Use `--run_gsplat_refinement`, not `--run_refinement_simple`.
- Use `--gsplat_refine_*` for refinement options. Do not expose
  `--simple_refinement_*` as the new public API.

## Gsplat Defaults

`refine_simple.py` is the only refinement implementation for this plan. The
pipeline should call it when `--run_gsplat_refinement` is set.

The default `gsplat_refine_*` values in `run_ego_reconstruction.py` should match
the current tuned values from
`reconstruction/data/garage/tissue_box/run_wilor_pipeline.sh`, so the shell
script can become much shorter.

Default values to bake into `run_ego_reconstruction.py`:

```bash
--gsplat_refine_epochs 10
--gsplat_refine_batch_size 4
--gsplat_refine_train_resolution_scale 0.5
--gsplat_refine_lr_gaussians 1e-3
--gsplat_refine_lr_object_pose 1e-3
--gsplat_refine_lr_object_scale 1e-3
--gsplat_refine_lr_hand_pose 1e-3
--gsplat_refine_lr_hand_articulation 1e-3
--gsplat_refine_lr_hand_shape 1e-3
--gsplat_refine_lr_hand_scale 1e-3
--gsplat_refine_lr_camera_pose 1e-3
--gsplat_refine_init_opacity_obj 0.5
--gsplat_refine_init_opacity_hand 0.5
--gsplat_refine_init_opacity_bg 0.5
--gsplat_refine_init_gaussian_scale_factor 1.0
--gsplat_refine_w_perceptual 1.0
--gsplat_refine_w_hand_object_penetration 0.0
--gsplat_refine_hand_object_penetration_margin 0.003
--gsplat_refine_object_sdf_resolution 96
--gsplat_refine_hand_object_penetration_max_verts 0
--gsplat_refine_perceptual_resize 224
--gsplat_refine_lr_schedule cosine
--gsplat_refine_lr_cosine_min_factor 0.1
--gsplat_refine_w_smooth_object_rot 1000.0
--gsplat_refine_w_smooth_object_trans 1000.0
--gsplat_refine_w_smooth_hand_rot 1000.0
--gsplat_refine_w_smooth_hand_articulation 1000.0
--gsplat_refine_w_smooth_hand_trans 1000.0
--gsplat_refine_w_smooth_camera_rot 1000.0
--gsplat_refine_w_smooth_camera_trans 1000.0
--gsplat_refine_w_mask 1.0
--gsplat_refine_w_relative_depth 1.0
--gsplat_refine_render_every 25
--gsplat_refine_w_smooth_hand_object_relative_rot 1000.0
--gsplat_refine_w_smooth_hand_object_relative_trans 1000.0
```

Optional refinement flags can still exist, but should use the same prefix:

```bash
--gsplat_refine_mask_background
--gsplat_refine_debug_frame INT
```

Internally, these can map directly to the current `refine_simple.py` arguments.
The old `--simple_refinement_*` names can be accepted temporarily as hidden
aliases, but new scripts should not use them.

## Setup and Docs Work

Update `reconstruction/docs/ego_e2e_setup.md` so it covers all three commands
from this file:

1. DynHaMR hand tracking with prompt-based SAM3D object reconstruction.
2. HaMeR hand tracking with prompt-based SAM3D object reconstruction, DROID-SLAM,
   gravity alignment, and gsplat refinement.
3. HaMeR hand tracking with prompt-based object masks and a provided object mesh,
   `--skip_object_scale_estimation`, DROID-SLAM, gravity alignment, and gsplat
   refinement.

The docs should keep the old `run_v2d_ego_e2e.py` example as legacy usage, but
make `run_ego_reconstruction.py` the recommended entrypoint for new runs.

Rename the focused setup scripts to match the new pipeline name. The docs should
use this setup sequence:

```bash
# 2. Install host packages
./scripts/install_ego_reconstruction_packages.sh

# 3. Build Docker images
./scripts/build_ego_reconstruction_packages.sh

# 4. Download model weights
./scripts/download_ego_reconstruction_weights.sh
```

Add `reconstruction/scripts/install_ego_reconstruction_packages.sh` so the
focused ego reconstruction install covers every supported configuration. In
addition to the current old pipeline packages, it should include:

```bash
-e modules/v2d_geocalib/docker
-e modules/v2d_droid_slam/docker
-e modules/v2d_gsplat_refinement/docker
-e modules/v2d_wilor/docker
```

Keep the existing packages for MoGe, AnyCalib, GroundingDINO, SAM2, SAM3D,
FoundationPose, HaMeR, hand alignment, DynHaMR ego hand reconstruction, common,
docker, depth, and pipelines.

Add `reconstruction/scripts/build_ego_reconstruction_packages.sh` so the focused
ego reconstruction container build covers every supported configuration. The
regular module list should include:

```bash
anycalib
moge
grounding_dino
sam2
sam3d
foundation_pose
hamer
wilor
geocalib
droid_slam
gsplat_refinement
```

Keep the special build calls for:

```bash
v2d_ego_hand_reconstruction
v2d_hand_alignment
```

Add the focused weight download script:

```bash
reconstruction/scripts/download_ego_reconstruction_weights.sh
```

The download script should cover all three configurations. A simple first
version can download all supported weights by default:

```bash
python -m v2d.moge.docker.run_download_weights --output_dir data/weights/moge
python -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
python -m v2d.sam2.docker.run_download_weights --output_dir data/weights/sam2
python -m v2d.sam3d.docker.run_download_weights --output_dir data/weights/sam3d
python -m v2d.foundation_pose.docker.run_download_weights --output_dir data/weights/foundation_pose
python -m v2d.anycalib.docker.run_download_weights --output_dir data/weights/anycalib
python -m v2d.wilor.docker.run_download_weights --weights_dir data/weights/wilor
python -m v2d.hamer.docker.run_download_weights --weights_dir data/weights/hamer
python -m v2d.droid_slam.docker.run_download_weights --output_dir data/weights/droid_slam
python -m v2d.geocalib.docker.run_download_weights --output_dir data/weights/geocalib
python -m v2d.gsplat_refinement.docker.run_download_weights --weights_path data/weights/gsplat_refinement
```

The docs should still call out manual DynHaMR/MANO requirements under
`data/weights/hand`:

```text
data/weights/hand/
  models/MANO_RIGHT.pkl
  BMC/*.npy
```

SAM3D should keep its Hugging Face token note.

If the download script later gets modes, use these names:

```bash
--mode all
--mode dynhamr_prompt
--mode hamer_prompt
--mode hamer_mesh
```

For now, defaulting to `--mode all` is simpler and less error-prone.

## Minimal Implementation Plan

1. Add `run_ego_reconstruction.py` argument parsing.
   - Add `--video`, `--object_prompt`, `--object_mesh`,
     `--skip_object_scale_estimation`, `--hand_tracking`, `--run_droid_slam`,
     `--run_gravity_alignment`, and `--run_gsplat_refinement`.
   - Add `--video_path`, `--prompt`, and `--depth_source` as compatibility
     aliases in the new script only.
   - Do not change the old scripts.
   - Remove depth-source branching from the new pipeline. Always run MoGe.

2. Reuse the HaMeR/WiLoR path from `run_ego_wilor.py`.
   - For `--hand_tracking hamer`, run WiLoR detections, SAM2 hand masks, HaMeR,
     hand depth alignment, and result packaging.
   - Use the same defaults as the tissue box script:
     `--sam2_bbox_pad 0.05`, `--no_sam2_square_wilor_bboxes`, and
     `--sam2_hand_prompt_source wilor_mask`.

3. Reuse the old DynHaMR path from `run_v2d_ego_e2e.py`.
   - For `--hand_tracking dynhamr`, keep the existing ViPE + DynHaMR hand
     reconstruction flow.
   - Feed its converted hand outputs into the same final result bundle writer
     used by the HaMeR path.

4. Add provided mesh override support.
   - If `--object_mesh` is set, skip SAM3D.
   - If `--skip_object_scale_estimation` is set, skip object scale estimation
     and trust the provided mesh units for FoundationPose tracking.
   - Always require `--object_prompt` for object masks and FoundationPose
     tracking initialization.
   - Package the provided mesh into the final result bundle.

5. Add optional stages.
   - `--run_droid_slam` runs the current DROID-SLAM path before gsplat
     refinement.
   - `--run_gsplat_refinement` runs `refine_simple.py` with the baked-in
     `gsplat_refine_*` defaults above.
   - `--run_gravity_alignment` runs GeoCalib/gravity alignment and writes a
     suffixed gravity-aligned result bundle without mutating `result/`.
   - `--export_threejs_result` writes a Three.js viewer under the final result
     bundle directory.

6. Update setup, docs, and data scripts.
   - Add `install_ego_reconstruction_packages.sh` with all host package
     requirements.
   - Add `build_ego_reconstruction_packages.sh` with all required containers.
   - Add `download_ego_reconstruction_weights.sh` for model weight setup.
   - Update `reconstruction/docs/ego_e2e_setup.md` to cover the three commands
     from this plan.
   - Simplify `reconstruction/data/garage/tissue_box/run_wilor_pipeline.sh` to
     the new full-pipeline command plus only values that differ from defaults.
   - Do not modify old script behavior unless we decide to deprecate it later.

## Expected Outputs

Always write the base bundle:

```text
<output_dir>/result/
```

Optional post-processing writes suffixed bundles instead of mutating `result/`:

```text
<output_dir>/result_slam/
<output_dir>/result_gravity_aligned/
<output_dir>/result_slam_gravity_aligned/
```

When gsplat refinement is enabled, cache refinement outputs under the existing
refinement cache directory. When Three.js export is enabled, write:

```text
<final_result_dir>/threejs_scene/index.html
```

The final result bundle should use refined poses when available, otherwise the
tracked/smoothed poses from the selected hand/object pipeline.

## Main Risk

The only nontrivial merge point is making DynHaMR and HaMeR produce the same
inputs for result bundling and gsplat refinement. Keep that contract narrow:
object mesh, object pose directory, hand MANO/mesh outputs, MoGe depth,
intrinsics, frames, and optional camera poses from DROID-SLAM.
