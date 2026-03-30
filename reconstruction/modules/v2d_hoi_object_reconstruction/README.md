# v2d_hoi_object_reconstruction

End-to-end textured 3D mesh reconstruction from hand-object interaction video using a **two-stage scan** (object stationary → rotated → stationary).

## Prerequisites

1. **Install host packages** (from `reconstruction/`):
   ```bash
   ./scripts/install_pacakages.sh
   ```

2. **Build all required containers**:
   ```bash
   ./scripts/build_containers.sh
   ```

3. **Download model weights**:
   ```bash
   python -m v2d.sam2.docker.run_download_weights --output_dir data/weights/sam2
   python -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
   python modules/v2d_foundation_stereo/docker/run_download_weights.py \
     --output_dir data/weights/foundationstereo
   python modules/v2d_foundation_pose/docker/run_download_weights.py \
     --output_dir data/weights/foundationpose
   python modules/v2d_bundlesdf/docker/run_download_weights.py --output_dir data/weights
   ```

## Quick Start

Run on the included example data (from `reconstruction/`):

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
  --mapping_data_dir modules/v2d_hoi_object_reconstruction/assets/basketball_example \
  --job_dir          data/outputs/hoi_recon/basketball_example \
  --prompt           "basketball"
```

Or on your own data:

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
  --mapping_data_dir data/hoi_obj_recon/raw_data/<job> \
  --job_dir          data/hoi_obj_recon/jobs/<job> \
  --prompt           "basketball"
```

Results are written to `<job_dir>/`:
- `merged_recon/textured_mesh.obj` — final textured mesh (+ `.mtl`, `_0.png`)
- `stage1_recon/textured_mesh.obj` — Stage-1 mesh (bottom missing)

## Pipeline

```
mapping_data_dir  (images + frames_meta.json)
    ↓
CuSFM  →  sfm/keyframes/frames_meta.json  (camera poses)
    ↓
Stage-1 auto-detect  →  stage1_end_frame  (transition from stationary to rotation)
    ↓
Grounding DINO  →  object bounding box
FoundationStereo  →  depth/  (all frames, parallel workers)
SAM2  →  masks/  (reference frame only)
    ↓
Stage-1 NeRF  →  stage1_recon/textured_mesh.obj
    ↓
FoundationPose tracking  →  poses/  (all frames)
    ↓
World poses + Stage-2 setup  →  merged_recon/  (both stages aligned)
    ↓
Full NeRF  →  merged_recon/textured_mesh.obj
    ↓
FoundationPose tracking (final)  →  poses_final/  (all frames, final mesh)
```

## Skip Flags

Resume from any checkpoint by skipping completed steps:

```
--skip_prepare  --skip_sfm  --skip_stage1_detect  --skip_dino  --skip_depth  --skip_mask
--skip_stage1_setup  --skip_stage1_nerf  --skip_center_mesh
--skip_fp_tracking  --skip_fp_render  --skip_world_poses  --skip_merged_setup  --skip_full_nerf
--skip_final_fp_tracking  --skip_final_fp_render
```

## Configuration

Two config files control the pipeline:

### Pipeline config — `lib/data/configs/hoi_pipeline.yaml`

Orchestration-level settings (override with `--pipeline_config`):

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `stage1_detect` | `buffer_deg` | 10.0 | Angle buffer (°) before detected Stage-1 transition |
| `sfm` | `config_set` | `backpack` | CuSFM preset (`backpack` \| `av` \| `isaac` \| `rgbd`) |
| `depth` | `num_workers` | 2 | Parallel FoundationStereo depth workers |
| `foundationpose` | `reference_frame` | 0 | Frame used as FoundationPose registration reference |
| `foundationpose` | `weights_dir` | `null` | FP weights path; `null` = `data/weights/foundationpose` |
### NeRF/SDF config — `modules/v2d_bundlesdf/lib/data/configs/theseus_optimizer_hawk.yaml`

Neural rendering settings (override with `--config`). Camera intrinsics are filled in automatically from `<job_dir>/calibration.json`.

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `camera_config` | `step` | 4 | CuSFM keyframe subsampling for SDF training (BundleSDF default; 1 = all keyframes) |
| `nerf` | `n_step` | 3000 | SDF training steps |
| `nerf` | `trunc` | 0.01 | TSDF truncation distance (normalized space). Larger = fewer holes in underobserved regions |
| `nerf` | `mesh_resolution` | 0.005 | Voxel size for mesh extraction (normalized space). Should be ≤ `trunc` |
| `texture_bake` | `texture_res` | 2048 | Output texture atlas resolution |
| `texture_bake` | `downscale` | 1.0 | Image downscale for texture baking (increase to speed up) |
| `texture_bake` | `min_keyframe_translation` | 0.0 | Min camera translation (m) between texture keyframes |
| `texture_bake` | `min_keyframe_rotation_deg` | 5.0 | Min camera rotation (°) between texture keyframes |
| `texture_bake` | `min_keyframes` | 30 | Minimum keyframes after subsampling |

## Internal: cross-container symlinks

The pipeline uses **relative symlinks** to map sparse keyframe indices (from SfM) to the contiguous `left{N:06d}` indices expected by BundleSDF, without copying large depth and mask files.

For these symlinks to resolve correctly across container boundaries, all symlink targets must live under `job_dir`. The `v2d_hoi_object_reconstruction` container and the `v2d_bundlesdf` container each mount `job_dir` at a different internal path (`/data/frames_meta` and `/data/config` respectively), so a symlink created as `../../depth/000001.png` from `stage1_recon/depth/` resolves to `job_dir/depth/000001.png` in both containers.

This constraint means:
- `depth/` must be a direct subdirectory of `job_dir` — ✓ always true
- `masks/` must also be a direct subdirectory of `job_dir` — ✓ SAM2 outputs to `job_dir/masks/0/`

If you supply external depth or mask directories outside `job_dir`, the pipeline will fail silently with broken symlinks inside BundleSDF. Keep all intermediate data within `job_dir`.

## Input from MCAP bag

If raw data is an MCAP bag instead of a `mapping_data` directory:

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
  --mcap_file /path/to/2026-03-12_airplane/ \
  --job_dir   data/hoi_obj_recon/jobs/2026-03-12_airplane \
  --prompt    "airplane"
```

## FoundationPose Tracking Only

To run FoundationPose tracking with an existing mesh (skips reconstruction), prepare the job directory first then skip all reconstruction steps:

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
  --mapping_data_dir data/hoi_obj_recon/raw_data/<job> \
  --job_dir          data/hoi_obj_recon/jobs/<job> \
  --prompt           "basketball" \
  --skip_sfm --skip_stage1_detect --skip_stage1_setup \
  --skip_stage1_nerf --skip_center_mesh --skip_world_poses \
  --skip_merged_setup --skip_full_nerf
```

Place the mesh at `<job_dir>/mesh_input.obj` before running.

## Tools

| Tool | Location | Description |
|------|----------|-------------|
| `detect_stage1_end.py` | `v2d_hoi_object_reconstruction/lib/` | Manually inspect Stage-1 end detection from a CuSFM trajectory |
| `visualize_reconstruction_standalone.py` | `v2d_bundlesdf/tools/` | Visualize BundleSDF camera trajectory and point cloud for reconstruction-quality checks |
| `plot_tum_file.py` | `v2d_cusfm/tools/` | Plot TUM-format trajectory file |
| `spin_mesh_video.py` | `tools/` | Render a spinning video of a mesh |
| `fuse_depth_to_pointcloud.py` | `v2d_bundlesdf/tools/` | Fuse depth maps into a point cloud |
| `view_glb.py` | `tools/` | View a `.glb` mesh file |

## Troubleshooting

Start from the final mesh and work backwards.

### Step 1 — Check the final mesh

Open `merged_recon/textured_mesh.obj`. If it looks good, you're done.

### Step 2 — If the final mesh is bad, check Stage-1 mesh

Open `stage1_recon/textured_mesh.obj`.

**If Stage-1 mesh is bad**, the inputs to NeRF are wrong. Check in order:

1. **Masks** — Spot-check `masks/0/`. The object should be cleanly segmented in every frame.
   - Box wrong: first try a more specific `--prompt` (e.g. `"red spray bottle"` instead of `"bottle"`); if the box is still wrong, override with `--bbox x1,y1,x2,y2`
   - Mask drifts: adjust `foundationpose.reference_frame` in the pipeline config

2. **Camera poses and point cloud** — Inspect the camera trajectory and fused point cloud used for Stage-1 NeRF:
   ```bash
   python modules/v2d_bundlesdf/tools/visualize_reconstruction_standalone.py <job_dir>/stage1_recon/
   ```
   Cameras should orbit the object cleanly and the point cloud should form a coherent object shape. Bad poses or a noisy/sparse point cloud will produce poor NeRF geometry regardless of mask quality.

3. **Holes on the mesh surface** — If cameras and point cloud look good but the mesh has surface holes, try adjusting parameters in `lib/data/configs/theseus_optimizer_hawk.yaml`:
   - Increase `nerf.trunc` to fill holes in under-observed regions (default `0.01`, try `0.02`)
   - Decrease `nerf.mesh_resolution` for a finer voxel grid, must be ≤ `trunc` (default `0.005`, try `0.003`)
   - Increase `texture_bake.texture_res` for sharper texture (default `2048`, try `4096`)

**If Stage-1 mesh is good but final mesh is bad**, the problem is in the two-stage alignment. Check:

1. **FP tracking** — Watch `fp_render/render.mp4`. The Stage-1 mesh overlay should stay locked to the object throughout. Drifting or jumping indicates poor FoundationPose tracking.

2. **World poses** — Inspect `poses_world_debug.png`:
   - Object position (subplot 3) should be near-flat during Stage 1 and Stage 2
   - Angular velocity (subplot 5) should spike only during the transition between stages
   - Vertical lines show the detected stage boundaries — verify they match the video
