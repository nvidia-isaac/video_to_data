# v2d_hoi_object_reconstruction

End-to-end textured 3D mesh reconstruction from hand-object interaction video.

Two reconstruction modes:
- **BundleSDF** (default) — two-stage scan (stationary → rotated → stationary) → full textured NeRF mesh
- **SAM3D** — select representative frames → per-frame single-image 3D → silhouette-based scale estimation

---

## Data Collection

_Coming soon._

---

## Environment Setup

### 1. Install host packages (from `reconstruction/`)

```bash
./scripts/install_pacakages.sh
```

### 2. Build all required containers

```bash
./scripts/build_containers.sh
```

### 3. Download model weights

```bash
# Shared (both modes)
python -m v2d.sam2.docker.run_download_weights --output_dir data/weights/sam2
python -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino

# BundleSDF mode
python modules/v2d_foundation_stereo/docker/run_download_weights.py \
  --output_dir data/weights/foundationstereo
python modules/v2d_foundation_pose/docker/run_download_weights.py \
  --output_dir data/weights/foundationpose
python modules/v2d_bundlesdf/docker/run_download_weights.py --output_dir data/weights

# SAM3D mode
python modules/v2d_sam3d/docker/run_download_weights.py --output_dir data/weights/sam3d
# (optional: FoundationStereo for depth-assisted scale estimation)
python modules/v2d_foundation_stereo/docker/run_download_weights.py \
  --output_dir data/weights/foundationstereo
```

---

## BundleSDF Pipeline

Two-stage scan: object stationary → rotated 360° → stationary again.

### Quick Start

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
  --mapping_data_dir data/hoi_obj_recon/raw_data/<job> \
  --job_dir          data/outputs/hoi_recon/<job> \
  --prompt           "basketball"
```

### Pipeline Steps

```
mapping_data_dir  (images + frames_meta.json)
    ↓
1.  prepare_FP_folder   → job_dir/left/, right/, calibration.json, video.mp4
    ↓
2.  CuSFM               → sfm/keyframes/frames_meta.json  (camera poses)
    ↓
2b. Stage-1 auto-detect → stage1_detect_debug/result.json  (transition frame)
    ↓
3.  Grounding DINO      → grounding_dino_bboxes.json
4a. FoundationStereo    → depth/  (all frames, parallel workers)
4b. SAM2                → masks/  (all frames, from reference frame)
    ↓
5.  Stage-1 setup       → stage1_recon/  (SfM keyframes + depth symlinks)
6.  BundleSDF NeRF      → stage1_recon/textured_mesh.obj
7.  Center mesh         → mesh_input.obj
    ↓
8.  FoundationPose      → poses/  (tracking with Stage-1 mesh)
8b. FP render           → fp_render/render.mp4
    ↓
9.  World poses         → poses_world.json  (T_world_from_obj + stage detection)
10. Merged setup        → merged_recon/  (both stages aligned)
11. BundleSDF NeRF      → merged_recon/textured_mesh.obj
    ↓
12. FoundationPose      → poses_final/  (tracking with final mesh)
13. FP render           → fp_render_final/render.mp4
```

### Results

- `merged_recon/textured_mesh.obj` — final textured mesh (+ `.mtl`, `_0.png`)
- `stage1_recon/textured_mesh.obj` — Stage-1 mesh (bottom missing)

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--stage1_end_frame N` | auto | Override Stage-1 end (sequential frame index) |
| `--stage1_end_timestamp NS` | auto | Override Stage-1 end (nanosecond timestamp) |
| `--stage1_buffer_deg DEG` | 10.0 | Angle buffer before detected transition |
| `--config PATH` | bundlesdf default | NeRF/SDF config YAML |
| `--pipeline_config PATH` | `lib/data/configs/hoi_pipeline.yaml` | Pipeline config YAML |
| `--reference_frame N` | 0 | SAM2/FP reference frame |
| `--num_depth_workers N` | 2 | Parallel FoundationStereo workers |
| `--fp_weights_dir PATH` | `data/weights/foundationpose` | FoundationPose weights |

### Skip Flags (BundleSDF)

Resume from any checkpoint by skipping completed steps:

```
--skip_prepare  --skip_sfm  --skip_stage1_detect  --skip_dino  --skip_depth  --skip_mask
--skip_stage1_setup  --skip_stage1_nerf  --skip_center_mesh
--skip_fp_tracking  --skip_fp_render  --skip_world_poses  --skip_merged_setup  --skip_full_nerf
--skip_final_fp_tracking  --skip_final_fp_render
```

### Configuration

**Pipeline config** — `lib/data/configs/hoi_pipeline.yaml` (override with `--pipeline_config`):

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `stage1_detect` | `buffer_deg` | 10.0 | Angle buffer (°) before detected Stage-1 transition |
| `sfm` | `config_set` | `backpack` | CuSFM preset (`backpack` \| `av` \| `isaac` \| `rgbd`) |
| `depth` | `num_workers` | 2 | Parallel FoundationStereo depth workers |
| `foundationpose` | `reference_frame` | 0 | FP registration reference frame |
| `foundationpose` | `weights_dir` | `null` | FP weights path; `null` = `data/weights/foundationpose` |

**NeRF/SDF config** — `modules/v2d_bundlesdf/lib/data/configs/theseus_optimizer_hawk.yaml` (override with `--config`):

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `camera_config` | `step` | 4 | CuSFM keyframe subsampling for SDF training |
| `nerf` | `n_step` | 3000 | SDF training steps |
| `nerf` | `trunc` | 0.01 | TSDF truncation distance (normalized). Larger = fewer holes |
| `nerf` | `mesh_resolution` | 0.005 | Voxel size for mesh extraction. Must be ≤ `trunc` |
| `texture_bake` | `texture_res` | 2048 | Output texture atlas resolution |
| `texture_bake` | `downscale` | 1.0 | Image downscale for texture baking |
| `texture_bake` | `min_keyframe_translation` | 0.0 | Min camera translation (m) between texture keyframes |
| `texture_bake` | `min_keyframe_rotation_deg` | 5.0 | Min camera rotation (°) between texture keyframes |
| `texture_bake` | `min_keyframes` | 30 | Minimum keyframes after subsampling |

---

## SAM3D Pipeline

Single-image 3D reconstruction per representative frame. No multi-stage scan required.

### Quick Start

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
  --mapping_data_dir data/hoi_obj_recon/raw_data/<job> \
  --job_dir          data/outputs/hoi_recon/<job> \
  --prompt           "basketball" \
  --mode sam3d
```

With depth-assisted scale estimation:

```bash
python ... --mode sam3d --sam3d_use_depth
```

### Pipeline Steps

```
mapping_data_dir  (images + frames_meta.json)
    ↓
1.  prepare_FP_folder   → job_dir/left/, right/, calibration.json, video.mp4
    ↓
2.  CuSFM               → sfm/keyframes/frames_meta.json  (camera poses for frame selection)
    ↓
2b. Stage-1 auto-detect → stage1_detect_debug/result.json  (exclude transition frames)
    ↓
3.  Grounding DINO      → grounding_dino_bboxes.json
4b. SAM2                → masks/  (all frames — required for SRT scale)
[4a. FoundationStereo]  → depth/  (optional, only with --sam3d_use_depth)
    ↓
S1. Select frames       → sam3d/selected_frames.json  (one per azimuthal bin)
S2. SAM3D               → sam3d/<frame_id>/mesh.glb + transform.json + intrinsics.json
S3. SRT scale           → sam3d/<frame_id>/srt/srt_result.json + output_scaled.glb
                          (Stage-1 frames only — object stationary)
S4. Render debug        → sam3d/<frame_id>/render_debug.jpg
S5. Render video        → sam3d/<frame_id>/render_video.mp4
                          (textured mesh overlaid on Stage-1 keyframes via open3d)
```

### Results

- `sam3d/<frame_id>/mesh.glb` — raw SAM3D mesh (SAM3D camera space)
- `sam3d/<frame_id>/srt/output_scaled.glb` — scale-corrected mesh (world space)
- `sam3d/<frame_id>/srt/srt_result.json` — estimated scale, rotation, translation
- `sam3d/<frame_id>/render_debug.jpg` — SAM3D mesh overlaid on source image (single frame)
- `sam3d/<frame_id>/render_video.mp4` — textured mesh overlaid on all Stage-1 keyframes

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sam3d_use_depth` | off | Use FoundationStereo depth as extra loss in SRT scale estimation |
| `--sam3d_bin_deg DEG` | 60.0 | Azimuthal bin size for frame selection |
| `--sam3d_seed N` | 42 | Random seed for SAM3D inference |

### Skip Flags (SAM3D)

```
--skip_prepare  --skip_sfm  --skip_stage1_detect  --skip_dino  --skip_depth  --skip_mask
--skip_select_frames  --skip_sam3d  --skip_srt_scale  --skip_render_debug  --skip_render_video
```

---

## Troubleshooting

Start from the final output and work backwards.

### BundleSDF: Bad final mesh

**Step 1 — Check Stage-1 mesh** (`stage1_recon/textured_mesh.obj`).

If Stage-1 mesh is bad, check inputs in order:

1. **Masks** — Spot-check `masks/0/`. Object should be cleanly segmented in every frame.
   - Wrong box: try a more specific `--prompt` (e.g. `"red spray bottle"` not `"bottle"`)
   - Mask drifts: adjust `foundationpose.reference_frame` in the pipeline config

2. **Camera poses and point cloud** — Run:
   ```bash
   python modules/v2d_bundlesdf/tools/visualize_reconstruction_standalone.py <job_dir>/stage1_recon/
   ```
   Cameras should orbit the object cleanly and the point cloud should form a coherent shape.

3. **Surface holes** — If cameras/point cloud are good but mesh has holes:
   - Increase `nerf.trunc` (default `0.01`, try `0.02`)
   - Decrease `nerf.mesh_resolution`, must stay ≤ `trunc` (default `0.005`, try `0.003`)
   - Increase `texture_bake.texture_res` for sharper texture (default `2048`, try `4096`)

If Stage-1 is good but final mesh is bad, the problem is in two-stage alignment:

1. **FP tracking** — Watch `fp_render/render.mp4`. The mesh overlay should stay locked to the object. Drifting indicates poor FoundationPose tracking.

2. **World poses** — Inspect `poses_world_debug.png`:
   - Object position (subplot 3): near-flat during Stage 1 and Stage 2
   - Angular velocity (subplot 5): spike only during stage transition
   - Vertical lines show detected stage boundaries — verify they match the video

### BundleSDF: FoundationPose weights not found

```
[error] FoundationPose weights not found at: ...
```

Download them:
```bash
python modules/v2d_foundation_pose/docker/run_download_weights.py \
  --output_dir data/weights/foundationpose
```

### BundleSDF: zfar clipping artifacts

If the mesh appears sliced at a fixed distance, the zfar clipping plane is too close. This is patched in the `v2d_bundlesdf` Dockerfile — rebuild the container:
```bash
docker build -t v2d_bundlesdf modules/v2d_bundlesdf/docker/
```

### Internal: cross-container symlinks

The pipeline uses **relative symlinks** to map sparse keyframe indices (from SfM) to contiguous `left{N:06d}` indices expected by BundleSDF, without copying large depth and mask files.

All symlink targets must live under `job_dir`. The `v2d_hoi_object_reconstruction` container and the `v2d_bundlesdf` container each mount `job_dir` at a different internal path, so a relative symlink like `../../depth/000001.png` resolves correctly in both.

**Constraints:**
- `depth/` must be a direct subdirectory of `job_dir` ✓
- `masks/` must also be a direct subdirectory of `job_dir` ✓

If you supply external depth or mask directories outside `job_dir`, symlinks inside BundleSDF will be broken. Keep all intermediate data within `job_dir`.

### FoundationPose Tracking Only

To run FoundationPose tracking with an existing mesh (skips reconstruction):

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

---

## Tools

| Tool | Location | Description |
|------|----------|-------------|
| `detect_stage1_end.py` | `v2d_hoi_object_reconstruction/lib/` | Manually inspect Stage-1 end detection from CuSFM trajectory |
| `visualize_reconstruction_standalone.py` | `v2d_bundlesdf/tools/` | Visualize camera trajectory and point cloud for reconstruction-quality checks |
| `plot_tum_file.py` | `v2d_cusfm/tools/` | Plot TUM-format trajectory file |
| `spin_mesh_video.py` | `tools/` | Render a spinning video of a mesh |
| `fuse_depth_to_pointcloud.py` | `v2d_bundlesdf/tools/` | Fuse depth maps into a point cloud |
| `view_glb.py` | `tools/` | View a `.glb` mesh file |
