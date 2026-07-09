# v2d_hoi_object_reconstruction

End-to-end textured 3D mesh reconstruction from hand-object interaction video.

Two reconstruction modes:
- **BundleSDF** (default) — two-stage scan (stationary → rotated → stationary) → full textured NeRF mesh
- **SAM3D** — select representative frames → per-frame single-image 3D → silhouette-based scale estimation

---

## Data Collection

### Capture hardware

The reference data was collected with a backpack-based capture rig whose main
components are a **Hawk stereo camera** and an **NVIDIA Orin** compute and
recording unit. This hardware is not a pipeline requirement. Input may originate
from stereo videos, image sequences, or another capture format as long as it is
converted to synchronized left/right JPEG keyframes and calibration metadata in
the HOI reconstruction layout described in
[Input Data Format](#input-data-format).

### Preparing the input dataset

The base image and metadata format is defined by public
[PyCuSFM](https://github.com/nvidia-isaac/pyCuSFM). This module consumes a
stricter, single-stereo-pair profile of PyCuSFM's `frames_meta.json` contract.
See the public [PyCuSFM input
tutorial](https://github.com/nvidia-isaac/pyCuSFM/blob/main/docs/tutorial.md#raw-data-requirements)
for the underlying `KeyframesMetadataCollection` format.

For source videos or another capture format:

1. Extract synchronized left and right frames as JPEG images under
   `front_stereo_camera_left/` and `front_stereo_camera_right/`. Preserve
   nanosecond capture timestamps in the filenames when available.
2. Create a PyCuSFM generator YAML containing the real image dimensions,
   rectified pinhole intrinsics, camera-to-rig transforms, sensor names, and
   stereo baseline. Do not infer or copy calibration from the example dataset.
3. Use PyCuSFM's public [keyframe metadata
   generator](https://github.com/nvidia-isaac/pyCuSFM/blob/main/pycusfm/generate_frame_meta.py)
   in stereo-images mode:

```bash
python pycusfm/generate_frame_meta.py \
  --images /path/to/mapping_data_dir \
  --config /path/to/camera_config.yaml \
  --output /path/to/mapping_data_dir/frames_meta.json
```

Add `--use-pseudo-timestamps` when synchronized image sequences have no capture
timestamps. The generator assigns synchronization IDs and emits the CuSFM
metadata structure. For rectified pinhole input, empty or zero distortion
coefficients cause it to emit the 3×4 `projection_matrix` required by this HOI
pipeline.

Finally, validate the generated JSON against
[`schemas/frames_meta.schema.json`](schemas/frames_meta.schema.json), check the
cross-field invariants below, and compare the directory structure with
[`assets/basketball_example/`](assets/basketball_example/). A custom converter
is also compatible if it produces the same contract with correct
synchronization and calibration.

Before a scan, verify that both stereo feeds are clear and synchronized, the
recording frame rate is stable (the reference collection uses 30 FPS), and the
capture system has enough free storage.

### Object scanning guidelines

Choose a rigid, opaque object with matte surfaces and visible texture whenever
possible. Transparent or reflective materials, nearly textureless surfaces,
very dark objects, and objects that deform while handled are substantially more
difficult to reconstruct. Use diffuse, stable lighting and a stationary,
feature-rich background; clean the object and make sure it can rest securely in
both scan orientations.

Use one continuous two-round recording to obtain six-face coverage:

1. Place the object near the center of the capture area with approximately
   20–30 cm of clearance. Keep it fully in frame, typically from a distance of
   50–100 cm.
2. Walk slowly through a full 360-degree pass at roughly the object's
   mid-height. Keep the camera aimed at the object, maintain a consistent
   distance, and target 50–70% overlap between consecutive views. A typical
   pass takes 20–40 seconds.
3. Without stopping the recording, rotate the object about 90 degrees to expose
   its previously hidden bottom face. Brief hand occlusion is expected; touch
   edges or the base where possible and keep most of the object and background
   visible.
4. Complete a second 360-degree pass around the new orientation, concentrating
   on the newly exposed face and its adjacent surfaces. Confirm that all six
   faces and any concave or easily missed areas have coverage, then stop and
   verify the capture was saved.

Move smoothly to limit motion blur and avoid abrupt changes in distance or
lighting.

---

## Input Data Format

`--mapping_data_dir` must point to a synchronized stereo dataset in this layout:

```text
mapping_data_dir/
├── frames_meta.json                         # required
├── frame_metadata.jsonl                     # optional compatibility metadata; not consumed
├── front_stereo_camera_left/
│   └── <timestamp_ns>.jpeg
└── front_stereo_camera_right/
    └── <timestamp_ns>.jpeg
```

The bundled [`assets/basketball_example/`](assets/basketball_example/) is the
reference dataset (203 synchronized stereo pairs at 960×600). JPEG is the
known-good input format. Metadata image paths are relative to
`mapping_data_dir`.

The reference directory also contains a `stereo.edex` source artifact, but it
is not part of this module's input contract and the current reconstruction path
does not read it. `frames_meta.json` is the calibration source of truth: its
camera entries provide the rectified projection matrices and image dimensions,
and its first `stereo_pair` entry provides the baseline.

This is a **CuSFM-compatible HOI stereo profile**. CuSFM determines the core
`frames_meta.json` representation, relative image paths, timestamp units,
camera parameter map, and synchronization IDs. The HOI wrapper additionally
requires one usable stereo pair, the sensor names shown below, rectified 3×4
projection matrices, and a positive baseline. JPEG is the currently tested
image format for the complete HOI pipeline even though CuSFM itself accepts
additional image formats.

### `frames_meta.json` (required)

This file is the pipeline's input contract:

| Field | Required | Description |
|-------|----------|-------------|
| `keyframes_metadata` | yes | One entry per camera image: camera ID, synchronized sample ID, relative image path, and timestamp in microseconds. |
| `camera_params_id_to_camera_params` | yes | Camera definitions keyed by ID, including sensor name, image dimensions, and calibration matrices. |
| `stereo_pair` | yes | Left/right camera IDs and physical stereo baseline in meters. The pipeline uses the first pair. |
| `initial_pose_type` | yes | CuSFM pose interpretation, normally `"EGO_MOTION"` for sequential stereo input. |
| `camera_params_id_to_session_name` | no | Producer/session metadata; not interpreted here. |

Minimal structural example (projection values abbreviated):

```json
{
  "initial_pose_type": "EGO_MOTION",
  "keyframes_metadata": [
    {
      "id": "0",
      "camera_params_id": "0",
      "synced_sample_id": "5",
      "image_name": "front_stereo_camera_left/1771445398921412053.jpeg",
      "timestamp_microseconds": "1771445398921412"
    },
    {
      "id": "1",
      "camera_params_id": "1",
      "synced_sample_id": "5",
      "image_name": "front_stereo_camera_right/1771445398921412053.jpeg",
      "timestamp_microseconds": "1771445398921412"
    }
  ],
  "camera_params_id_to_camera_params": {
    "0": {
      "sensor_meta_data": {"sensor_name": "front_stereo_camera_left"},
      "calibration_parameters": {
        "image_width": 960,
        "image_height": 600,
        "projection_matrix": {
          "row_count": 3,
          "column_count": 4,
          "data": [426.2, 0, 473.2, 0, 0, 426.2, 278.4, 0, 0, 0, 1, 0]
        }
      }
    },
    "1": {
      "sensor_meta_data": {"sensor_name": "front_stereo_camera_right"},
      "calibration_parameters": {
        "image_width": 960,
        "image_height": 600,
        "projection_matrix": {
          "row_count": 3,
          "column_count": 4,
          "data": [426.2, 0, 473.2, 0, 0, 426.2, 278.4, 0, 0, 0, 1, 0]
        }
      }
    }
  },
  "stereo_pair": [{
    "left_camera_param_id": "0",
    "right_camera_param_id": "1",
    "baseline_meters": 0.14956
  }]
}
```

Important invariants:

- Left and right entries pair by equal `synced_sample_id`; incomplete pairs are
  dropped.
- Camera IDs in frame and stereo-pair entries must reference keys in
  `camera_params_id_to_camera_params`.
- Sensor names must be `front_stereo_camera_left` and
  `front_stereo_camera_right`; current frame indexing relies on these names.
- `timestamp_microseconds` is a decimal string. Timestamp-based filenames use
  nanoseconds, but `image_name` is the authoritative path.
- `projection_matrix.data` is a row-major 3×4 rectified projection matrix. The
  pipeline reads `fx`, `fy`, `cx`, and `cy` from indices 0, 5, 2, and 6.
- `baseline_meters` is in meters and must be positive.

See the machine-readable
[`schemas/frames_meta.schema.json`](schemas/frames_meta.schema.json). JSON
Schema cannot verify cross-field references, stereo pairing, image existence,
or calibration accuracy.

### `frame_metadata.jsonl` (optional compatibility metadata)

The current `run_reconstruction.py` path does not read this file, but some
dataset producers and older tools emit it. Each non-empty line is one
synchronized-frame JSON record:

```json
{"frame_id":0,"cams":[{"id":0,"filename":"front_stereo_camera_left/1771445398921412053.jpeg","timestamp":1771445398921412000},{"id":1,"filename":"front_stereo_camera_right/1771445398921412053.jpeg","timestamp":1771445398921412000}]}
```

`timestamp` is in nanoseconds. Validate each line independently against
[`schemas/frame_metadata_record.schema.json`](schemas/frame_metadata_record.schema.json).
A JSONL file as a whole is not a single JSON value.

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

The `v2d_sam3d` image built from this tree includes the EGL/GLVND loader and
Pyrender required for headless overlay-video rendering. Rebuild that image
after changing `v2d_sam3d` code or Docker dependencies; model weights remain a
separate runtime download.

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
2b. CuSFM scan quality  → sfm_scan_quality/result.json  (two-loop pose check)
    ↓
2c. Stage-1 auto-detect → stage1_detect_debug/result.json  (transition frame)
    ↓
3.  Grounding DINO      → grounding_dino_bboxes.json
4a. FoundationStereo    → depth/  (all frames, parallel workers)
4b. SAM2                → masks/  (all frames, from reference frame)
    ↓
5.  Stage-1 setup       → stage1_recon/  (SfM keyframes + depth symlinks)
6.  BundleSDF NeRF      → stage1_recon/textured_mesh.obj + output.glb
7.  Center mesh         → mesh_input.obj
    ↓
8.  FoundationPose      → poses/  (tracking with Stage-1 mesh)
8b. FP render           → fp_render/render.mp4
    ↓
9.  World poses         → poses_world.json  (T_world_from_obj + stage detection)
10. Merged setup        → merged_recon/  (both stages aligned)
11. BundleSDF NeRF      → merged_recon/textured_mesh.obj + output.glb
    ↓
12. FoundationPose      → poses_final/  (tracking with final mesh)
13. FP render           → fp_render_final/render.mp4
```

### Results

- `merged_recon/textured_mesh.obj` — final textured mesh (+ `.mtl`, `_0.png`)
- `merged_recon/output.glb` — self-contained GLB exported from the final textured mesh
- `stage1_recon/textured_mesh.obj` — Stage-1 mesh (bottom missing)
- `stage1_recon/output.glb` — self-contained GLB exported from the Stage-1 mesh

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
--skip_prepare  --skip_sfm  --skip_sfm_quality_check  --skip_stage1_detect  --skip_dino  --skip_depth  --skip_mask
--skip_stage1_setup  --skip_stage1_nerf  --skip_center_mesh
--skip_fp_tracking  --skip_fp_render  --skip_world_poses  --skip_merged_setup  --skip_full_nerf
--skip_glb_export  --skip_final_fp_tracking  --skip_final_fp_render
```

### Configuration

**Pipeline config** — `lib/data/configs/hoi_pipeline.yaml` (override with `--pipeline_config`):

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `stage1_detect` | `buffer_deg` | 10.0 | Angle buffer (°) before detected Stage-1 transition |
| `sfm` | `config_set` | `backpack` | CuSFM preset (`backpack` \| `av` \| `isaac` \| `rgbd`) |
| `sfm_scan_quality` | `enabled` | `true` | Validate CuSFM poses before automatic stage split |
| `sfm_scan_quality` | `min_angle_span_deg` | 600.0 | Minimum projected orbit span for the expected two-loop scan |
| `sfm_scan_quality` | `max_backtracking_fraction` | 0.25 | Maximum reverse angular motion fraction |
| `sfm_scan_quality` | `max_translation_step_m` | 2.0 | Maximum allowed consecutive CuSFM translation jump |
| `depth` | `num_workers` | 1 | FoundationStereo depth workers; increase explicitly for multi-GPU hosts |
| `foundationpose` | `reference_frame` | 0 | FP registration reference frame |
| `foundationpose` | `weights_dir` | `null` | FP weights path; `null` = `data/weights/foundationpose` |

**NeRF/SDF config** — `modules/v2d_bundlesdf/lib/data/configs/theseus_optimizer_hawk.yaml` (override with `--config`):

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `camera_config` | `step` | 4 | CuSFM keyframe subsampling for SDF training |
| `nerf` | `n_step` | 3000 | SDF training steps |
| `nerf` | `far` | `auto` | SDF training depth range, resolved from masked object depth |
| `nerf` | `trunc` | 0.01 | TSDF truncation distance (normalized). Larger = fewer holes |
| `nerf` | `mesh_resolution` | 0.005 | Voxel size for mesh extraction. Must be ≤ `trunc` |
| `texture_bake` | `texture_res` | 2048 | Output texture atlas resolution |
| `texture_bake` | `zfar` | `auto` | Texture-renderer clipping plane, resolved from camera distance |
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
2b. CuSFM scan quality  → sfm_scan_quality/result.json  (two-loop pose check)
    ↓
2c. Stage-1 auto-detect → stage1_detect_debug/result.json  (exclude transition frames)
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
S6. Select best         → sam3d/best/best_frame.json + output_scaled.glb
                          (suggested candidate; still inspect per-frame videos)
```

### Results

- `sam3d/<frame_id>/mesh.glb` — raw SAM3D mesh (SAM3D camera space)
- `sam3d/<frame_id>/srt/output_scaled.glb` — scale-corrected mesh (world space)
- `sam3d/<frame_id>/srt/srt_result.json` — estimated scale, rotation, translation
- `sam3d/<frame_id>/render_debug.jpg` — SAM3D mesh overlaid on source image (single frame)
- `sam3d/<frame_id>/render_video.mp4` — textured mesh overlaid on all Stage-1 keyframes
- `sam3d/best/best_frame.json` — ranked candidate frames and selection score
- `sam3d/best/output_scaled.glb` — copied suggested best aligned mesh

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sam3d_use_depth` | off | Use FoundationStereo depth as extra loss in SRT scale estimation |
| `--sam3d_bin_deg DEG` | 60.0 | Azimuthal bin size for frame selection |
| `--sam3d_seed N` | 42 | Random seed for SAM3D inference |
| `--sam3d_srt_max_views N` | 25 | Maximum silhouette views per SRT candidate |
| `--sam3d_srt_maxiter N` | 60 | Powell iterations per SRT optimisation |
| `--sam3d_srt_top_k N` | 1 | Axis orientations fully optimised when no SAM3D orientation prior is available |
| `--sam3d_srt_parallel N` | 8 | Candidate-level SRT workers; each worker caps native numerical threads to one |
| `--sam3d_force_srt` | off | Recompute candidates that already have a valid result and scaled mesh |

The SRT defaults match the OSMO fast path. Candidates are independent and run
in parallel. A resumed run automatically reuses a candidate when both
`srt_result.json` and `output_scaled.glb` are valid; use `--sam3d_force_srt` to
deliberately recompute them. The underlying library retains its higher-cost
100-view, 120-iteration, top-5 defaults for direct callers that need the full
accuracy reference.

### Skip Flags (SAM3D)

```
--skip_prepare  --skip_sfm  --skip_sfm_quality_check  --skip_stage1_detect  --skip_dino  --skip_depth  --skip_mask
--skip_select_frames  --skip_sam3d  --skip_srt_scale  --skip_render_debug  --skip_render_video
--skip_select_best_mesh
```

---

## Quality Envelope and Limitations

The current generated-mesh inventory is a practical guide, not a formal
benchmark. In general, the pipeline works best on rigid, opaque household
objects with enough visible surface area and texture for segmentation, stereo
depth, and camera tracking: boxes, bottles, cans, balls, mugs, toy tools, and
similar tabletop or carryable objects.

Use extra visual QA for these cases:

- **Thin or high-aspect-ratio geometry** — hoops, rackets, swords, canes, chair
  legs, handles, and tool tips are easy to miss, thicken, fuse to nearby
  surfaces, or reconstruct with holes.
- **Large support-like objects** — desks, tables, platforms, crates, and large
  boxes need broad view coverage. Partial views often leave missing backs,
  undersides, or weak texture alignment.
- **Reflective, dark, transparent, or textureless surfaces** — depth and SfM can
  become unstable, and masks may leak onto background or hands.
- **Occluded hand-object interaction frames** — hands can hide important object
  surfaces; BundleSDF quality depends on clean masks and coherent stereo depth
  across the selected scan frames.
- **Two-stage alignment failures** — BundleSDF final meshes depend on
  FoundationPose staying locked between the stationary and rotated stages. Drift
  usually shows up in `fp_render/render.mp4` or `poses_world_debug.png`.
- **SAM3D scale and back-side geometry** — SAM3D is useful as a quick fallback
  and for representative-frame meshes, but scale is estimated after inference
  and unseen geometry can be less reliable. Check `render_debug.jpg` and
  `render_video.mp4` before treating the mesh as final.

For production assets, prefer an available scanner mesh as the geometric
reference. Treat HOI-generated meshes as requiring inspection with the overlay,
spin, point-cloud, and chamfer tools before accepting them.

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

4. **No zero surface** — If BundleSDF logs
   `Surface level must be within volume data range`, inspect
   `stage1_recon/resolved_config.yaml`.
   - `nerf.far` must cover the masked object depth range during SDF training.
   - The default config resolves `nerf.far: auto` from masked depth; if using a custom config, use `nerf.far: auto` or set an explicit larger value.
   - This is separate from `texture_bake.zfar`, which only affects texture rendering after a mesh exists.

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

### SAM3D: `libEGL.so.1` missing during overlay-video rendering

If SRT completes but `render_textured_video` exits with code 139 and reports
`Could not find library libEGL.so.1`, the local `v2d_sam3d` image predates the
EGL/Pyrender renderer dependencies. Rebuild only that image:

```bash
python modules/v2d_sam3d/docker/build.py
```

Then rerun the same job with the documented skip flags. Complete SRT
candidates are reused automatically, so rebuilding the renderer does not
require regenerating meshes or scale results.

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
