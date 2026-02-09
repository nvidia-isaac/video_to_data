# FoundationPose Module

FoundationPose for 6D object pose estimation and tracking.

## Functions

### video_to_poses
Track 6D object poses through a video sequence given a 3D mesh.

**Inputs:**
- `video_path`: Path to input video.
- `depth_folder`: Folder containing depth images.
- `masks_folder`: Folder containing mask images.
- `camera_intrinsics_path`: Path to camera intrinsics JSON.
- `mesh_path`: Path to object mesh file (GLB/OBJ).
- `poses_dir`: Directory to save output pose JSONs.
- `reference_frame`: Frame index to initialize tracking (default: 0).

**Outputs:**
- Sequence of JSON files containing 4x4 pose matrices.

### align_mesh_scale
Refine the scale and translation of a mesh to match observed depth data.

**Inputs:**
- `--mesh`: Path to mesh GLB.
- `--depth`: Path to depth image.
- `--mask`: Path to object mask.
- `--intrinsics`: Path to camera intrinsics JSON.
- `--transform`: Path to original transform JSON.
- `--output-transform`: Path to save refined transform JSON.

**Outputs:**
- Refined transform JSON.
- Debug visualizations (original vs refined).

### transform_mesh
Apply scaling from a transform JSON to a mesh.

**Inputs:**
- `--input-mesh`: Path to input mesh.
- `--output-mesh`: Path to save scaled mesh.
- `--transform`: Path to transform JSON containing scale.

**Outputs:**
- Scaled 3D mesh.

### simplify_mesh
Simplify a 3D mesh using quadratic decimation.

**Inputs:**
- `--input-mesh`: Path to input mesh.
- `--output-mesh`: Path to save simplified mesh.
- `--faces`: Target face count (optional).
- `--factor`: Reduction factor (0.0 to 1.0, optional).

**Outputs:**
- Simplified 3D mesh.

## Usage via Docker Compose

### Shared Data Volume
Place your data in the root `data/` directory. It is mounted to `/data` inside the container.

### Manual Execution (Exec Profile)
Run on custom data:
```bash
# Video to poses
docker compose run --profile exec foundationpose-video-to-poses \
  --video_path /data/video.mp4 \
  --depth_folder /data/output/depth \
  --masks_folder /data/output/masks \
  --camera_intrinsics_path /data/output/intrinsics.json \
  --mesh_path /data/output/mesh.glb \
  --poses_dir /data/output/poses

# Align mesh scale
docker compose run --profile exec foundationpose-align-mesh \
  --mesh /data/mesh.glb \
  --depth /data/depth.png \
  --mask /data/mask.png \
  --intrinsics /data/intrinsics.json \
  --transform /data/transform.json \
  --output-transform /data/refined_transform.json
```

### Running Tests (Tests Profile)
*(Note: No standard automated test service currently defined for this module in docker-compose.yaml)*

### Launching Workers (Workers Profile)
```bash
# Video to poses worker (GPU)
docker compose --profile workers up foundationpose-video-to-poses-worker

# Align mesh worker (GPU)
docker compose --profile workers up foundationpose-align-mesh-worker

# Transform mesh worker (CPU)
docker compose --profile workers up foundationpose-transform-mesh-worker

# Simplify mesh worker (CPU)
docker compose --profile workers up foundationpose-simplify-mesh-worker
```

### HTTP API Server (API Profile)
Start the HTTP API server on port 8005:
```bash
docker compose --profile api up foundationpose-api
```

**Endpoints:**
- `POST /process/video_to_poses` - Files: `video`, `mesh`, Optional: `depth_folder`, `masks_folder`, `camera_intrinsics`, `reference_frame`, `target_width`, `target_height`
- `POST /process/align_mesh` - Files: `mesh`, `depth`, `mask`, `intrinsics`, `transform`
- `POST /process/transform_mesh` - Files: `input_mesh`, `transform`
- `POST /process/simplify_mesh` - Files: `input_mesh`, Optional: `faces`, `factor` (form fields)

**Example using curl:**
```bash
# Align mesh
curl -X POST http://localhost:8005/process/align_mesh \
  -F "mesh=@/path/to/mesh.glb" \
  -F "depth=@/path/to/depth.png" \
  -F "mask=@/path/to/mask.png" \
  -F "intrinsics=@/path/to/intrinsics.json" \
  -F "transform=@/path/to/transform.json" \
  --output results.zip

# Simplify mesh
curl -X POST http://localhost:8005/process/simplify_mesh \
  -F "input_mesh=@/path/to/mesh.glb" \
  -F "faces=5000" \
  --output results.zip
```

