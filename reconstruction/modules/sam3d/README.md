# SAM3D Module

SAM3D for generating 3D meshes from images and masks.

## Functions

### image_to_mesh
Generate a 3D mesh from an image and a corresponding object mask.

**Inputs:**
- `image_path`: Path to input image.
- `mask_path`: Path to input mask image.
- `mesh_path`: Path to save output GLB mesh.
- `transform_path`: Path to save output transform JSON.
- `intrinsics_path`: Path to save output intrinsics JSON.
- `seed`: Random seed (optional).
- `stage1_only`: Only run stage 1 (flag).
- ... other optional flags for post-processing.

**Outputs:**
- 3D mesh in GLB format.
- Transform JSON (rotation, translation, scale).
- Camera intrinsics JSON.

### render_debug_image
Render a debug visualization showing the projected mesh vertices on the image.

**Inputs:**
- `image_path`: Path to original image.
- `mesh_path`: Path to GLB mesh.
- `transform_path`: Path to transform JSON.
- `intrinsics_path`: Path to intrinsics JSON.
- `output_image_path`: Path to save the debug visualization.
- `num_vertices_to_use`: Number of vertices to project (default: 5000).

**Outputs:**
- Visualization image (JPG/PNG).

## Usage via Docker Compose

### Shared Data Volume
Place your data in the root `data/` directory. It is mounted to `/data` inside the container.

### Manual Execution (Exec Profile)
Run on custom data:
```bash
# Image to mesh
docker compose run --profile exec sam3d-image-to-mesh \
  --image_path /data/image.jpg \
  --mask_path /data/mask.png \
  --mesh_path /data/output/mesh.glb \
  --transform_path /data/output/transform.json \
  --intrinsics_path /data/output/intrinsics.json

# Render debug image
docker compose run --profile exec sam3d-render-debug \
  /data/image.jpg \
  /data/output/mesh.glb \
  /data/output/transform.json \
  /data/output/intrinsics.json \
  /data/output/debug.jpg
```

### Running Tests (Tests Profile)
```bash
docker compose run --profile tests sam3d-image-to-mesh-test
```

### Launching Workers (Workers Profile)
```bash
# Image to mesh worker
docker compose --profile workers up sam3d-image-to-mesh-worker

# Render debug worker
docker compose --profile workers up sam3d-render-debug-worker
```

### HTTP API Server (API Profile)
Start the HTTP API server on port 8004:
```bash
docker compose --profile api up sam3d-api
```

**Endpoints:**
- `POST /process/image_to_mesh` - Files: `image`, `mask`, Optional form fields: `seed`, `stage1_only`, `with_mesh_postprocess`, etc.
- `POST /process/render_debug` - Files: `image`, `mesh`, `transform`, `intrinsics`, Optional: `num_vertices_to_use` (form field)

**Example using curl:**
```bash
# Image to mesh
curl -X POST http://localhost:8004/process/image_to_mesh \
  -F "image=@/path/to/image.jpg" \
  -F "mask=@/path/to/mask.png" \
  --output results.zip

# Render debug
curl -X POST http://localhost:8004/process/render_debug \
  -F "image=@/path/to/image.jpg" \
  -F "mesh=@/path/to/mesh.glb" \
  -F "transform=@/path/to/transform.json" \
  -F "intrinsics=@/path/to/intrinsics.json" \
  --output results.zip
```

