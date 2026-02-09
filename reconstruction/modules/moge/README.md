# MoGe Module

Monocular Geometry Estimation (MoGe) for depth and intrinsics estimation.

## Functions

### image_to_depth
Process a single image to estimate depth and camera intrinsics.

**Inputs:**
- `image_path`: Path to input image.
- `depth_path`: Path to save output depth image.
- `intrinsics_path`: Path to save output camera intrinsics JSON.

**Outputs:**
- Depth PNG file.
- Camera intrinsics JSON file.

### video_to_depth
Process a video to estimate depth frames and camera intrinsics.

**Inputs:**
- `video_path`: Path to input video.
- `depth_folder`: Folder to save output depth images.
- `intrinsics_folder`: Folder to save output camera intrinsics JSONs.
- `batch_size`: Number of frames to process in a batch (default: 8).

**Outputs:**
- Sequence of depth PNG files.
- Sequence of camera intrinsics JSON files.

## Usage via Docker Compose

### Shared Data Volume
Place your data in the root `data/` directory. It is mounted to `/data` inside the container.

### Manual Execution (Exec Profile)
Run on custom data:
```bash
# Image to depth
docker compose run --profile exec moge-image-to-depth \
  --image_path /data/image.jpg \
  --depth_path /data/output/depth.png \
  --intrinsics_path /data/output/intrinsics.json

# Video to depth
docker compose run --profile exec moge-video-to-depth \
  --video_path /data/video.mp4 \
  --depth_folder /data/output/depth \
  --intrinsics_folder /data/output/intrinsics
```

### Running Tests (Tests Profile)
```bash
docker compose run --profile tests moge-video-to-depth-test
```

### Launching Workers (Workers Profile)
```bash
# Video to depth worker
docker compose --profile workers up moge-video-to-depth-worker

# Image to depth worker
docker compose --profile workers up moge-image-to-depth-worker
```

### HTTP API Server (API Profile)
Start the HTTP API server on port 8002:
```bash
docker compose --profile api up moge-api
```

**Endpoints:**
- `POST /process/image_to_depth` - Files: `image`
- `POST /process/video_to_depth` - Files: `video`, Optional: `batch_size` (form field)

**Example using curl:**
```bash
# Image to depth
curl -X POST http://localhost:8002/process/image_to_depth \
  -F "image=@/path/to/image.jpg" \
  --output results.zip

# Video to depth
curl -X POST http://localhost:8002/process/video_to_depth \
  -F "video=@/path/to/video.mp4" \
  -F "batch_size=8" \
  --output results.zip
```

