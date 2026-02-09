# SAM2 Module

Segment Anything Model 2 for video object segmentation.

## Functions

### video_to_masks
Process a video with SAM2 prompts and save masks to files.

**Inputs:**
- `video_path`: Path to input video file.
- `prompts_path`: Path to JSON file containing SAM2 prompts.
- `masks_dir`: Directory to save output mask images.

**Outputs:**
- Sequence of PNG mask files in the specified `masks_dir`.

## Usage via Docker Compose

### Shared Data Volume
Place your data in the root `data/` directory. It is mounted to `/data` inside the container.

### Manual Execution (Exec Profile)
To run the segmentation on custom data:
```bash
docker compose run --profile exec sam2-video-to-masks \
  --video_path /data/your_video.mp4 \
  --prompts_path /data/your_prompts.json \
  --masks_dir /data/output/masks
```

### Running Tests (Tests Profile)
To run the built-in test with sample data:
```bash
docker compose run --profile tests sam2-video-to-masks-test
```

### Launching Workers (Workers Profile)
To start the Celery worker for this function:
```bash
docker compose --profile workers up sam2-video-to-masks-worker
```

### HTTP API Server (API Profile)
Start the HTTP API server on port 8001:
```bash
docker compose --profile api up sam2-api
```

**Endpoint:** `POST /process/video_to_masks`
- **Files:** `video` (video file), `prompts` (JSON file)
- **Returns:** ZIP file containing all output masks

**Example using curl:**
```bash
curl -X POST http://localhost:8001/process/video_to_masks \
  -F "video=@/path/to/video.mp4" \
  -F "prompts=@/path/to/prompts.json" \
  --output results.zip
```

