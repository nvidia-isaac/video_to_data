# Grounding DINO Tests

## Test Data

- `reconstruction/test_data/test_frame.jpg` — single extracted frame (2048×1536)
- `reconstruction/test_data/test_video.mp4` — 100-frame video (2048×1536)

## Checkpoint

The scripts auto-download `groundingdino_swint_ogc.pth` (~694 MB) to
`reconstruction/data/grounding_dino/models/` on first run.

To download manually in advance:
```bash
cd reconstruction
bash modules/grounding_dino/download.sh
```

## Running Tests

All examples use `--vis` to save annotated images to `reconstruction/data/test_output/debug/`.

### Single image → JSON

```bash
cd reconstruction

docker compose run --rm \
  -v "$(pwd)/test_data:/test_data:ro" \
  grounding-dino-image-to-object-bboxes \
  --image_path /test_data/test_frame.jpg \
  --output_path /data/test_output/gdino_frame_bboxes.json \
  --prompt "person" \
  --vis
```

Output: `reconstruction/data/test_output/gdino_frame_bboxes.json`
```json
[
  {"label": "person", "confidence": 0.83, "box": {"x0": 1015.5, "y0": 606.7, "x1": 1212.7, "y1": 1250.4}}
]
```
Annotated image: `reconstruction/data/test_output/debug/test_frame.jpg`

### Video → single JSON (keyed by frame index)

```bash
cd reconstruction

docker compose run --rm \
  -v "$(pwd)/test_data:/test_data:ro" \
  grounding-dino-video-to-object-bboxes \
  --video_path /test_data/test_video.mp4 \
  --output_path /data/test_output/gdino_video_bboxes.json \
  --prompt "person" \
  --vis
```

Output: `reconstruction/data/test_output/gdino_video_bboxes.json`
```json
{
  "0": [{"label": "person", "confidence": 0.84, "box": {...}}],
  "1": [{"label": "person", "confidence": 0.83, "box": {...}}],
  ...
}
```
Annotated frames: `reconstruction/data/test_output/debug/000000.jpg`, `000001.jpg`, ...

### Image list → single JSON (keyed by image stem)

```bash
cd reconstruction

docker compose run --rm \
  -v "$(pwd)/modules/foundation_stereo/tests/test_data/left:/test_data:ro" \
  grounding-dino-image-list-to-object-bboxes \
  --image_dir /test_data \
  --output_path /data/test_output/gdino_list_bboxes.json \
  --prompt "person" \
  --vis
```

Output: `reconstruction/data/test_output/gdino_list_bboxes.json`
```json
{
  "1707938736136244532": [...],
  "1707938736169573532": [...],
  "1707938736202920532": [...]
}
```
Annotated images: `reconstruction/data/test_output/debug/`

## Output Format

All outputs use the same detection schema:
```json
[
  {
    "label": "person",
    "confidence": 0.84,
    "box": {"x0": 1015.5, "y0": 606.7, "x1": 1212.7, "y1": 1250.4}
  }
]
```
- Boxes are **absolute pixel coordinates** (x0, y0 = top-left; x1, y1 = bottom-right)
- Sorted by confidence descending
- Empty list `[]` means no objects matched the prompt

## Pipeline with SAM2

The highest-confidence bounding box from `image_to_object_bboxes` / `image_list_to_object_bboxes`
feeds directly into `sam2-video-to-masks` as a prompt. Box format is compatible with
`Sam2Prompt.box` (`BoundingBox` with x0/y0/x1/y1).
