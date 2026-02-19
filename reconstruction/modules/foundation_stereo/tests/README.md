# Foundation Stereo Tests

## Test Data

`test_data/` contains 3 synchronized stereo image pairs:

```
test_data/
  left/    — 3 JPEG images (1920×1200)
  right/   — 3 JPEG images (matching filenames)
  calibration.json — camera intrinsics and stereo baseline
```

Calibration (`calibration.json`):
- `fx`, `fy`: 844.68 px
- `cx`, `cy`: 927.30, 566.74 px
- `baseline`: 0.15 m

## Running the Integration Test

Requires the TensorRT engine to be built first (see below). Output is written to
`reconstruction/data/test_output/`.

```bash
cd reconstruction

docker compose run --rm \
  -v "$(pwd)/modules/foundation_stereo/tests/test_data:/test_data:ro" \
  foundation-stereo-image-list-to-depth \
  --left_dir /test_data/left \
  --right_dir /test_data/right \
  --calibration_file /test_data/calibration.json \
  --depth_folder /data/test_output/depth \
  --intrinsics_folder /data/test_output/intrinsics
```

Expected output:
```
Done. processed=3, skipped=0
```

Output files in `reconstruction/data/test_output/`:
```
depth/
  1707938736136244532.png   — uint16 PNG, pixel = depth in mm (max 65.535 m)
  1707938736169573532.png
  1707938736202920532.png
intrinsics/
  1707938736136244532.json  — CameraIntrinsics JSON
  1707938736169573532.json
  1707938736202920532.json
```

## TensorRT Engine

On first run the engine is built automatically from the ONNX file. This takes
**10–40 minutes** depending on GPU. The engine is cached at:

```
reconstruction/data/foundation_stereo/models/<model>_sm_<gpu>.engine
```

Subsequent runs load the cached engine and start in seconds.

The ONNX file must be downloaded first:

```bash
cd reconstruction
bash modules/foundation_stereo/download.sh
```

## Depth Image Format

Depth PNGs use **uint16 inverse depth encoding**:

```python
# pixel = 65535 * (1 / (depth_m + 1))
# closer → higher pixel value; zero depth → 65535; infinity → 0
```

Load with:
```python
from modules.common.datatypes import DepthImage
depth = DepthImage.load("depth.png")  # depth.depth is float32 meters
```
