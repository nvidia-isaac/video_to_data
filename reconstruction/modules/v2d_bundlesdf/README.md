# v2d_bundlesdf

BundleSDF SDF training and texture baking from pre-computed poses, depth, and masks.

## Usage

Run from `reconstruction/`:

```bash
python modules/v2d_bundlesdf/docker/run_reconstruct.py \
  --output_path data/outputs/bundlesdf/my_object \
  --weights_dir data/weights
```

By default, `output_path` must contain:

```
output_path/
├── keyframes.yml       # camera poses (YAML)
├── left/               # RGB images
├── depth/              # depth maps (one per keyframe)
├── masks/              # object masks (one per keyframe)
└── calibration.json    # camera intrinsics (optional)
```

### Custom input directories

Use these flags to point directly to existing directories/files instead of
relying on the default folder structure:

| Flag | Default | Description |
|------|---------|-------------|
| `--images_dir` | `<output_path>/left/` | RGB images directory |
| `--depth_dir` | `<output_path>/depth/` | Depth maps directory |
| `--masks_dir` | `<output_path>/masks/` | Object masks directory |
| `--poses_file` | `<output_path>/keyframes.yml` | Camera poses YAML file |
| `--intrinsics_file` | `<output_path>/calibration.json` | Camera intrinsics JSON file |

Example:

```bash
python modules/v2d_bundlesdf/docker/run_reconstruct.py \
  --output_path  data/outputs/bundlesdf/my_object \
  --weights_dir  data/weights \
  --images_dir   /data/raw/my_object/images \
  --depth_dir    /data/raw/my_object/depth \
  --masks_dir    /data/raw/my_object/masks \
  --poses_file   /data/raw/my_object/keyframes.yml \
  --intrinsics_file /data/raw/my_object/calibration.json
```

When custom paths are provided, symlinks are created inside `output_path` pointing
to those locations so BundleSDF can find them without copying data.

### Other flags

| Flag | Description |
|------|-------------|
| `--config` | NeRF/SDF config YAML (uses container default if omitted) |
| `--bbox_str` | Bounding box `x1,y1,x2,y2` (informational only) |
| `--skip-texture` | Skip texture baking; produce untextured mesh only |
| `--skip-sdf` | Skip SDF training; reuse existing `model_latest.pth` |
| `--gpu_id` | GPU index to use |
| `--dev` | Mount local modules for development |

## Outputs

Results are written to `output_path/`:

| File | Description |
|------|-------------|
| `textured_mesh.obj` | Final textured mesh (+ `.mtl`, `_0.png` atlas) |
| `mesh_cleaned.obj` | Untextured SDF mesh |
| `model_latest.pth` | Saved SDF model (reusable with `--skip-sdf`) |
| `run_time.yaml` | Timing breakdown |

## Tools

| Tool | Description |
|------|-------------|
| `tools/visualize_reconstruction_standalone.py` | Visualize camera trajectory and point cloud |
| `tools/fuse_depth_to_pointcloud.py` | Fuse depth maps into a point cloud |
