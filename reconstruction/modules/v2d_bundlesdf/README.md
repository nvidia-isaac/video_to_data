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

## Config Resolution

`v2d_bundlesdf.lib.reconstruct` reads the YAML config, resolves v2d policy
fields into concrete BundleSDF values, and writes the effective config to
`resolved_config.yaml`.

Important depth-range fields:

| Field | Purpose |
|-------|---------|
| `nerf.far` | SDF training depth range in meters. The default `auto` value is resolved from masked object depth so far objects are still supervised. |
| `texture_bake.zfar` | Texture-renderer clipping range. This is only used after a mesh exists. |

The default config also contains optional `auto_tune.trunc` and
`auto_tune.mesh_resolution` policies. They are disabled by default; enable them
only when intentionally tuning surface thickness or mesh extraction resolution.

## Outputs

Results are written to `output_path/`:

| File | Description |
|------|-------------|
| `textured_mesh.obj` | Final textured mesh (+ `.mtl`, `_0.png` atlas) |
| `output.glb` | Self-contained GLB exported from `textured_mesh.obj` |
| `mesh_cleaned.obj` | Untextured SDF mesh |
| `model_latest.pth` | Saved SDF model (reusable with `--skip-sdf`) |
| `resolved_config.yaml` | Effective config after auto policy resolution |
| `run_time.yaml` | Timing breakdown |

## Tools

| Tool | Description |
|------|-------------|
| `tools/visualize_reconstruction_standalone.py` | Visualize camera trajectory and point cloud |
| `tools/fuse_depth_to_pointcloud.py` | Fuse depth maps into a point cloud |
