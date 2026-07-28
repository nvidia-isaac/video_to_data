# Offline Vega + Sharpa renderer scaffold

This package consumes the common `robot_trajectory.npz`, matching TACO camera
calibration, and the complete sibling robotic-grounding asset tree. It produces:

- `robot_rgb.mp4`: source-sized Vega arms plus two articulated Sharpa hands on a flat background;
- `robot_mask.npy`: `(N,H,W)` boolean robot occupancy;
- `robot_depth.npy`: `(N,H,W)` float32 positive camera-z depth, with `inf` off robot;
- `render_metadata.json`: geometry, calibration convention, immutable container
  image ID, SHA-256 input/asset/source/artifact provenance, IK diagnostics, and
  artifact sizes.

`render_metadata.json` is the multi-file commit marker. It changes to
`committing` before any final artifact rename and to `complete` only after all
three artifacts have been verified and installed; consumers must reject every
other state.

The renderer is offline and does not import Isaac. It uses `yourdfpy` for URDF
forward kinematics, `pyrender` for camera rendering, and the existing
`robotic-grounding:photo-render-v6` image for pinned runtime dependencies.

## Coordinate and validation contract

`--world-to-camera` is always required and must be exactly `(N,4,4)` rigid
`T_camera_world` matrices aligned one-to-one with source frames. Intrinsics must
be a `3x3` OpenCV matrix (or text `fx fy cx cy`) already scaled for the supplied
source width and height. The renderer never assumes identity calibration,
inverts an ambiguous extrinsic, or silently rescales intrinsics.

World-frame wrist poses pass through unchanged. Camera-frame wrist poses are
first transformed back to the common world frame using the matching inverse
extrinsic. This is necessary to mount one rigid Vega shoulder hub in the world;
placing a new hub independently in every camera frame would make the torso
float with the headset.

The v1 renderer requires both wrist tracks to be valid and finite on every
frame. It deliberately rejects gaps rather than filling or holding robot poses.
Finger names must exactly cover the corresponding 22-joint Sharpa URDF, and
values outside URDF limits are errors rather than clamped silently.

## Asset and IK provenance

Pass the complete sibling directory ending in
`robotic_grounding/robotic_grounding/assets` as `--asset-root`. The current
worktree's Sharpa meshes are Git LFS pointers and its Vega mesh tree is absent;
they are intentionally not used. The container mounts the complete sibling
tree read-only.

The sidecar hashes the trajectory, intrinsics, world-to-camera calibration,
all three URDFs, every directly or transitively referenced mesh resource, and
every production Python source file in this renderer package. Asset/source
records use stable repository- or asset-root-relative paths and sorted order.
Completed RGB, mask, and depth artifacts are hashed only after their final
rename. The host wrapper resolves the human image tag through local
`docker image inspect`, runs that immutable `sha256:...` image ID, and passes
both the tag and ID into metadata.

`arm_ik.py` and `arm_mount_opt.py` are loaded explicitly from
`--scene-utils-root` under a synthetic private package. This never executes the
`robotic_grounding` package initializer. The real `arm_replay.py` cannot be
imported safely in isolation because it immediately imports an absolute
Isaac-oriented replay module, so the loader injects an equivalent implementation
of its 13-line `build_hand_mount_inverses` helper. The physical l8-to-C_MC mount
constants and IK settings are copied verbatim from sibling `assets/vega_arms.py`
and recorded in dry-run/full metadata.

If `--arm-center-world` is absent, the renderer calls the existing
`arm_mount_opt.place_hub_from_wrists` world-frame search and then runs the
existing global `ArmIK.solve_trajectory`. An explicit 4x4 `T_world_arm-center`
can be supplied for a calibrated installation. Full rendering fails when the
attachment residual or frame-to-frame arm-joint step exceeds its configured
threshold.

The defaults are 0.01 m maximum attachment residual and 0.4 rad maximum
frame-to-frame arm-joint step. The generic visibility guard remains a
blank/nearly-blank safety check (at least 10% of frames over its pixel
threshold); it does not impose all-frame visibility on arbitrary future clips.

## Dry-run calibration check

The safe first command imports no Pinocchio, Pink, OpenGL, or GPU libraries and
creates no output artifacts:

```bash
python3 -m inpainting.robot_renderer.container_runner \
  --trajectory /path/to/robot_trajectory.npz \
  --intrinsics /path/to/egocentric_intrinsic.txt \
  --world-to-camera /path/to/world_to_camera.npy \
  --width 1920 --height 1080 --fps 30 \
  --asset-root /path/to/robotic_grounding/robotic_grounding/assets \
  --scene-utils-root /path/to/robotic_grounding/tasks/scene_utils \
  --output-dir /path/to/robot \
  --dry-run
```

By default this prints a fully quoted Docker command. Add `--execute` to run the
CPU-only validation container.

The image's bundled Isaac Python tree is not traversable by an arbitrary
`--user` UID, so the wrapper uses the image's default runtime user. It passes the
caller's numeric UID/GID to the renderer and chowns only the four completed
output files back to that caller before exit.

For a full render, add an explicit available GPU and remove `--dry-run`:

```bash
python3 -m inpainting.robot_renderer.container_runner ... --gpu 0 --execute
```

The calibrated GT batch has now been rendered at full 1920x1080 resolution.
All three clips achieved observed 100% frame visibility while also passing the
tighter default IK gates:

| Sequence suffix | Visible frames | Max IK residual | Max joint step |
| --- | ---: | ---: | ---: |
| `20231013_105` | 152/152 | 0.251 mm | 0.156 rad/frame |
| `20231005_253` | 74/74 | 0.063 mm | 0.225 rad/frame |
| `20231031_060` | 155/155 | 0.524 mm | 0.309 rad/frame |

A CPU-only container probe also confirmed the pinned image contains `yourdfpy 0.0.60`,
`pyrender 0.1.45`, and `trimesh 4.5.1`, and that all three complete URDFs load.
The image does not put `ffmpeg`/`ffprobe` on `PATH`, so the backend uses its
bundled `imageio-ffmpeg 0.6.0` executable and verifies the decoded geometry,
frame count, and FPS through its FFmpeg-enabled OpenCV 4.11 build.

## Required visual QA

- The three calibrated GT renders were visually reviewed for camera convention,
  robot scale, hand/wrist following, and arm/head framing. A completed metadata
  sidecar still does not replace that review for a new trajectory or camera.
- Trajectories with invalid wrist frames need an explicitly approved upstream
  interpolation or renderer visibility policy before they can be consumed.

## Retrofitting completed v1 metadata

`enrich_metadata` verifies a sidecar is `complete`, belongs to its bundle,
matches the supplied inputs/assets, has exact byte counts, decodes to the
recorded RGB geometry/FPS/frame count, has valid mask/depth semantics and
mask-derived statistics, and has not changed during hashing. Only then does it
atomically replace `render_metadata.json`; RGB/mask/depth are never rewritten.
`--verify-only` performs the same work without replacing metadata.

The exact command used to migrate the current three GT bundles is:

```bash
REPO=/home/mverghese/visual_inpainting/video_to_data_internal
RUN="$REPO/inpainting/artifacts/runs/taco_hand_tracking_v1"
MANIFEST="$RUN/manifest.resolved.json"
ASSET_ROOT=/home/mverghese/video_to_data_internal/robotic_grounding/source/robotic_grounding/robotic_grounding/assets

jq -r '.sequences[] | [.sequence_id, .camera.intrinsic, .camera.extrinsic] | @tsv' "$MANIFEST" |
while IFS=$'\t' read -r sequence intrinsic extrinsic; do
  python3 -m inpainting.robot_renderer.enrich_metadata \
    --metadata "$RUN/$sequence/ground_truth/robot_render/render_metadata.json" \
    --trajectory "$RUN/$sequence/ground_truth/tracking/robot_trajectory.npz" \
    --intrinsics "$intrinsic" \
    --world-to-camera "$extrinsic" \
    --asset-root "$ASSET_ROOT" \
    --repository-root "$REPO" \
    --image robotic-grounding:photo-render-v6
done
```

For these older bundles, `capture_mode=retrospective_enrichment` is explicit:
artifact/input/asset hashes attest to the verified files now, while the source
hashes and locally resolved image ID cannot retroactively prove which source
bytes or mutable tag target existed at the historical render instant. New
renders record them at render/launch time.

## Unit tests

The tests use only temporary synthetic arrays/URDFs and require no GPU or real
assets:

```bash
python3 -m unittest discover -s inpainting/robot_renderer/tests -v
```
