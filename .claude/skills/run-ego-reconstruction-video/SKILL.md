---
name: run-ego-reconstruction-video
description: Run this repository’s egocentric reconstruction pipeline on a video. Use when a user asks to process an MP4 with HaMeR or DynHaMR hand tracking, an object prompt or mesh, undistortion, DROID-SLAM, gravity alignment, gsplat refinement, or a Three.js result export.
---

# Run Ego Reconstruction Video

Run from `reconstruction/`. Prefer the consolidated entrypoint and use HaMeR
unless the user explicitly needs legacy DynHaMR:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video <video.mp4> \
  --output_dir data/outputs/<run> \
  --object_prompt "<object>" \
  --hand_tracking hamer \
  --reference_frame 0
```

Add `--object_mesh <mesh.obj> --skip_object_scale_estimation` for a supplied
mesh. Add `--undistort`, `--run_droid_slam`, `--run_gravity_alignment`,
`--run_gsplat_refinement`, and `--export_threejs_result` only when requested.

Re-run the same command to resume cached stages. Confirm `result/result.npz`
and `result/mesh.obj`; with post-processing, use the final suffixed result
directory. Use the setup skill first when images, weights, Docker, or GPU
prerequisites are missing.
