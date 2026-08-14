# Run Ego Reconstruction Video Reference

## Entrypoints

Use `reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py` for user-facing runs. It accepts both `--video` and `--video_path`, both `--object_prompt` and `--prompt`, and both `--object_mesh` and `--object_mesh_path`.

`--hand_tracking hamer` maps to `run_ego_wilor.py` with HaMeR as the primary hand source. It supports:

- `--object_prompt`
- `--object_mesh` with `--skip_object_scale_estimation`
- `--run_droid_slam`
- `--run_gravity_alignment`
- `--run_gsplat_refinement`
- `--export_threejs_result`

`--hand_tracking dynhamr` maps to `run_v2d_ego_e2e.py`. It supports prompt-based object reconstruction but not `--object_mesh` or `--run_gsplat_refinement` in the consolidated entrypoint.

## Common Output Map

HaMeR/WiLoR path commonly writes:

```text
<output_dir>/frames/
<output_dir>/depth/
<output_dir>/intrinsics/
<output_dir>/intrinsics_stable.json
<output_dir>/wilor_raw/
<output_dir>/hand_detections.json
<output_dir>/sam2_prompts.json
<output_dir>/hand_tracks.json
<output_dir>/masks/
<output_dir>/prompts_overlay.png
<output_dir>/masks_overlay.mp4
<output_dir>/hamer_aligned_filled/
<output_dir>/hamer_aligned_filled_overlay.mp4
<output_dir>/result/
```

Object branch outputs may include:

```text
<output_dir>/mesh/
<output_dir>/mesh_pretransformed.obj
<output_dir>/mesh_scaled.obj
<output_dir>/scale.json
<output_dir>/poses/
<output_dir>/poses_smoothed/
```

Optional postprocessing may include:

```text
<output_dir>/slam_poses/
<output_dir>/slam_trajectory.txt
<output_dir>/geocalib/
<output_dir>/result_slam/
<output_dir>/result_gravity_aligned/
<output_dir>/result_slam_gravity_aligned/
<final_result_dir>/threejs_scene/index.html
```

DynHaMR path commonly writes both MoGe and DynHaMR depth artifacts, then packages `result/`:

```text
<output_dir>/hand_reconstruction/
<output_dir>/depth/
<output_dir>/depth_vipe/
<output_dir>/intrinsics_stable.json
<output_dir>/intrinsics_vipe.json
<output_dir>/mesh_moge/ or mesh_vipe/
<output_dir>/poses_smoothed_moge/ or poses_smoothed_vipe/
<output_dir>/hamer_aligned_from_dynhamr_moge/
<output_dir>/result/
```

## Result Bundle Checks

A complete final result directory should include `result.npz` and `mesh.obj`.
When `--export_threejs_result` is enabled, the runner exports a browser scene to
`<final_result_dir>/threejs_scene/index.html` unless `--threejs_output_dir` is set.

If the requested final result is gravity aligned and DROID-SLAM is enabled, the
best final directory is usually `result_slam_gravity_aligned/`; otherwise prefer
`result_gravity_aligned/`, then `result_slam/`, then `result/`.

## Example From The Repo

`reconstruction/data/garage/tissue_box/run_wilor_pipeline.sh` runs:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --dev \
  --video data/garage/tissue_box/IMG_5969.mp4 \
  --output_dir data/garage/tissue_box/outputs \
  --object_prompt "a tissue box" \
  --reference_frame 0 \
  --undistort \
  --hand_tracking hamer \
  --run_droid_slam \
  --run_gravity_alignment \
  --run_gsplat_refinement \
  --export_threejs_result
```

Keep `--dev` only when live-editing source code mounted into containers.
