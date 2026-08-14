---
name: run-ego-reconstruction-video
description: Run the egocentric reconstruction pipeline on an input video. Use when a user asks Codex to process an MP4 or video with the reconstruction ego pipeline, choose between HaMeR/WiLoR and DynHaMR hand tracking, pass an object prompt or object mesh, enable undistortion, SLAM, gravity alignment, gsplat refinement, or export/check the final result bundle and Three.js scene.
---

# Run Ego Reconstruction Video

## Decide the Run Mode

Use the consolidated entrypoint unless the user explicitly asks for a legacy script:

```bash
cd reconstruction
python modules/v2d_pipelines/run_ego_reconstruction.py --help
```

Default to `--hand_tracking hamer` for new work. It uses WiLoR/SAM2/HaMeR,
supports a caller-provided object mesh, and supports gsplat refinement. Use
`--hand_tracking dynhamr` only when the user needs the legacy ViPE + DynHaMR path
and the manual MANO/BMC assets are present.

Before running, gather or infer:

- input video path, preferably absolute or relative to `reconstruction/`
- object prompt, such as `"a tissue box"`
- output directory
- whether the video is fisheye/wide angle and should use `--undistort`
- whether to use `--object_mesh` for an existing metric OBJ
- whether to enable `--run_droid_slam`, `--run_gravity_alignment`, `--run_gsplat_refinement`, or `--export_threejs_result`

If setup may be incomplete, use `$ego-reconstruction-setup` first.

## Standard Commands

Minimal HaMeR/WiLoR run:

```bash
cd reconstruction
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video data/path/to/video.mp4 \
  --output_dir data/outputs/my_run \
  --object_prompt "a handled object" \
  --hand_tracking hamer \
  --reference_frame 0
```

Fuller HaMeR/WiLoR run with common postprocessing:

```bash
cd reconstruction
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video data/path/to/video.mp4 \
  --output_dir data/outputs/my_run \
  --object_prompt "a handled object" \
  --hand_tracking hamer \
  --reference_frame 0 \
  --undistort \
  --run_droid_slam \
  --run_gravity_alignment \
  --run_gsplat_refinement \
  --export_threejs_result
```

HaMeR/WiLoR with a provided object mesh:

```bash
cd reconstruction
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video data/path/to/video.mp4 \
  --output_dir data/outputs/my_mesh_run \
  --object_prompt "a handled object" \
  --object_mesh /absolute/path/to/object.obj \
  --skip_object_scale_estimation \
  --hand_tracking hamer
```

DynHaMR path:

```bash
cd reconstruction
python modules/v2d_pipelines/run_ego_reconstruction.py \
  --video data/path/to/video.mp4 \
  --output_dir data/outputs/my_dynhamr_run \
  --object_prompt "a handled object" \
  --hand_tracking dynhamr \
  --undistort
```

Do not combine `--hand_tracking dynhamr` with `--object_mesh` or
`--run_gsplat_refinement`; the consolidated runner rejects those combinations.

## Reruns and Partial Outputs

Rerun the exact same command after an interruption. The runner checks many stage
outputs and prints `[skip]` for completed stages. Use `--dev` only when actively
editing module source and wanting containers to mount local code.

For repo examples, inspect `reconstruction/data/garage/tissue_box/run_wilor_pipeline.sh`.
Use it as a shape reference, not as a hard-coded path template.

## Verify Results

Expect the main portable bundle at one of these directories:

- `<output_dir>/result/`
- `<output_dir>/result_slam/` when DROID-SLAM is enabled
- `<output_dir>/result_gravity_aligned/` or `<output_dir>/result_slam_gravity_aligned/` when gravity alignment is enabled

A complete result bundle contains at least:

```text
result.npz
mesh.obj
```

If `--export_threejs_result` was used, inspect:

```text
<final_result_dir>/threejs_scene/index.html
```

Useful quick checks:

```bash
test -s data/outputs/my_run/result/result.npz
test -s data/outputs/my_run/result/mesh.obj
ls data/outputs/my_run/*overlay*.mp4
```

## Debugging

- Missing Python module: re-run `bash reconstruction/scripts/install_ego_reconstruction_packages.sh` in the active environment.
- Missing Docker image: re-run `bash reconstruction/scripts/build_ego_reconstruction_packages.sh`.
- Missing weights: use `$ego-reconstruction-setup` and choose the download mode that matches the run mode.
- No object detection: adjust `--object_prompt` and/or `--reference_frame`.
- Bad fisheye geometry: add `--undistort` and re-run into a fresh output directory.
- Bad final motion or background camera: try `--run_droid_slam`; add `--run_gravity_alignment` when world-up matters.
- Refinement failures: first produce a non-refined `result/`, then add `--run_gsplat_refinement` after the base run is healthy.

See `references/run-ego-reconstruction-video.md` for entrypoint details and output map.
