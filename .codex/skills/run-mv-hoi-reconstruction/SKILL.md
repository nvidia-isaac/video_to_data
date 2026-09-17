---
name: run-mv-hoi-reconstruction
description: Run and validate the repository-local multi-view camera calibration and human-object reconstruction pipelines. Use when Codex is asked to prepare, launch, monitor, troubleshoot, or verify `v2d.pipelines.run_mv_calibration` or `v2d.pipelines.run_mv_hoi_reconstruction`, including calibration EDEX generation and the Grounding DINO reconstruction path.
---

# Run MV Calibration and HOI Reconstruction

Use the local Docker-orchestrated runners to calibrate the multi-view rig and
reconstruct one multi-view sequence. Keep the Grounding DINO prompt path as the
only supported reconstruction object-detection path.

## Read the Canonical Runbook

Locate the repository root containing this skill, then read
`reconstruction/docs/mv_hoi_local_pipeline.md` completely before preparing or
running the pipeline. Treat that file and the current runner CLI as the source
of truth; do not duplicate or guess commands from memory.

This skill covers local calibration, reconstruction, and diagnostics only. Do
not access the production database, CSS, OSMO campaigns, HITL, QC, trimming, or
final export.

## Choose the Workflow

- For calibration, gather a calibration rosbag directory and a fresh output
  directory, then run and validate `v2d.pipelines.run_mv_calibration`.
- For reconstruction, use a matching validated calibration EDEX. Reuse an
  existing one when the rig has not changed; otherwise complete calibration
  first.

## Calibrate the Rig

Obtain or infer absolute paths for a calibration directory containing a
nonempty `.mcap` and a new output path that does not exist. Calibration needs
the `v2d_rosbag` and `v2d_mv_calibration` images, but no GPU or model weights.
Confirm both images are available with `docker image inspect`. If either is
missing, cite the build command from the runbook; do not build it without user
authorization.

Use `stereo4_6x10_100mm_marker` by default. Select another packaged setup only
when the physical board is known to differ. From `reconstruction`, run:

```bash
python -m v2d.pipelines.run_mv_calibration \
  --rosbag_path "$CALIBRATION_SEQUENCE_DIR" \
  --output_dir "$CALIBRATION_OUTPUT_DIR"
```

Add `--dev` only when the user asks to run checked-out module source. Monitor
both extraction and extrinsic-calibration stages and provide progress updates
at each stage or at least once per minute. Require the completion banner and
validate these nonempty JSON artifacts:

```text
raw/edex
extrinsics/edex
extrinsics/calibration_accuracy.json
```

Review the bundle-adjustment reprojection statistics before using the result.
On failure, preserve the output and retry the complete calibration only with a
new output directory after addressing the cause. Do not use partial extrinsics.

## Gather the Reconstruction Inputs

Obtain or infer these absolute paths:

- sequence directory containing an MCAP and `hoi_metadata.yaml`
- calibration `edex` JSON file
- `output_aligned.glb` in a standalone object-mesh directory
- a new output path that does not exist

Infer paths from the user's sequence and nearby calibration/object assets when
the choice is unambiguous. Ask only when multiple valid calibration or mesh
assets exist. Never put the output inside the sequence or mesh directory.

## Run Reconstruction Preflight

Use the repository reconstruction virtual environment when it exists:

```bash
reconstruction/.venv/bin/python \
  .codex/skills/run-mv-hoi-reconstruction/scripts/preflight.py \
  --sequence-dir /absolute/path/to/sequence \
  --calibration-edex /absolute/path/to/calibration/edex \
  --object-mesh /absolute/path/to/object_mesh/output_aligned.glb \
  --output-dir /absolute/path/to/new_output
```

The helper is read-only. It validates inputs, the Grounding DINO prompt,
weights, images, Docker, GPU access, and output-path separation. Resolve every
error before launch. Treat warnings about SOMA-X assets or TensorRT engines as
setup guidance, not permission to download or rebuild automatically.

If setup is missing, cite the exact runbook command. Do not build images,
download gated models, accept a model license, or force-rebuild TensorRT engines
without explicit user authorization.

## Launch and Monitor Reconstruction

Run from `reconstruction` with the active virtual environment:

```bash
python -m v2d.pipelines.run_mv_hoi_reconstruction \
  --rosbag_path "$SEQUENCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --calibration_camera_params_path "$CALIBRATION_EDEX" \
  --obj_mesh_path "$OBJECT_MESH"
```

Use `--dev` only when the user asks to run live checked-out source rather than
the code baked into the local images.

Start the command in a persistent execution session. Monitor it until terminal,
and give the user a concise update at each major stage or at least once per
minute. Do not treat long FoundationPose, SAM3D Body, or SOMA-X compute as a
hang without checking logs and GPU activity.

## Handle Reconstruction Failures Safely

- Preserve the failed output tree and logs.
- Diagnose the first failing stage and distinguish missing setup, source-data,
  GPU/container infrastructure, and implementation failures.
- Do not delete, overwrite, or silently reuse a partial output directory.
- Retry the complete sequential runner only with a new output path and after
  addressing the cause.
- Do not switch to a production workflow or another object-prompt path as a
  workaround.

## Verify Reconstruction Completion

Require the completion banner and verify at least:

```text
foundation_pose/poses.npy
sam3d_body/mhr_params_mv.pt
sam3d_body/mhr_mesh_mv.pt
sam3d_body/export_soma/soma_params.npz
postprocess/hoi_overlay/tiled_hoi_overlay.mp4
postprocess/wis3d/
```

Report the output path, the validated artifacts, and any non-fatal warnings.
Make clear that these checks do not replace production accuracy gates or human
QC.
