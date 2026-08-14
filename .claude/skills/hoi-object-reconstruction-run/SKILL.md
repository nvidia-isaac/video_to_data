---
name: hoi-object-reconstruction-run
description: Launch, monitor, resume, and verify this repository's BundleSDF or SAM3D HOI object reconstruction pipeline. Use when a user asks to run object reconstruction, process calibrated stereo mapping data, try the included basketball example, produce a BundleSDF GLB, produce a SAM3D candidate mesh, resume a partial reconstruction job, or verify the final mesh and alignment artifacts.
---

# Run HOI Object Reconstruction

Run the top-level host orchestrator from `reconstruction/`. Unless the user
explicitly requests a command only, launch the pipeline and verify that a real
stage starts; do not stop after printing a command.

## Resolve defaults without an interview

Use explicit user values first. Fill missing values as follows:

| Value | Default |
|---|---|
| mode | `bundlesdf` |
| input | `modules/v2d_hoi_object_reconstruction/assets/basketball_example/` |
| prompt | `basketball` for the included example; otherwise infer a concise object noun from the input folder name |
| GPU | `0` |
| output | a fresh directory under `data/outputs/hoi_recon/` using the input stem, mode, and current timestamp |
| SAM3D depth scale | off unless explicitly requested |

Never ask about optional tuning flags before the default run. Ask at most one
blocking question, and only when the user supplied no usable input and the
included example is unavailable, or when no object noun can be inferred after
inspecting the input. For an ambiguous custom prompt, explain the proposed noun
and proceed when it is a reasonable interpretation.

Create a fresh output path by default. Treat an existing user-specified job
directory as a resume after inspecting its artifacts; never delete it.

## Preflight and continue

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/reconstruction"
python ../.claude/skills/hoi-object-reconstruction-setup/scripts/preflight_input.py \
  <absolute-mapping-data-dir>
python -c 'from v2d_hoi_object_reconstruction.docker.run_reconstruction import main'
```

Check the selected-mode images and weights described by the
`hoi-object-reconstruction-setup` skill. If a prerequisite is missing, use that
skill to repair it and then return to this run automatically. Do not ask the
user to choose between setup and run.

## Launch BundleSDF

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
  --mapping_data_dir <absolute-input> \
  --job_dir <absolute-output> \
  --prompt "<object noun>" \
  --gpu_ids 0
```

Keep the built-in scan-quality and stage-boundary checks enabled. The included
basketball input and prompt form the default no-question smoke run.

## Launch SAM3D

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
  --mapping_data_dir <absolute-input> \
  --job_dir <absolute-output> \
  --prompt "<object noun>" \
  --mode sam3d \
  --gpu_ids 0
```

Add `--sam3d_use_depth` only when requested. Do not block a normal SAM3D run by
asking whether depth assistance is desired.

## Prove that the run started

Record the revision, exact command, input, output, mode, GPU, and start time.
Then monitor until there is concrete startup evidence:

- the wrapper process is still alive after initialization;
- a stage container started or the log names the active stage; and
- the job directory contains a stage artifact or log.

For a long run, keep monitoring and provide brief stage updates. A quiet log is
not by itself a failure; check the process, container, GPU, disk, and artifacts.
Preserve partial output on failure and hand diagnosis to
the `hoi-object-reconstruction-doctor` skill.

## Resume deliberately

Inspect the first incomplete stage and skip only stages whose required outputs
are complete and readable. SAM3D reuses candidates when both
`srt_result.json` and `output_scaled.glb` are valid. Use `--sam3d_force_srt`
only when the user explicitly requests recomputation.

## Verify completion

BundleSDF requires:

- nonempty `<job_dir>/merged_recon/output.glb`;
- `<job_dir>/fp_render_final/render.mp4` with sustained alignment; and
- visual inspection for missing major surfaces, duplicated shells, fused
  background or hand geometry, and floating fragments.

SAM3D requires:

- valid `<job_dir>/sam3d/best/best_frame.json`;
- nonempty `<job_dir>/sam3d/best/output_scaled.glb`; and
- the chosen `render_debug.jpg` and `render_video.mp4` to show plausible
  silhouette alignment, orientation, and scale.

Report the exact artifact paths and limitations. A zero exit code or generated
GLB alone is not a completed visual-QA result. Keep mesh-to-USD conversion and
drop testing as a separate downstream workflow.
