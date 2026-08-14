---
name: hoi-object-reconstruction-doctor
description: Diagnose and repair failures in this repository's BundleSDF or SAM3D HOI object reconstruction workflow. Use when setup checks fail, Docker or GPU access is broken, weights or images are missing, input preflight fails, CuSFM or scan-quality gates fail, a BundleSDF stage produces bad geometry, FoundationPose alignment fails, SAM3D inference or EGL rendering fails, a job stalls, or final GLB and review artifacts are missing or invalid.
---

# HOI Object Reconstruction Doctor

Inspect evidence first and ask later. Work from `reconstruction/`, preserve the
job directory, and diagnose the first bad stage instead of rebuilding
everything.

## Discover context automatically

1. Read the exact command and error from the conversation or available log.
2. If no job directory is named, inspect the newest directory under
   `data/outputs/hoi_recon/` and other output paths referenced by recent
   commands.
3. Infer the mode from the command and artifacts:
   `merged_recon/` means BundleSDF; `sam3d/` means SAM3D.
4. Infer the input from the recorded command or job metadata.
5. Ask one focused question only if neither logs nor a job directory can be
   discovered.

## Run the fast diagnostic set

```bash
git rev-parse --short HEAD
python --version
docker version
nvidia-smi
df -h .
docker ps -a --no-trunc
python -c 'from v2d_hoi_object_reconstruction.docker.run_reconstruction import main'
```

When the input is known:

```bash
python ../.claude/skills/hoi-object-reconstruction-setup/scripts/preflight_input.py \
  <absolute-mapping-data-dir>
```

Inspect the job tree, nonempty files, timestamps, and the last relevant log.
Separate observed evidence, root cause, repair, and resume command.

## Classify the first failure

| Evidence | Classification | Narrow repair |
|---|---|---|
| Docker unavailable or GPU invisible in a container | setup/runtime | Repair Docker or NVIDIA Container Toolkit; re-run the GPU probe |
| missing image or weight directory | setup | Use `hoi-object-reconstruction-setup` for the selected mode only |
| preflight errors | input contract | Fix the named metadata, calibration, synchronization, or file path issue |
| CuSFM scan-quality gate rejects BundleSDF input | capture/input | Do not bypass it; inspect whether the stationary-rotate-stationary scan pattern exists |
| `no kernel image is available` | image/GPU compatibility | Rebuild for that architecture or use a validated GPU; do not blame the dataset |
| `libEGL.so.1` or offscreen renderer failure | SAM3D image | Rebuild `v2d_sam3d`, run its EGL probe, then resume rendering/selection |
| Stage-1 BundleSDF mesh is already bad | masks/depth/CuSFM/Stage-1 | Inspect masks, trajectory, depth range, and Stage-1 extraction first |
| Stage-1 is good but merged mesh is bad | FoundationPose/alignment | Inspect tracking overlays and world poses before rerunning merged reconstruction |
| SAM3D candidate is bad | frame/mask/inference | Inspect source frame and mask, then SRT result and render overlay |
| wrapper exited but final review files are absent | incomplete pipeline | Resume from the first missing artifact; do not mark complete |

## Detect a stall correctly

Check the wrapper PID, active containers, `nvidia-smi`, disk growth, and artifact
timestamps. A quiet log alone is not a stall. Report which signal stopped
changing and for how long.

## Repair and re-verify

Prefer the smallest reversible repair. Do not delete partial outputs, weaken
quality gates, reset unrelated state, or force all stages to rerun. After a
repair:

1. Re-run the failed diagnostic probe.
2. Resume with the original input, prompt, mode, GPU, and job directory.
3. Verify the first formerly failing stage produces its complete artifact set.
4. Return to the `hoi-object-reconstruction-run` skill for completion and
   visual QA.

If the evidence shows a product defect rather than an environment or data
problem, preserve the reproducer and report the source revision, exact command,
first bad artifact, logs, hardware, and expected versus observed behavior.
