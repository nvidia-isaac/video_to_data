# MECKA Panda inpainting pipeline

This package turns MECKA camera-frame hand tracks into a rendered bimanual
Franka Panda replacement. Its on-disk seams match the contracts used by the
`mverghese/visual-inpainting-investigation` branch:

```text
MECKA parquet
  -> tracking/tracking.npz
  -> retarget/parallel_jaw_trajectory.npz
  -> robot_render/{robot_rgb.mp4,robot_mask.npy,robot_depth.npy}
  -> composite/final_overlay.mp4
  -> four_stage_compare.mp4
```

The human-removal video is an upstream input in this first version. Pass it
with `--background`; its frame count, size, and FPS are checked before
compositing. An optional pre-rendered arm-mask preview can be supplied with
`--mask-preview`. Without one, the second review panel is explicitly marked as
a source-video fallback.

## Retarget methods

- `--ik dls`: 6-DoF damped least-squares Panda IK with temporal posture and
  outward-elbow null-space terms.
- `--ik hybrid`: SSIK analytic Panda IK with per-target DLS fallback.

Both methods consume the same robot-neutral
`v2d.inpainting.parallel-jaw-target/v1` archive. That archive uses the current
MECKA contact policy: thumb tip against the mean of the four non-thumb
fingertips, with the hand-web-to-contact direction as approach.

Sharpa, Dex3, and G1 retargeting remain in `robotic_grounding`; they are not
silently dispatched through this Panda-specific pipeline.

## Run

Planning is read-only and is the default:

```bash
.venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  --dataset /path/to/mecka \
  --episode 51 \
  --output-dir /path/to/run \
  --background /path/to/hand_removed.mp4 \
  --rig-config debug/mecka_bimanual_rig.json
```

After reviewing the JSON plan, add `--execute`. Existing complete generations
are validated by configuration plus source/output hashes and reported as
`skipped_complete`. Use `--overwrite` only to deliberately replace selected
stages. Repeat `--stage` to run a subset:

```bash
MUJOCO_GL=egl .venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  ... --stage tracking --stage retarget --stage render \
  --ik hybrid --execute
```

The Panda MJCF directory defaults to the existing local
`debug/third_party/mujoco_menagerie/franka_emika_panda` checkout. A portable
run should pass `--panda-dir` explicitly; the directory must contain
`panda.xml` and its `assets/` tree.

## Validation

```bash
.venv/bin/ruff check inpainting
.venv/bin/python -m pytest -q inpainting/tests
```

