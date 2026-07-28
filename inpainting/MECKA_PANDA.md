# MECKA Panda inpainting pipeline

This pipeline turns MECKA camera-frame hand tracks into a rendered bimanual
Franka Panda replacement:

```text
MECKA parquet
  -> tracking/tracking.npz
  -> retarget/parallel_jaw_trajectory.npz
  -> robot_render/{robot_rgb.mp4,robot_mask.npy,robot_depth.npy}
  -> composite/final_overlay.mp4
  -> four_stage_compare.mp4
```

Its schemas and I/O helpers live in `inpainting/mecka_panda/` instead of the
shared `inpainting/contracts.py` and `inpainting/video_io.py`. This pipeline
predates the parallel-jaw and robot-render contracts used by the rest of this
directory and still carries its own tracking and render schemas, so keeping it
self-contained lets both stacks change without breaking each other.

Two known duplications are not resolved yet: `inpainting/panda_renderer/`
overlaps with `inpainting/parallel_jaw_renderer/`, and
`inpainting/mecka_panda/contracts.py` re-implements validation that
`inpainting/adapters/parallel_jaw_from_tracking.py` also provides.

The human-removal video is an upstream input in this version. Pass it with
`--background`; its frame count, size, and FPS are checked before compositing.
An optional pre-rendered arm-mask preview can be supplied with
`--mask-preview`. Without one, the second review panel is explicitly marked as
a source-video fallback.

## Data source

Episodes come from a LeRobot v3 shard, either a local directory or an
S3-compatible bucket. `inpainting/adapters/mecka_lerobot.py` reads it without
any LeRobot dependency, following the path templates in `meta/info.json`.

```bash
.venv/bin/python -m inpainting.adapters.mecka_lerobot \
  --dataset s3://nv-00-10206-robot/cosmos3_action_data/mecka/20260509_46000h_everyday_mono_no_wrist_lerobot/v0/shard_00 \
  --episode 199 \
  --output-dir /path/to/run/tracking
```

Remote reads require credentials. Put a JSON file in `credentials/` (see
[`credentials/README.md`](../credentials/README.md) for the required fields) and
pass `--credentials` if it is not the default `credentials/gcp_training.secret`.
Nothing under `credentials/` is tracked by git.

Only the bytes one episode needs are transferred, which matters because a shard
packs hundreds of episodes into one ~700 MB parquet file and one ~29 GB mp4:

- Frames are read from just the parquet row groups the episode spans, since the
  shards are written with one row group per episode.
- The video is cut with `ffmpeg` seeking a presigned URL over HTTP range
  requests, using the episode's `from_timestamp`/`to_timestamp`. The extracted
  clip's frame count is checked against the episode `length`, so a mis-seek
  fails loudly instead of silently shifting the tracking alignment.

Camera intrinsics come from the episode metadata (`camera_intrinsics`, eight
values calibrated at 1920x1080) and are rescaled to the decoded resolution.

GCS does not return the response checksum headers that recent AWS SDKs validate
by default, so the reader sets `AWS_RESPONSE_CHECKSUM_VALIDATION=when_required`;
without it every read fails with a checksum mismatch.

`inpainting/adapters/mecka.py` still reads the older local MECKA export, which
locates episodes through a `manifest.jsonl`. Both adapters emit the same
artifacts, so downstream stages are unaffected by which one produced them.

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

`--dataset` takes either layout, so a remote shard needs no other change. The
layout is detected by looking for `meta/info.json`, and `--credentials` applies
to `s3://` datasets:

```bash
.venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  --dataset s3://nv-00-10206-robot/cosmos3_action_data/mecka/20260509_46000h_everyday_mono_no_wrist_lerobot/v0/shard_00 \
  --episode 199 \
  --output-dir /path/to/run \
  --rig-config debug/mecka_bimanual_rig.json \
  --stage tracking --stage retarget --execute
```

Planning a remote episode reads only metadata, so it stays fast and does not
download the clip; geometry and frame count come from `info.json` and the
episode `length` rather than from decoding. One consequence is that
`--stage review` without `--stage tracking` is blocked in preflight unless an
earlier run already extracted the clip, because for a remote shard the source
video is a tracking output rather than an input.

The Panda MJCF directory defaults to the existing local
`debug/third_party/mujoco_menagerie/franka_emika_panda` checkout. A portable
run should pass `--panda-dir` explicitly; the directory must contain
`panda.xml` and its `assets/` tree.

## Validation

```bash
.venv/bin/ruff check inpainting/mecka_panda inpainting/panda_renderer \
  inpainting/adapters/mecka.py inpainting/adapters/mecka_lerobot.py
.venv/bin/python -m pytest -q \
  inpainting/tests/test_mecka_pipeline.py \
  inpainting/tests/test_mecka_lerobot.py
```

Both test modules are CPU-only and need no credentials; the LeRobot reader is
exercised against a synthetic local shard.
