# MECKA Panda inpainting pipeline

This pipeline turns one MECKA episode into a hand-removed, Panda-rendered
video. Human removal and robot generation are independent branches that meet
only at compositing:

```text
                                  +-> arm mask -> ProPainter --+
MECKA episode -> tracking --------+                            +-> composite -> review
                                  +-> retarget -> Panda render +
```

The formal stage DAG is:

| Stage | Dependencies | Published artifacts |
|---|---|---|
| `tracking` | none | `tracking/tracking.npz`, `intrinsic.npy`, `camera_to_world_xyzw.npy`, `tracking.json`; S3/LeRobot runs also materialize the complete episode as `tracking/source_video.mp4` |
| `mask` | `tracking` | `arm_mask/arm_mask.npy`, `mask_preview.mp4`, `arm_mask.json` |
| `inpaint` | `mask` | `inpaint/hand_removed.mp4`, `hand_removed.json` |
| `retarget` | `tracking` | `retarget/parallel_jaw_trajectory.npz`, `parallel_jaw_trajectory.json` |
| `render` | `retarget` | `robot_render/robot_rgb.mp4`, `robot_mask.npy`, optional `robot_depth.npy`, and `render_metadata.json` |
| `composite` | `render` plus `inpaint` unless an external background is supplied | `composite/final_overlay.mp4`, `final_overlay.json` |
| `review` | `composite` plus the built-in mask/inpaint artifacts that were not overridden | `four_stage_compare.mp4`, `four_stage_compare.json` |

The schemas and I/O helpers live in `inpainting/mecka_panda/` instead of the
shared `inpainting/contracts.py` and `inpainting/video_io.py`. This pipeline
predates the parallel-jaw and robot-render contracts used by the rest of this
directory and still carries its own tracking and render schemas, so keeping it
self-contained lets both stacks change without breaking each other.

Two known duplications are not resolved yet: `inpainting/panda_renderer/`
overlaps with `inpainting/parallel_jaw_renderer/`, and
`inpainting/mecka_panda/contracts.py` re-implements validation that
`inpainting/adapters/parallel_jaw_from_tracking.py` also provides.

## Automatic hand and arm removal

### Mask stage

`inpainting/mecka_panda/arm_mask.py` is the production form of the automatic
masking logic previously exercised by `debug/mecka_arm_pipeline.py`. It does
not reread a local parquet file. Its inputs are the formal
`tracking/tracking.npz`, `tracking/intrinsic.npy`, `tracking/tracking.json`, and
the episode video, so local-manifest and S3/LeRobot tracking feed exactly the
same implementation.

For the selected frame window, the stage:

1. conditions the camera-frame 21-joint tracks;
2. projects them with the scaled camera matrix and the original
   `k1,k2,p1,p2` lens-distortion coefficients recorded by the tracking stage;
3. makes a temporary exact-window clip at width 1280 by default, preserving
   aspect ratio and an even height;
4. selects spaced, sharp frames with both hands visible, runs Grounding-DINO
   to obtain arm boxes, assigns them to the projected left/right hand, and
   sends the resulting prompts to SAM2;
5. checks hand/palm support, mask area, evidence that the mask reaches the arm
   root, temporal area and centroid jumps, left/right overlap, valid fraction,
   and consecutive failure length;
6. adds correction prompts and retries when the quality gate fails.

The quality gate is mandatory. If all attempts fail, the stage raises
`ArmMaskQualityError` and does not publish a complete formal mask for
ProPainter to consume.

The temporary detector/SAM2 geometry does not leak into the contract.
`arm_mask.npy` is an exact boolean array of shape `(N,H,W)` at the original
source resolution; `True` means remove the pixel. The union mask is dilated,
resized with nearest-neighbor sampling, and published together with a
full-resolution `mask_preview.mp4`. `arm_mask.json` records the input/model
fingerprints, configuration, frame window, working and source geometry, and
quality diagnostics.

Use `--arm-mask-config path.json` to replace `ArmMaskConfig` defaults. The file
must contain one JSON object whose keys match the dataclass fields. The default
runtime is `reconstruction/.venv/bin/python` with the existing Grounding-DINO
and SAM2 Docker runners and weights below `reconstruction/`. Cache provenance
includes the runner implementations, concrete Docker image identities, and
model weights.

### ProPainter stage

`inpainting/mecka_panda/propainter.py` consumes only the formal full-resolution
boolean mask and the same source-video window. It expands the NPY into a
temporary PNG mask directory for ProPainter; PNGs are an adapter detail and
are not a second pipeline contract.

The default backend options match the proven debug run:

```text
resize_ratio=0.5, subvideo_length=40, neighbor_length=6,
ref_stride=10, fp16=true
```

ProPainter runs at `resize_ratio=0.5` by default and saves its candidate
frames. The adapter validates their count, geometry, decoding, and FPS,
upsamples the candidate, and copies it only inside the original
full-resolution boolean mask. Pixels outside the mask come from the exact
full-resolution source window. The published `inpaint/hand_removed.mp4`
therefore has the source `N,H,W,fps`; the backend's half-resolution or
macroblock-padded video is never used as the compositing background.

The default checkout is `debug/third_party/ProPainter`, and the default Python
is `~/miniconda3/envs/vlmevalkit/bin/python`. Preflight fingerprints the
ProPainter source tree (including `inference_propainter.py`), that Python
executable, and `ProPainter.pth`, `raft-things.pth`, and
`recurrent_flow_completion.pth`, so a backend-code or weight change invalidates
the inpaint cache.

E2FGVI is not part of this MECKA-to-Panda execution path. There is no silent
fallback to E2FGVI and no E2FGVI runtime is needed for these stages.

## Frame-window semantics

`--start-frame` and `--max-frames` describe a window in the complete episode:

```text
episode frame [start_frame, start_frame + count)
```

Tracking arrays contain only that window. For a LeRobot shard, the tracking
stage first extracts the complete episode from the shared MP4 and verifies its
frame count; mask and inpaint then sequentially decode and select the exact
window beginning at `--start-frame` within that episode clip. This prevents an
S3 video timestamp cut from being confused with a frame-zero window and avoids
backend-dependent random-seek behavior.

Formal mask, preview, inpaint, render, composite, and review artifacts all
start at their own frame zero and contain exactly `count` frames. Consequently
the built-in mask preview and built-in ProPainter background use offset zero.
Only external overrides use `--mask-start-frame` or
`--background-start-frame`.

## Data source

Episodes come from a LeRobot v3 shard, either a local directory or an
S3-compatible bucket. `inpainting/adapters/mecka_lerobot.py` reads it without
any LeRobot dependency, following the path templates in `meta/info.json`.
Episode indices restart at zero in every shard, so the dataset must resolve to
one exact shard before an episode index is meaningful.

An exact shard URI needs no selector:

```bash
.venv/bin/python -m inpainting.adapters.mecka_lerobot \
  --dataset s3://nv-00-10206-robot/cosmos3_action_data/mecka/20260509_46000h_everyday_mono_no_wrist_lerobot/v0/shard_00 \
  --episode 199 \
  --output-dir /path/to/run/tracking
```

The orchestrator can instead receive the parent prefix and an explicit shard.
An integer selector is formatted as `shard_XX`; a directory name such as
`shard_00` is also accepted:

```bash
.venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  --dataset s3://nv-00-10206-robot/cosmos3_action_data/mecka/20260509_46000h_everyday_mono_no_wrist_lerobot/v0 \
  --shard 0 \
  --episode 199 \
  --output-dir /path/to/run \
  --rig-config debug/mecka_bimanual_rig.json
```

Do not pass `--shard` when `--dataset` already points at a shard root. If a
parent prefix is given without `--shard`, the current default is
`shard_00`.

Remote reads require credentials. Put a JSON file in `credentials/` (see
[`credentials/README.md`](../credentials/README.md) for the required fields)
and pass `--credentials` if it is not the default
`credentials/gcp_training.secret`. Nothing under `credentials/` except its
README is tracked by git.

Only the bytes one episode needs are transferred, which matters because a shard
packs hundreds of episodes into one parquet file and one large MP4:

- Frames are read from just the parquet row groups the episode spans, since the
  shards are written with one row group per episode.
- The video is cut with `ffmpeg` seeking a presigned URL over HTTP range
  requests, using the episode's `from_timestamp`/`to_timestamp`. The extracted
  complete episode's frame count is checked against the declared episode
  `length`, so a mis-seek fails loudly instead of silently shifting tracking.

Camera intrinsics come from the episode metadata (`camera_intrinsics`, eight
values calibrated at 1920x1080) and are rescaled to the decoded resolution.
The four distortion values remain scale-independent and are retained in
`tracking.json` for the mask projection.

GCS does not return the response checksum headers that recent AWS SDKs validate
by default, so the reader sets
`AWS_RESPONSE_CHECKSUM_VALIDATION=when_required`; without it every read fails
with a checksum mismatch.

`inpainting/adapters/mecka.py` still reads the older local MECKA export, which
locates episodes through a `manifest.jsonl`. Both adapters emit the same
tracking artifacts needed downstream.

## Retarget methods

- `--ik dls` (default): standalone 6-DoF damped-least-squares Panda IK with
  temporal posture and outward-elbow null-space terms. SSIK is not required.
- `--ik hybrid`: optionally proposes an SSIK analytic solution first. The
  proposal is accepted only if it passes the same joint-limit and
  `--max-joint-step-rad` gate as DLS; unavailable, unsolved, or discontinuous
  SSIK proposals fall back to bounded DLS.

Both methods consume the same robot-neutral
`v2d.inpainting.parallel-jaw-target/v1` archive. The MECKA target policy is:

1. Position is the midpoint of thumb tip p4 and index tip p8.
2. Aperture is `distance(p4, p8)`.
3. Orientation defaults to a handedness-normalized palm frame when the p4/p8
   distance is small relative to palm width. This keeps a closed pinch from
   deriving direction from an almost-zero, noisy fingertip vector.
4. Between `--palm-ratio-max` and `--tip-ratio-min`, the palm and fingertip
   frames are blended on SO(3); above it, the fingertip frame dominates.
5. The nearest of the two equivalent parallel-jaw frames
   `R`/`R @ diag(1,-1,-1)` is selected before temporal filtering.
6. Accepted target rotations are low-pass filtered and hard-limited by
   `--max-rotation-step-deg` (6 degrees by default).

Quaternion signs are also made continuous, but that is only serialization:
the SO(3) and parallel-jaw gates above are what prevent actual frame jumps.
Missing tracking frames remain invalid/NaN and do not clear the last accepted
target or robot joint state, so reappearance cannot bypass either continuity
limit. The output NPZ keeps the existing exact-key v1 contract; phase, ratio,
fallback, symmetry, and maximum-step diagnostics live in the
`v2d.inpainting.mecka-parallel-jaw-run/v2` JSON metadata.

Sharpa, Dex3, and G1 retargeting remain in `robotic_grounding`; they are not
silently dispatched through this Panda-specific pipeline.

## Run

Planning is read-only and remains the default. With no overrides, the default
plan includes all seven stages and uses
`inpaint/hand_removed.mp4` as the compositing background:

```bash
.venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  --dataset /path/to/mecka \
  --episode 51 \
  --output-dir /path/to/run \
  --rig-config debug/mecka_bimanual_rig.json
```

Review the JSON plan, then add `--execute`:

```bash
MUJOCO_GL=egl .venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  --dataset /path/to/mecka \
  --episode 51 \
  --output-dir /path/to/run \
  --rig-config debug/mecka_bimanual_rig.json \
  --execute
```

For an exact remote shard:

```bash
MUJOCO_GL=egl .venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  --dataset s3://nv-00-10206-robot/cosmos3_action_data/mecka/20260509_46000h_everyday_mono_no_wrist_lerobot/v0/shard_00 \
  --episode 199 \
  --output-dir /path/to/run \
  --rig-config debug/mecka_bimanual_rig.json \
  --execute
```

To replace built-in hand removal with an existing hand-removed video, pass
`--background`. This prunes `inpaint` from the composite dependency graph and,
unless otherwise selected, avoids running it. Use
`--background-start-frame` only when the external video contains frames before
the selected window:

```bash
.venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  ... \
  --background /path/to/hand_removed.mp4 \
  --background-start-frame 120 \
  --execute
```

Likewise, `--mask-preview` replaces only the review panel; it does not change
the mask consumed by ProPainter. Pair it with `--mask-start-frame` when needed.
An external background and an external mask preview are fingerprinted in the
downstream cache metadata.

Repeat `--stage` to request a subset. Required upstream stages must either be
selected or already exist as complete, current artifacts:

```bash
.venv/bin/python -m inpainting.run_mecka_panda_pipeline \
  ... \
  --stage tracking --stage mask --stage inpaint \
  --execute
```

Caching follows the DAG rather than the textual stage order. A changed mask
invalidates `inpaint`, `composite`, and `review`, but does not invalidate
`retarget` or `render`. A retarget change invalidates `render`, `composite`, and
`review`, but not mask/inpaint. Supplying an external background removes the
inpaint dependency from composite. Complete artifacts are validated against
schema, dataset/shard/episode, frame window, configuration, implementation and
model identities, and file size/SHA-256 records. A stale unselected ancestor
blocks the plan and tells you which `--stage` to include; existing stale outputs
require explicit `--overwrite`.

Planning a remote episode reads metadata only. `--stage review` without
`--stage tracking` is blocked until an earlier run has materialized
`tracking/source_video.mp4`, because for S3 that complete episode clip is a
tracking output rather than a local input.

The Panda MJCF directory defaults to the existing
`debug/third_party/mujoco_menagerie/franka_emika_panda` checkout. A portable
run should pass `--panda-dir` explicitly; the directory must contain
`panda.xml` and its `assets/` tree.

## Validation

The CPU-only unit suite does not require S3 credentials or GPUs:

```bash
.venv/bin/ruff check \
  inpainting/mecka_panda \
  inpainting/panda_renderer \
  inpainting/adapters/mecka.py \
  inpainting/adapters/mecka_lerobot.py \
  inpainting/adapters/mecka_parallel_jaw.py \
  inpainting/run_mecka_panda_pipeline.py

.venv/bin/python -m pytest -q \
  inpainting/tests/test_mecka_pipeline.py \
  inpainting/tests/test_mecka_lerobot.py \
  inpainting/tests/test_mecka_arm_mask.py \
  inpainting/tests/test_mecka_propainter.py
```

The mask and ProPainter tests exercise geometry, exact-window offsets, quality
failure, artifact publication, and full-resolution mask-only compositing with
small synthetic inputs. Running Grounding-DINO, SAM2, ProPainter, MuJoCo, or an
S3 episode remains an end-to-end validation step.
