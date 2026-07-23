# Parallel-jaw and GraspGenX reproduction

The generated robot bundles, model checkpoints, object reconstructions, and
videos live under `inpainting/artifacts/` and are intentionally ignored by git.
This document rebuilds the source-derived seams needed by the Galbot Golf, YAM,
and GraspGenX experiments.

Run commands from the `video_to_data_internal` repository root. Paths below
named `/absolute/path/...` must be replaced with local storage paths.

## Source revisions

| Source | Revision | Role |
|---|---|---|
| `GalaxyGeneralRobotics/galbot_one_golf_description` | `b311f5ca1acf506e9b7026397e2c74fb2db11df6` | Galbot URDF and meshes |
| RoboLab MR 62 | `8224d5fb8a2a3d21ce445bb198476c1faa4d69e6` | Executed Galbot posture and gripper-linkage build input |
| `ARISE-Initiative/yamlab` | `ec0455d2b4ce35f21fc126418ea5e74ac567133d` | YAM USD, robot config, and license |
| RoboLab MR 68 | `543a08bf3b46aa8fb2abc79ffba09cf4d09e09ae` | YAM usage audit only; it is not a bundle-build input |
| `NVlabs/GraspGenX` | `b9429097728cb1c430dd78b92edf17ba318aad03` | Sweep-volume-conditioned grasp inference |

Clone each source into a directory outside Video2Data and detach it at the
declared revision. For example:

```bash
external_root=/absolute/path/to/visual_inpainting_external
mkdir -p "$external_root"

git clone https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description.git \
  "$external_root/galbot_one_golf_description"
git -C "$external_root/galbot_one_golf_description" checkout --detach \
  b311f5ca1acf506e9b7026397e2c74fb2db11df6

git clone https://gitlab-master.nvidia.com/xuningy/robolab.git \
  "$external_root/robolab_mr62"
git -C "$external_root/robolab_mr62" checkout --detach \
  8224d5fb8a2a3d21ce445bb198476c1faa4d69e6

git clone https://github.com/ARISE-Initiative/yamlab.git \
  "$external_root/yamlab"
git -C "$external_root/yamlab" checkout --detach \
  ec0455d2b4ce35f21fc126418ea5e74ac567133d
```

The RoboLab clone requires normal NVIDIA GitLab access. Do not place a token in
a command, config, or committed URL.

## Build the ignored robot bundles

Galbot is derived directly on the host. The builder rejects a dirty or
incorrectly pinned input and fingerprints the public assets, MR 62 definitions,
converter, and outputs:

```bash
python3 -m inpainting.parallel_jaw_galbot_assets \
  --galbot-root "$external_root/galbot_one_golf_description" \
  --robolab-root "$external_root/robolab_mr62" \
  --output-dir inpainting/artifacts/parallel_jaw_assets/galbot_one_golf
```

YAM conversion requires USD, Pinocchio, pyrender, trimesh, and yourdfpy. The
study used the already-pinned `robotic-grounding:photo-render-v6` environment:

```bash
mkdir -p inpainting/artifacts/parallel_jaw_assets/yam_bimanual

docker run --rm --gpus device=0 \
  --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/repo \
  -e PYOPENGL_PLATFORM=egl \
  -v "$PWD:/repo:ro" \
  -v "$external_root/yamlab:/assets/yamlab:ro" \
  -v "$PWD/inpainting/artifacts/parallel_jaw_assets/yam_bimanual:/output:rw" \
  -w /repo \
  robotic-grounding:photo-render-v6 \
  python3 -m inpainting.robot_assets.yam_usd_to_urdf \
    --source-usd /assets/yamlab/yamlab/robot/yam/arm/yam.usd \
    --input-repository /assets/yamlab \
    --output-dir /output \
    --expected-commit ec0455d2b4ce35f21fc126418ea5e74ac567133d \
    --expected-usd-sha256 \
      8b997d9c864a53a00abba54d2381d3aebf71853f7918450305afc11e73aeb499 \
    --expected-config-sha256 \
      edb89ddf2b418a48a0b46632aafea6a082a306048df4f7c2f2a2301cced17dd1
```

The expected bundle manifests are:

```text
inpainting/artifacts/parallel_jaw_assets/galbot_one_golf/bundle_manifest.json
inpainting/artifacts/parallel_jaw_assets/yam_bimanual/bundle_manifest.json
```

Use `inpainting.adapters.parallel_jaw_from_tracking` to produce the common
baseline target, then `inpainting.run_parallel_jaw_comparison` to plan or run
the original GT/V2D/Phantom five-panel study. Their full CLI contracts are
available through `--help`.

## Install the pinned GraspGenX runtime

```bash
git clone https://github.com/NVlabs/GraspGenX.git \
  "$external_root/GraspGenX"
git -C "$external_root/GraspGenX" checkout --detach \
  b9429097728cb1c430dd78b92edf17ba318aad03

cd "$external_root/GraspGenX"
uv sync
uv run python -c \
  "from graspgenx import get_checkpoints_version_dir; print(get_checkpoints_version_dir())"
cd -
```

The last import performs GraspGenX's official first-use checkpoint acquisition.
The production checkpoint argument is normally:

```text
<GraspGenX>/ext/graspgenx_checkpoints/release
```

Downloaded checkpoints remain outside Video2Data. Candidate JSON sidecars
record the exact source, config, generator, and discriminator hashes.

## Generate candidates

This is the exact Galbot sweep-volume configuration and production sampling
count. Replace the metric mesh and output paths for each object:

```bash
PYTHONPATH="$PWD" "$external_root/GraspGenX/.venv/bin/python" \
  -m inpainting.graspgenx_candidates \
  --mesh /absolute/path/to/object/metric_mesh.glb \
  --output /absolute/path/to/candidates/object.npz \
  --graspgenx-root "$external_root/GraspGenX" \
  --checkpoint-root \
    "$external_root/GraspGenX/ext/graspgenx_checkpoints/release" \
  --gripper-name galbot_one_golf \
  --gripper-type revolute_2f \
  --extents-open 0.12490876627340242 0.0203 0.05952241 \
  --offset-open 0 0 0.12910008 \
  --extents-mid 0.07423274 0.0203 0.0595224 \
  --offset-mid 0 0 0.15629605 \
  --fingertip-depth 0.13996 \
  --seed 254 \
  --num-grasps 600 \
  --top-k 150 \
  --num-sample-points 3500
```

For YAM, replace the profile arguments with:

```text
--gripper-name yam_bimanual
--gripper-type parallel_2f
--extents-open 0.09490105891248288 0.068 0.068
--offset-open 0 0 0.10856
--extents-mid 0.04742217 0.068 0.068
--offset-mid 0 0 0.10856
--fingertip-depth 0.14256
```

Production used deterministic per-clip seeds recorded in each candidate
sidecar. Reusing a sidecar's seed and exact mesh reproduces its sampled point
cloud and candidate ordering.

## Select and propagate one interaction

The following is the complete production policy for the `253` left-cup event.
It leaves contact-registration magnitude unbounded and unpenalized; the
resulting `172.6--305.5` mm registrations are a reported limitation, not a
claim of globally metric alignment.

```bash
PYTHONPATH="$PWD" "$external_root/GraspGenX/.venv/bin/python" \
  -m inpainting.run_graspgenx_refinement \
  --base-target /absolute/path/to/parallel_jaw/targets/v2d/parallel_jaw_trajectory.npz \
  --tracking /absolute/path/to/v2d/tracking/tracking.npz \
  --T-camera-world /absolute/path/to/egocentric_frame_extrinsic.npy \
  --foundationpose-poses /absolute/path/to/cup/poses_smoothed \
  --mesh /absolute/path/to/cup/metric_mesh.glb \
  --candidates /absolute/path/to/candidates/cup.npz \
  --robot-profile galbot_one_golf \
  --side left \
  --object-name cup \
  --event-start 8 \
  --event-anchor 14 \
  --event-end 58 \
  --no-starts-in-contact \
  --output-target /absolute/path/to/output/parallel_jaw_trajectory.left_cup.npz \
  --propagation-mode base_local_offset \
  --approach-blend-frames 8 \
  --release-blend-frames 8 \
  --min-antipodal-score 0.65 \
  --score-contact-weight 1.0 \
  --score-confidence-weight 0.005 \
  --score-pose-translation-weight 0.1 \
  --score-pose-rotation-weight 0.04 \
  --score-approach-weight 0.015 \
  --score-registration-weight 0
```

Run the second interaction with the first interaction's output as
`--base-target`, then write the final combined file as
`parallel_jaw_trajectory.npz`. The checked-in
`configs/graspgenx_refinement_events.json` contains the inclusive
start/anchor/end windows, start-in-contact flags, and RGB/Phantom evidence for
all four interactions.

The final refined render gates were:

| Clip | Robot | Orientation gate | Joint-step gate |
|---|---|---:|---:|
| `105` | Galbot Golf | 65 deg | 0.58 rad/frame |
| `105` | YAM | 25 deg | 0.45 rad/frame |
| `253` | Galbot Golf | 22 deg | 0.50 rad/frame |
| `253` | YAM | 26 deg | 0.45 rad/frame |

All conditions retain the 10 mm position gate. Exact selected candidates,
scores, symmetry choices, registrations, target hashes, and renderer policies
are recorded in the ignored output sidecars.
