---
name: hoi-object-reconstruction-setup
description: Prepare this repository's HOI object reconstruction environment for BundleSDF or SAM3D. Use when a user asks to install host orchestration packages, validate Docker/GPU access, build missing reconstruction images, download mode-specific weights, preflight calibrated stereo input, or make the checkout ready before an HOI reconstruction run.
---

# HOI Object Reconstruction Setup

Prepare the selected reconstruction mode and prove that it is runnable. Work
from the repository checkout; run host orchestration from `reconstruction/`.

## Act before asking

- Probe the checkout, Python, Docker, GPU, images, weights, and input immediately.
- Default to `bundlesdf` when the user has not selected a mode.
- Default to GPU 0 when the user has not constrained GPU use.
- Use
  `modules/v2d_hoi_object_reconstruction/assets/basketball_example/` for input
  preflight when no dataset was supplied.
- Install, build, or download only what the selected mode is missing. Unless the
  user requested commands or a plan only, perform those steps instead of merely
  describing them.
- Do not ask for an object prompt or output directory during setup; those belong
  to the `hoi-object-reconstruction-run` skill.
- Ask only when credentials, gated-model approval, or another genuinely
  user-only action blocks progress. Never request a token in chat.

## Establish the checkout

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/reconstruction"
git status --short
python --version
docker version
nvidia-smi
df -h .
```

Treat `modules/v2d_hoi_object_reconstruction/README.md` and this command as the
current interface:

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py --help
```

Preserve unrelated local changes. Keep heavy numerical dependencies in the
containers; install only the lightweight host wrappers on the host.

## Validate an input without interviewing

Resolve `mapping_data_dir` from the request. If it is absent, use the included
basketball example. Then run:

```bash
python ../.claude/skills/hoi-object-reconstruction-setup/scripts/preflight_input.py \
  <absolute-mapping-data-dir>
```

Pass means `frames_meta.json` is valid, both stereo cameras are calibrated, and
at least one synchronized JPEG pair exists. This static check cannot prove that
a BundleSDF capture contains the required two scan stages; leave the pipeline's
CuSFM scan-quality gate enabled.

## Install the host package when needed

Test the import first:

```bash
python -c 'from v2d_hoi_object_reconstruction.docker.run_reconstruction import main'
```

If it fails, install repository host packages from `reconstruction/`:

```bash
./scripts/install_packages.sh
```

Re-run the import after installation.

## Build only missing images

Both modes require these images:

```text
v2d_hoi_object_reconstruction  v2d_cusfm  v2d_grounding_dino  v2d_sam2
```

BundleSDF additionally requires:

```text
v2d_foundation_stereo  v2d_bundlesdf  v2d_foundation_pose
```

SAM3D requires `v2d_sam3d`; depth-assisted SAM3D also requires
`v2d_foundation_stereo`.

Check with `docker image inspect <image>`. Build a missing image with its
existing entrypoint:

```bash
# Shared
python modules/v2d_hoi_object_reconstruction/docker/build.py
python modules/v2d_cusfm/docker/build.py
python -m v2d.grounding_dino.docker.build
python -m v2d.sam2.docker.build

# BundleSDF
python -m v2d.foundation_stereo.docker.build
python modules/v2d_bundlesdf/docker/build.py
python -m v2d.foundation_pose.docker.build

# SAM3D
python modules/v2d_sam3d/docker/build.py
```

Build only the shared entries and selected mode. Add FoundationStereo to SAM3D
only for depth assistance. Use `./scripts/build_containers.sh` only when the
user explicitly wants all reconstruction modules prepared.

## Download only missing weights

Shared:

```bash
python -m v2d.sam2.docker.run_download_weights --output_dir data/weights/sam2
python -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
```

BundleSDF:

```bash
python modules/v2d_foundation_stereo/docker/run_download_weights.py --output_dir data/weights/foundationstereo
python modules/v2d_foundation_pose/docker/run_download_weights.py --output_dir data/weights/foundationpose
python modules/v2d_bundlesdf/docker/run_download_weights.py --output_dir data/weights
```

SAM3D:

```bash
python modules/v2d_sam3d/docker/run_download_weights.py --output_dir data/weights/sam3d
```

SAM3D weights require authorized access to `facebook/sam-3d-objects`. If access
is missing, report that single blocker and finish all non-gated setup first.

## Prove readiness

Before declaring setup complete, show evidence for all applicable gates:

1. Host wrapper import succeeds.
2. Input preflight passes.
3. Selected-mode images exist.
4. Selected-mode weight directories are populated.
5. The requested GPU is visible inside an already-built image:

   ```bash
   docker run --rm --gpus '"device=0"' v2d_hoi_object_reconstruction nvidia-smi
   ```

6. For SAM3D, the EGL renderer initializes in `v2d_sam3d`:

   ```bash
   docker run --rm --gpus '"device=0"' v2d_sam3d \
     python -c 'import pyrender; r=pyrender.OffscreenRenderer(64,64); print("egl_renderer=pass"); r.delete()'
   ```

Report PASS, FAIL, or BLOCKED for each gate, then continue with
the `hoi-object-reconstruction-run` skill when the user's request also includes
a run.
Do not stop after setup simply because reconstruction is long.
