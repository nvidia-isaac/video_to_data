---
name: ego-reconstruction-setup
description: Prepare the repository-local egocentric reconstruction pipeline for Codex-driven work. Use when a user asks to finish setup, install host orchestration packages, build Docker images, download model weights, check GPU/Docker prerequisites, or debug missing setup for `reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py`, `run_ego_wilor.py`, or `run_v2d_ego_e2e.py`.
---

# Ego Reconstruction Setup

## Core Workflow

Work from the repo root unless a command explicitly says `cd reconstruction`.
Treat `reconstruction/` as the pipeline root.

Prefer the scoped ego setup scripts before the broad all-module scripts:

```bash
bash reconstruction/scripts/install_ego_reconstruction_packages.sh
bash reconstruction/scripts/build_ego_reconstruction_packages.sh
bash reconstruction/scripts/download_ego_reconstruction_weights.sh --mode all
```

Use the broad scripts only when the user needs every reconstruction module:

```bash
bash reconstruction/scripts/install_packages.sh
bash reconstruction/scripts/build_containers.sh
```

The setup scripts install lightweight host-side Docker wrappers and build ML
containers. Do not install heavy CUDA/PyTorch ML stacks on the host unless the
repo has changed and explicitly requires it.

## Weight Modes

Choose the download mode from the intended run path:

- `--mode hamer_prompt`: WiLoR/SAM2/HaMeR path with an object text prompt and SAM3D mesh generation.
- `--mode hamer_mesh`: WiLoR/SAM2/HaMeR path with `--object_mesh`; skips SAM3D weights but still needs object masks.
- `--mode dynhamr_prompt`: legacy ViPE + DynHaMR path with object text prompt.
- `--mode all`: safest when the run mode is unknown or multiple modes will be tested.

SAM3D can require `HF_TOKEN` or prior `huggingface-cli login` for gated model
access. DynHaMR/MANO setup is partly manual; verify this layout before running
`--hand_tracking dynhamr`:

```text
reconstruction/data/weights/hand/
- models/MANO_RIGHT.pkl
- BMC/*.npy
```

## Validation

After setup, run lightweight checks before a long pipeline run:

```bash
cd reconstruction
python modules/v2d_pipelines/run_ego_reconstruction.py --help
bash -n scripts/install_ego_reconstruction_packages.sh
bash -n scripts/build_ego_reconstruction_packages.sh
bash -n scripts/download_ego_reconstruction_weights.sh
```

If Docker/GPU access is in question, validate Docker and NVIDIA Container
Toolkit before building or running containers. In Codex, request approval before
long builds, downloads, or commands that require network/GPU access.

## Troubleshooting Priorities

1. Confirm the current working directory. The runners expect paths relative to
   `reconstruction/` unless absolute paths are used.
2. Confirm host packages are installed in the active Python environment.
3. Confirm the matching Docker images were built after package or Dockerfile changes.
4. Confirm weights exist under `reconstruction/data/weights/<module>`.
5. For gated or manual weights, surface the exact missing path and let the user
   provide credentials/assets.
6. Re-run the same setup script when interrupted; the scripts are intended to be
   repeatable, and Docker builds reuse cache.

See `references/ego-reconstruction-setup.md` for the repo map and setup command
selection details.
