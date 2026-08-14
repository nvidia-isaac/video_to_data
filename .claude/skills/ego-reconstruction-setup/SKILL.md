---
name: ego-reconstruction-setup
description: Prepare this repository’s egocentric reconstruction pipeline. Use when a user asks to install host packages, build Docker images, download model weights, verify Docker/GPU prerequisites, or diagnose missing setup for `reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py`.
---

# Ego Reconstruction Setup

Work from `reconstruction/`. Install the lightweight host wrappers, then build
only the pipeline images and download the matching weights:

```bash
bash scripts/install_ego_reconstruction_packages.sh
bash scripts/build_ego_reconstruction_packages.sh
bash scripts/download_ego_reconstruction_weights.sh --mode all
```

Select `hamer_prompt`, `hamer_mesh`, or `dynhamr_prompt` instead of `all` when
the run mode is known. `hamer_mesh` is for a caller-provided object mesh.

Before a long run, check the entrypoint and Docker/GPU availability:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py --help
docker version
nvidia-smi
```

Keep heavy ML dependencies in containers. For DynHaMR, verify the manual MANO
and BMC assets under `data/weights/hand/` before running.
