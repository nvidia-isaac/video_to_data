---
name: robotic_grounding_doctor
description: Troubleshooting skill for robotic_grounding failures — Docker/container issues, GPU-not-visible, missing MANO or object assets, motion-file-not-found, Isaac Lab startup crashes, permission errors, and pipeline-stage failures. Use this skill whenever something in robotic_grounding is broken or erroring: "the dummy agent crashes", "missing asset", "MANO not found", "Isaac won't start", "container permission denied", "0 URDFs generated", "motion file not found", "CUDA/GPU not available in the container", "the pipeline failed at stage X", or when a robotic_grounding command from another skill errors out. For first-time setup use robotic_grounding_onboard; to generate a working command use robotic_grounding_run.
---

# robotic_grounding — Troubleshooting

Diagnose and fix common robotic_grounding failures. **Work from the symptom, not from a fixed
script:** get the exact error text or the failing command first, match it to a section below, apply
the narrowest fix, and re-verify. Don't blindly rebuild the image or reset state.

## Step 0: Gather context (fast)

Ask for / collect:
1. The **exact command** and where it ran (host vs container).
2. The **error text** (last ~20 lines — Isaac stack traces bury the real cause mid-stack).
3. Whether this ever worked before, and what changed.

Quick environment probes (host):
```bash
docker images | grep -E 'robotic-grounding|task_library_loader'   # images present?
docker ps --format '{{.Names}}' | grep robotic-grounding          # container running?
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # GPU in Docker?
```

## Section 1: Docker / container

| Symptom | Cause | Fix |
|---------|-------|-----|
| `permission denied` on `docker ...` | User not in the `docker` group | Docker post-install steps (README prereqs); re-login |
| Workflow image pull denied / unauthorized | Registry access is not configured | Build locally, or set `V2D_IMAGE_REGISTRY` to a registry you can access and run `docker login <registry>` |
| `no space left on device` during build | Docker disk full | `docker system df` then `docker system prune` (careful); free host disk |
| Container name not found on `run.sh exec` | Container isn't running | `./workflow/run.sh start latest 0` first; name is `robotic-grounding-<version>-gpu<gpu>` |
| "I have no name!" bash prompt inside container | Cosmetic UID-mapping quirk | Harmless; `run.sh start` writes a per-container passwd entry |
| Files owned by root on the host after a container run | Isaac image runs as root | On the host: `sudo chown -R $(whoami) .` |
| Edits not taking effect | Editing outside the mounted tree, or expecting a rebuild | The repo is volume-mounted at `/workspace/video_to_data/robotic_grounding`; Python edits apply on next run, no rebuild. Confirm you're editing the mounted path |

## Section 2: GPU / CUDA

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvidia-smi` fails inside a `--gpus all` container | NVIDIA Container Toolkit not configured | Install/configure it (README prereqs), restart Docker |
| `CUDA error` / `no CUDA-capable device` at Isaac start | GPU not passed to the container, or wrong GPU index | Start with the right index: `./workflow/run.sh start latest <gpu>`; check `CUDA_VISIBLE_DEVICES` |
| Visualization / rendering errors | Driver mismatch | README recommends Driver 580.126.09 / CUDA 13.0; check `nvidia-smi` driver version |
| OOM during training | `--num_envs` too high for the GPU | Lower `--num_envs`; for a smoke run use `--num_envs 1 --max_iterations 1` |

## Section 3: MANO & datasets

| Symptom | Cause | Fix |
|---------|-------|-----|
| `MANO ... not found` / load stage fails | MANO `.pkl` files missing or misplaced | Place `MANO_LEFT.pkl` / `MANO_RIGHT.pkl` under `<HMD>/mano/models/` and pass `--mano-dir <HMD>/mano` (docs/SETUP.md §5). MANO is read only at the load stage and never committed |
| `<ds>_loaded` missing when retarget starts | The load stage (IMAGE 1) didn't run/produce output | Run the pipeline from the host with `run_pipeline_docker.py` (it handles both images), or `run_load_local.sh` first; check the MANO path |
| Dataset dir empty / wrong layout | Data not laid out as the loader expects | Re-check the per-dataset `docs/<DATASET>_SETUP.md`; sequence ids/patterns are dataset-specific |
| Adding a brand-new dataset (not one of the seven supported) | It's not in the registry | That's a different flow — follow "Adding a New Dataset" in `robotic_grounding/workflow/data_pipeline.md`, not this skill |

## Section 4: Object assets & URDFs

| Symptom | Cause | Fix |
|---------|-------|-----|
| Missing-asset exception in dummy/train/eval | Object URDFs/meshes not generated yet | Add `--use_primitive_urdfs` for an asset-free run, OR generate assets: run the pipeline's `urdf` stage, or `python scripts/generate_rigid_urdfs.py --dataset <dataset>` |
| Object renders as a plain sphere | Mesh path points to a missing file | Object meshes weren't placed — see the per-dataset `*_SETUP.md` object-assets section |
| `0 URDFs generated` with no error (OSMO/local) | `HUMAN_MOTION_DATA_DIR` redefined locally in `generate_rigid_urdfs.py` | It must `from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR`; a local redefinition silently reverts to the in-image path (the workflow swallows it with `\|\| true`) |

## Section 5: Motion files & RL scripts

| Symptom | Cause | Fix |
|---------|-------|-----|
| `motion_file not found` | Shorthand doesn't resolve, or partition not visible to the container | Shorthand `<dataset>/<dataset>_processed/<sequence_id>/<robot>` resolves under `assets/human_motion_data/`; copy/symlink the partition there, or pass an absolute path |
| `motion_file not found` for data committed in the repo (e.g. `whole_body/`) | Container started with `HUMAN_MOTION_DATA_DIR` set, overlaying the committed data (pre-fix run.sh mounted the whole root); or the container mounts a different clone/branch than the one you pulled | `docker inspect <container>` and check the `/workspace/video_to_data` mount source is the tree you're testing; restart the container from the right checkout, without the overlay if it lacks the dataset |
| "is a git-LFS pointer" error, or parquet read crash on a fresh clone | Clone made without `git lfs install` (or `GIT_LFS_SKIP_SMUDGE=1`) left pointer stubs | `git lfs install && git lfs pull` in the checkout the container mounts |
| Task not registered / unknown task | Wrong task id | Floating-hand tasks: `Sharpa-V2D-v0` (train), `Sharpa-V2D-v0-Play` (eval/dummy). Whole-body is a different skill |
| `eval.py` can't find a checkpoint | No `--checkpoint` and no local run | Point `--checkpoint` at `logs/rsl_rl/<run>/model_*.pt`, or use `--use_pretrained_checkpoint` |
| Dummy agent loads but sim doesn't advance | Motion Parquet is empty/corrupt | Re-run the pipeline for that sequence; try a known-good example (`dataset_s01_box_grab_01`) |
| W&B errors during training | `--logger wandb` without W&B configured | Use `--logger tensorboard` for local runs |

## Section 6: Isaac Lab startup

Isaac Lab crashes print long stack traces — the real cause is usually **before** the final frame.

| Symptom | Cause | Fix |
|---------|-------|-----|
| Hangs at first launch | Isaac Sim is compiling shaders / downloading assets | Wait it out on first run; subsequent runs are faster |
| Segfault / GLFW / display error in GUI mode | No display available | Use `--headless` (add `--record_video --output_dir <dir>` to inspect output) |
| Fabric-related errors | Fabric I/O issue | Try `--disable_fabric` |

## When it's not in the tables

- Re-read the failing stage's doc: `workflow/data_pipeline.md` (pipeline stages),
  the per-dataset `docs/*_SETUP.md`, or `docs/ARCHITECTURE.md` (§11 conventions & gotchas).
- Reduce to the smallest reproduction: a `dummy_agent.py ... --use_primitive_urdfs` run on a known
  example sequence isolates asset/motion problems from RL problems.
- If a command is the issue rather than the environment, regenerate it with **`robotic_grounding_run`**.
