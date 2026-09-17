---
name: mesh-to-usd-setup
description: Prepare this repository's mesh-to-USD generation, structural validation, and Isaac Sim drop-test environment. Use when a user asks to build the mesh-to-USD images, verify Docker/GPU access, prepare FoundationPose support calibration, check model weights, validate a standalone GLB or exported HOI sequence input, or make the checkout ready before generating a rigid USD.
---

# Mesh-to-USD Setup

Prepare the workflow and prove that the requested input mode is runnable. Work
from `reconstruction/`; keep Isaac Sim, OpenUSD, CoACD, and FoundationPose
dependencies inside their existing containers.

## Act before asking

- Inspect the checkout, Python, Docker, GPU, disk, images, and supplied input
  immediately.
- Default to GPU 0 when the user has not constrained GPU use.
- Use
  `modules/v2d_hoi_object_reconstruction/mesh_to_usd/tests/data/toy_airplane/einstar/output_aligned.glb`
  for a non-destructive input check when no asset was supplied.
- Build only missing images. Add FoundationPose only for cross-mesh recorded
  support, not for a standalone mesh or an exact recorded mesh with poses.
- Do not ask about mass, friction, output paths, or drop-test tuning during
  setup; those belong to `mesh-to-usd-run`.
- Never accept the NVIDIA Isaac Sim EULA on the user's behalf. Existing
  `ACCEPT_EULA=Y` or an explicit user instruction is sufficient for a later
  generation or drop-test run.
- Ask only when credentials, EULA acceptance, missing external data, or another
  genuinely user-only action blocks progress.

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

Treat
`modules/v2d_hoi_object_reconstruction/mesh_to_usd/README.md` and the current
CLI help as authoritative:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/build.py --help
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_mesh_to_usd_workflow.py --help
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/run_drop_test.py --help
```

Preserve unrelated local changes and existing results.

## Determine the input contract

Choose the contract from the supplied paths without interviewing the user:

- Standalone mesh: one nonempty USD/USDA/USDC, OBJ, FBX, glTF/GLB, or STL
  asset. Its physical scale must already be correct.
- Exact exported sequence: `object_mesh/output_aligned.glb`, `poses.npy`, and
  `ground_plane.json`.
- Catalog or cross-mesh support: a target mesh beside
  `output_symmetry.json`, plus a sequence containing `edex`, per-camera RGB,
  depth and mask H5 sources, and `ground_plane.json`.
- Batch: catalog directories joined to exported sequences by exact
  `object.id` in `hoi_metadata.yaml`.

Fail on a missing required file. Do not silently substitute geometry-derived
support for missing recording data.

## Build only missing images

Normal generation plus validation uses:

```text
v2d_hoi_mesh_to_usd
v2d_hoi_mesh_to_usd_validator
```

Inspect them with `docker image inspect`. Build both when either is missing:

```bash
python modules/v2d_hoi_object_reconstruction/mesh_to_usd/build.py
```

Use `--target generator` or `--target validator` only for a deliberately scoped
repair. Use `--dev` at runtime only when validating local source changes.

Cross-mesh support additionally needs FoundationPose:

```bash
python modules/v2d_foundation_pose/docker/build.py
python modules/v2d_foundation_pose/docker/run_download_weights.py \
  --output_dir data/weights/foundationpose
```

Do not download FoundationPose weights for modes that do not invoke it.

## Prove readiness

Before declaring setup complete, show PASS, FAIL, or BLOCKED for each applicable
gate:

1. All host `--help` commands above succeed.
2. The selected input contract exists and every required file is nonempty.
3. `v2d_hoi_mesh_to_usd` and `v2d_hoi_mesh_to_usd_validator` exist.
4. GPU 0 is visible inside the generator image:

   ```bash
   docker run --rm --gpus '"device=0"' --entrypoint nvidia-smi \
     v2d_hoi_mesh_to_usd
   ```

5. Cross-mesh mode, when selected, has the FoundationPose image, populated
   weights, and complete per-camera inputs.
6. The user has explicitly accepted the EULA or will provide
   `ACCEPT_EULA=Y` before an Isaac Sim command. Treat this as a run gate, not a
   reason to skip other setup checks.

Continue with `mesh-to-usd-run` when the request also includes generation or a
drop test. Do not stop after setup merely because the full simulation is long.
