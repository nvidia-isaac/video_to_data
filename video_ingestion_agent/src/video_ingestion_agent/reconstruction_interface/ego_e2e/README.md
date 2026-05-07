# Ego e2e — full hand + object reconstruction per ingested action segment

For each action segment produced by `video_ingestion_agent`, run the full
16-stage [ego hand + object reconstruction pipeline](../../../../../../reconstruction/modules/v2d_pipelines/run_v2d_ego_e2e.py)
end-to-end: ego-hand reconstruction (ViPE + Dyn-HaMR) → MoGe + VIPE depth
(dual-source) → Grounding DINO → SAM2 → SAM3D textured mesh →
FoundationPose scale + 6-DoF tracking + EKF smoothing → hand/object depth
alignment → 2×2 grid renders.

This is the sole reconstruction chain wired into the webapp's Reconstruct
tab — ~10–15 min per segment, producing hand-aligned 6-DoF object poses
plus the metric object mesh.

## Architecture

```
                ingestion's .venv (no v2d.* / torch / CUDA)
                            │
                            │ python -m video_ingestion_agent.reconstruction_interface.ego_e2e.run_ego_e2e
                            ▼
       ┌──────────────────────────────────────────────────────┐
       │  this wrapper (~80 LOC)                              │
       │  - read clips_final.jsonl                            │
       │  - per segment: ffmpeg-slice + subprocess.run        │
       │  - stdout streams straight through (orchestrator     │
       │    emits `[run ]` / `[skip] <label>` markers; the    │
       │    webapp's reconstruction service parses them to    │
       │    drive the 16-stage status bar)                    │
       └────────────────────┬─────────────────────────────────┘
                            │ subprocess.run([reconstruction_python, run_v2d_ego_e2e.py, ...])
                            ▼
                reconstruction/.venv  (lightweight, ~20 MB:
                                        torch-free orchestration only)
                            │
                            │ python -m v2d.pipelines.run_v2d_ego_e2e
                            ▼
              run_v2d_ego_e2e.py orchestrator (628 LOC)
                            │ 16 host-side docker subprocess calls
                            ▼
              v2d_*:latest containers (heavy ML lives here)
```

The orchestrator is the source of truth for the 16 stages and their params;
this wrapper only does ffmpeg slicing and subprocess hand-off.

## Prerequisites

Three one-time setup tasks. Most of this is documented authoritatively at
[`reconstruction/docs/ego_e2e_setup.md`](../../../../../../reconstruction/docs/ego_e2e_setup.md);
this is a quick reference plus the gotchas we found during real-clip
validation.

### 1. Build all v2d_* container images (~30 min, ~135 GB)

```bash
cd reconstruction
bash scripts/build_containers.sh
# Builds 9 images we use: v2d_moge, v2d_grounding_dino, v2d_sam2, v2d_sam3d,
# v2d_mesh, v2d_foundation_pose, v2d_hand_alignment, ego_vipe, ego_dynhamr.
# The latter two come from v2d_ego_hand_reconstruction's vendored sources;
# `sync.sh` runs first to fetch IsaacTeleop subtree.
```

### 2. Set up reconstruction's `.venv` (~5 min, ~20 MB)

The orchestrator imports `v2d.*` Python directly and has three host-side
helpers (EXR convert, SAM3D-transform application, intrinsics
stabilization) that need their deps in *its* venv. None of this is torch —
just lightweight wrappers + trimesh + OpenEXR.

**Don't use** `reconstruction/scripts/install_packages.sh` — it's
incomplete (missing `v2d_depth`, `v2d_hand_alignment/docker`,
`v2d_mesh/docker`, `trimesh`, `OpenEXR`, `Imath`) and the integration
will fail at run time with `ModuleNotFoundError`. Use this recipe
instead — it's the working set:

```bash
cd reconstruction
uv venv
uv pip install \
    -e modules/v2d_docker \
    -e modules/v2d_common \
    -e modules/v2d_depth \
    -e modules/v2d_sam2/docker \
    -e modules/v2d_sam3d/docker \
    -e modules/v2d_mesh/docker \
    -e modules/v2d_moge/docker \
    -e modules/v2d_grounding_dino/docker \
    -e modules/v2d_foundation_pose/docker \
    -e modules/v2d_ego_hand_reconstruction/docker \
    -e modules/v2d_hand_alignment/docker \
    -e modules/v2d_pipelines \
    trimesh OpenEXR Imath

# Sanity:
uv run python -c "from v2d.pipelines.run_v2d_ego_e2e import run_v2d_ego_e2e; print('OK')"
```

### 3. Stage MANO + BMC weights (~15 min, ~10 MB)

Two consumers expect different layouts; **stage MANO_RIGHT.pkl in both
locations**:

```bash
mkdir -p /tmp/hand_weights/{models,BMC}

# (3a) MANO — gated download from https://mano.is.tue.mpg.de/
unzip mano_v1_2.zip -d /tmp/mano_extract
cp /tmp/mano_extract/mano_v1_2/models/MANO_RIGHT.pkl   /tmp/hand_weights/MANO_RIGHT.pkl         # for Dyn-HaMR
cp /tmp/mano_extract/mano_v1_2/models/MANO_RIGHT.pkl   /tmp/hand_weights/models/MANO_RIGHT.pkl  # for v2d_hand_alignment

# (3b) BMC — generate from https://github.com/MengHao666/Hand-BMC-pytorch
git clone --depth 1 https://github.com/MengHao666/Hand-BMC-pytorch /tmp/Hand-BMC-pytorch
cd /tmp/Hand-BMC-pytorch
uv tool run gdown 1_wV8QjmsVCMBEBhm56gFA2XTyU8VEHzk -O joints.zip   # 91 MB joints data from Google Drive
unzip -q joints.zip
uv run --with numpy --with tqdm --with matplotlib python calculate_bmc.py
cp BMC/*.npy /tmp/hand_weights/BMC/
```

Final layout:

```
/tmp/hand_weights/
├── MANO_RIGHT.pkl                 # Dyn-HaMR step (1) reads it from here
├── models/
│   └── MANO_RIGHT.pkl             # v2d_hand_alignment step (14) reads it from here
└── BMC/
    └── *.npy                      # Dyn-HaMR's bone-length / curvature constraints
```

## Run

From `video_ingestion_agent/`, in its `.venv`:

```bash
uv run --no-sync python -m video_ingestion_agent.reconstruction_interface.ego_e2e.run_ego_e2e \
    --segments runs/<run>/clips_final.jsonl \
    --reconstruction-python ../reconstruction/.venv/bin/python \
    --reconstruction-root   ../reconstruction \
    --out outputs/ego_e2e/ \
    --moge-weights                /tmp/moge_weights \
    --grounding-dino-weights      /tmp/gd_weights \
    --sam2-weights                /tmp/sam2_weights \
    --sam3d-weights               /tmp/sam3d_weights \
    --foundation-pose-weights     /tmp/fp_weights \
    --hand-reconstruction-weights /tmp/hand_weights \
    --depth-source moge \
    --limit 1
```

Useful flags:

- `--depth-source moge|vipe` — both depth sources are always computed by the
  orchestrator (for comparison). This flag selects which one feeds SAM3D +
  FoundationPose + alignment.
- `--ref-frame N` — frame index for DINO / SAM3D / FP registration (default 0).
- `--reregister-iou-thresh F` — FP re-registration IoU threshold (default 0.3).
- `--smooth-sigma F` — Gaussian sigma in frames for hand-translation smoothing
  (default 5.0).
- `--limit N` — process only the first N segments (smoke testing).

The webapp's Reconstruct tab exposes the same wrapper as a UI button.

## Output (per segment, with `--depth-source moge`)

```
outputs/ego_e2e/<segment_id>/
├── clip.mp4                              # host ffmpeg slice
├── frames/<idx:06d>.png                  # extracted RGB
├── depth/<idx:06d>.png                   # MoGe depth
├── depth_vipe/<idx:06d>.png              # DynHaMR depth (always computed)
├── intrinsics/<idx:06d>.json
├── intrinsics_stable.json                # MoGe temporally medianed
├── intrinsics_vipe.json                  # DynHaMR camera intrinsics
├── masks/1/<idx:06d>.png                 # SAM2 per-frame masks
├── dino_detections.json
├── sam2_prompts.json
├── mesh_moge/textured_mesh.obj           # SAM3D output (with mesh_transform.json + mesh_intrinsics.json)
├── mesh_pretransformed_moge.obj          # SAM3D rotation+scale baked in
├── mesh_scaled_moge.obj                  # final metric-scale object mesh  ← UI viewer
├── scale_moge.json                       # FoundationPose scale-estimation report
├── poses_moge/<idx:06d>.json             # raw FP per-frame 6-DoF
├── poses_smoothed_moge/<idx:06d>.json    # EKF-smoothed
├── poses_smoothed_render_moge/           # FP overlay PNGs
├── world_results_aligned_moge.npz        # final aligned hand + object poses
├── render_aligned_moge.mp4                # 2×2 grid (trans_aligned)  ← UI viewer
├── render_unaligned_moge.mp4              # 2×2 grid (trans, comparison)
└── hand_reconstruction/                   # Dyn-HaMR + ViPE outputs
    └── logs/.../smooth_fit/*_world_results.npz
```

## Caching / re-running

The orchestrator's own `_step()` short-circuits each stage if its primary
output already exists on disk — re-invoking the wrapper on the same `--out`
is fast (everything `[skip]`s). To force a fresh run of a particular stage,
delete its primary output (e.g. `rm seg_dir/mesh_scaled_moge.obj` to redo
the FoundationPose scale + everything downstream).

## How the isolation works

- This wrapper runs in `video_ingestion_agent/.venv` — pure stdlib + the
  package's own `_common/ingestion_io.py`. **No `v2d.*` or torch packages
  installed alongside.** Verify with `pip list | grep v2d` after a run —
  empty.
- Reconstruction's lightweight orchestration packages live in
  `reconstruction/.venv` (separate, ~20 MB, torch-free). The orchestrator
  spawns the heavy ML containers via `docker run`.
- We cross the boundary between the two venvs once per segment via
  `subprocess.run([reconstruction_python, run_v2d_ego_e2e.py, ...])`. The
  orchestrator's stdout (including `[run ]` / `[skip]` markers) streams
  straight through to whoever invoked us.

## What this is for

Research-grade output: paired (hand pose, object pose, mesh) tuples in
metric space, ready for downstream learning / sim-to-real / dataset
construction. Outputs land at:

- `render_aligned_<ds>.mp4` — 2×2 grid render
- `mesh_scaled_<ds>.obj` — final metric-scale object mesh
- `poses_smoothed_<ds>/<idx>.json` — EKF-smoothed 6-DoF object poses
- `world_results_aligned_<ds>.npz` — final aligned hand + object world poses

…where `<ds>` is the chosen `--depth-source` (`moge` or `vipe`).
