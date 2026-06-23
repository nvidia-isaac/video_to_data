---
name: add-dataset
description: Guide for adding a new hand-object motion dataset to the robotic_grounding retarget-to-training pipeline. Use this skill whenever the user mentions adding a new dataset, integrating a new data source, new motion capture data, a new hand-object dataset, or wants to connect a new dataset to the retargeting or RL training pipeline. Also trigger when the user asks "how do I add a dataset", "new dataset support", or mentions dataset names that aren't already in the registry (taco, arctic, oakink2, hot3d, h2o, grab, dexycb).
---

# Adding a New Dataset to robotic_grounding

This skill walks through the 3 steps required to integrate a new hand-object motion dataset into the retarget-to-training pipeline.

> **Repo split (MANO migration):** Stage 1 (Load) lives in the **`reconstruction`**
> repo's `v2d_task_library_loader` module — it uses MANO forward kinematics
> (manotorch, GPL-3.0) and is built into its own image. Stages 1.5+ (URDFs,
> retarget/IK, training) stay in **`robotic_grounding`**, which is manotorch-free
> and consumes the `{dataset}_loaded` Parquet the loader produces. So: the
> **loader** (Step 2) is authored in `reconstruction`; the **registry entry**
> (Step 1) and **retarget script** (Step 3) are authored here in `robotic_grounding`.

## Before You Start

All work is done from the `robotic_grounding/` directory. Gather this information from the user before starting:

| Info needed | Example | Why |
|-------------|---------|-----|
| Dataset name | `"handobj"` | Used everywhere as an identifier |
| Source data format | JSONL, pickle, CSV, NPY | Determines loader implementation |
| FPS | 30.0, 120.0 | Motion sampling rate |
| MANO hand format | PCA coefficients, axis-angle, quaternions | Affects how hand data is parsed |
| MANO kwargs | `flat_hand_mean`, `center_idx` | Controls MANO forward kinematics |
| Mesh format & scale | OBJ in cm, GLB in meters | Determines vertex_scale and mesh_format |
| Object type | Rigid or articulated | Affects URDF generation and scene config |
| Has contact data? | Yes/No | Whether contact rewards will be active |
| Coordinate frame | Z-up, Y-up, OpenCV camera | May need rotation transforms |
| Hands per sequence | Bimanual / single (right or left) | Single-hand datasets need zero-fill for the idle hand |
| File count per subject | ~500 (fine) / ~60k+ (needs tar bundles) | If thousands of tiny .npz files per subject, upload as per-subject tarballs + loader auto-extract — individual S3 uploads can take 12+ hours |
| Provider auth | None / username+password / session cookie / Google Drive | Determines download approach (direct URL, `gdown`, MPG cookie flow, etc.) |

## Step 0: Acquire Raw Data — Two Parallel Tracks

This step has **two things that should run in parallel**:

- **Track A (long-running, background)**: Upload the full raw dataset to CSS
  via an OSMO dev_env. This stages the data for the eventual large-scale
  retargeting workflow. Kick this off first because it can take 30min-2hrs.
- **Track B (start immediately after A is submitted)**: Download a small
  sample to your local machine so you can iterate on the loader/retarget
  scripts in Step 4 without waiting for A to finish.

Don't wait for A to complete before starting B — they're independent.

### Track A: Upload the full dataset to CSS (OSMO dev_env)

Use `workflow/dev_env.yaml` to spin up an SSH-accessible OSMO worker with 2TB of storage, then download + upload interactively.

**Before launching the dev env**, create a dataset-specific download script at
`scripts/download_sources/<dataset_name>.sh`. This script runs inside the OSMO
container and handles the full data acquisition: install tools, download raw
data, extract archives, upload to CSS. Having it as a committed script means
anyone can reproduce the data acquisition, and the logic lives in version
control rather than in someone's shell history.

**Download-source cheat sheet** (from past onboardings):

| Provider style | How to script it | Example |
|----------------|------------------|---------|
| Direct URL / wget | `curl -L -o X.zip <url>` | TACO |
| Google Drive | `pip install gdown; gdown --fuzzy --id <FILE_ID> -O X.tar.gz` (provider must have accepted click-through license on that account first) | DexYCB, Hot3D |
| Session-cookie login (TUE MPG) | `curl -c cookies.txt -d "username=$U&password=$P" https://<dataset>.is.tue.mpg.de/login.php` then `curl -b cookies.txt <download_url>`. **Signed** URLs from the downloads page don't work in a clean shell — they require `PHPSESSID`. | GRAB, AMASS, SMPL-X |
| Provider-bundled helper | Some datasets ship their own extractor (GRAB uses `__`-delimited filenames reassembled by `unzip_grab.py`); prefer the provider's helper over ad-hoc `unzip` | GRAB |
| Single mega-archive | `gdown <single_119GB_id> && tar -xzf` | DexYCB (`dex-ycb-20210415.tar.gz`) |

**Upload strategy** — tune this to the shape of the dataset, not the raw
byte count. If a subject's raw data is thousands of tiny files (e.g. 60k
per-frame `labels_*.npz`), **don't** upload them individually — S3
per-request overhead with 32-way parallelism will take 10+ hours. Pack
per-subject tarballs and teach the loader to auto-extract (see
"Auto-extract archives in the loader" under Step 2 / Common pitfalls).
For datasets with a handful of files per subject, plain `aws s3 sync`
works fine.

Template for `scripts/download_sources/<dataset_name>.sh`:

```bash
#!/bin/bash
# Download <dataset_name> dataset and upload to CSS.
# Run inside an OSMO dev_env container (needs CSS creds + dataset-specific creds).
#
# Required env vars:
#   CSS_ENDPOINT_URL, CSS_ACCESS_KEY, CSS_SECRET_KEY
#   <DATASET_USERNAME>, <DATASET_PASSWORD>  (if dataset requires auth)
set -ex

DATASET=<dataset_name>
STAGING=/tmp/${DATASET}
CSS_DEST=s3://datasets/v2d/human_motion_data/${DATASET}/dataset/

# 1. Install tools (aws CLI, any dataset-specific deps)
apt-get update -qq && apt-get install -y -qq unzip
pip install -q awscli  # may need: python3 -m pip install awscli

# 2. Download the raw dataset (dataset-specific)
mkdir -p ${STAGING} && cd ${STAGING}
# e.g. git clone https://github.com/<dataset>/downloader.git
#      python downloader/download_script.py --username ... --dest ${STAGING}

# 3. Extract archives if any (many datasets ship as .tar.gz or .zip)
# for f in *.tar.gz; do tar -xzf "$f"; done
# for f in *.zip; do unzip -q "$f"; done

# 4. Upload to CSS (use high concurrency for many small files)
aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.max_queue_size 10000
export AWS_ACCESS_KEY_ID=${CSS_ACCESS_KEY}
export AWS_SECRET_ACCESS_KEY=${CSS_SECRET_KEY}
aws s3 sync ${STAGING}/ ${CSS_DEST} \
  --endpoint-url ${CSS_ENDPOINT_URL} --region us-east-1 \
  --exclude "*.tar.gz" --exclude "*.zip"

echo "Upload complete. Verify from host:"
echo "  python scripts/list_css_sequences.py --dataset ${DATASET} --stage raw"
```

See `scripts/download_sources/h2o.sh` as a working example.

Then run the acquisition:

1. Launch the dev env workflow:
```bash
python scripts/run_osmo.py \
  --experiment-name download-<dataset_name> \
  --workflow-yaml workflow/dev_env.yaml
```

2. Port-forward SSH to localhost and connect:
```bash
osmo workflow port-forward robotic_grounding_download-<dataset_name> dev-env --port 6000:22
ssh root@localhost -p 6000
```

3. Inside the worker, set credentials and run the script:
```bash
export CSS_ENDPOINT_URL=https://pdx.s8k.io
export CSS_ACCESS_KEY=<your_access_key>
export CSS_SECRET_KEY=<your_secret_key>
# + any dataset-specific credentials

cd /workspace/robotic_grounding  # or wherever the repo is mounted
bash scripts/download_sources/<dataset_name>.sh
```

4. Verify from host, then terminate the dev-env workflow:
```bash
source scripts/setup_css_env.sh
python scripts/list_css_sequences.py --dataset <dataset_name> --stage raw
```

### Track B: Download a small sample locally for loader development

Don't wait for Track A to finish. Grab the smallest meaningful slice of the
dataset directly from the provider (or wherever the raw data can be
obtained) so you can start implementing the loader.

Source: use whatever the provider supports — project website, GitHub
download script, direct URL, `wget`, `gdown`, etc. Pick just enough to
cover one sequence (one subject / one session / one take).

Target path: `source/robotic_grounding/robotic_grounding/assets/human_motion_data/<dataset>/dataset/`

```bash
mkdir -p source/robotic_grounding/robotic_grounding/assets/human_motion_data/<dataset>/dataset
cd source/robotic_grounding/robotic_grounding/assets/human_motion_data/<dataset>/dataset

# e.g. for H2O (requires auth):
git clone https://github.com/<provider>/<dataset>.git /tmp/dl
python /tmp/dl/download_script.py --mode pose --username $USER --password $PASS --dest .

# e.g. from a direct URL:
wget https://dataset-host/subset.tar.gz
```

This small sample drives **Step 4 (local iteration)**. Keep it tiny — loaders
typically go through 5-10 iterations before the data parses correctly, and
you want each iteration to be seconds, not minutes.

### Check in object meshes to the monorepo

Object meshes must be committed to the repo (tracked with Git LFS):

```bash
cp -r <source_meshes> source/robotic_grounding/robotic_grounding/assets/meshes/<dataset_name>/
git lfs track "source/robotic_grounding/robotic_grounding/assets/meshes/<dataset_name>/*"
git add .gitattributes source/robotic_grounding/robotic_grounding/assets/meshes/<dataset_name>/
```

### Verify upload

```bash
python scripts/list_css_sequences.py --dataset <dataset_name> --stage raw
```

## Step 1: Add a Registry Entry

**File:** `source/robotic_grounding/robotic_grounding/retarget/dataset_registry.py`

Add an entry to the `DATASET_CONFIGS` dict. This is the **single source of truth** — every other script (workflow, CSS sync, URDF generation, training validation) reads from here.

```python
"newdata": DatasetConfig(
    name="newdata",
    fps=30.0,
    mano_kwargs={"flat_hand_mean": False, "center_idx": None},
    mesh_vertex_scale=1.0,      # 0.01 if meshes are in centimeters
    mesh_format="glb",          # "obj" or "glb"
    has_articulated_objects=False,
    has_contact_data=False,      # set True if contact analysis will run
    has_support_surfaces=True,
    link_to_site_quat_wxyz=None, # quaternion (w,x,y,z) for IK, or None
    retarget_scripts={          # robot name -> retarget script (relative to repo root)
        "sharpa_wave": "scripts/retarget/newdata_to_sharpa.py",
    },
),
```

### DatasetConfig field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Short identifier, must match dict key |
| `fps` | float | required | Source motion frame rate |
| `mano_kwargs` | dict | `{}` | `flat_hand_mean` and `center_idx` for ManoLayer |
| `mesh_vertex_scale` | float | 1.0 | Scale to meters (0.01 for cm data like TACO) |
| `mesh_format` | str | "obj" | Source mesh format: "obj" or "glb" |
| `has_articulated_objects` | bool | False | True if objects have moving parts (like Arctic) |
| `has_contact_data` | bool | True | Whether processed parquets will include contact normals/positions |
| `has_support_surfaces` | bool | True | Whether support surface reconstruction is expected |
| `link_to_site_quat_wxyz` | tuple/None | None | MANO link-to-site quaternion for IK retargeting |
| `loaded_suffix` | str | "_loaded" | Directory suffix for loaded stage |
| `processed_suffix` | str | "_processed" | Directory suffix for processed stage |
| `css_raw_prefix` | str | "" | CSS raw data subdirectory (empty = "dataset/") |
| `loader_script` | str | "" | **Deprecated/unused** — Stage-1 loaders moved to reconstruction's `v2d_task_library_loader`. Left empty. |
| `retarget_scripts` | dict | `{}` | Maps robot name (`"sharpa_wave"`, `"dex3"`) → retarget script path (relative to repo root) |

## Step 2: Write the Loader Script (in the `reconstruction` repo)

**New file:** `reconstruction/modules/v2d_task_library_loader/lib/newdata_loader.py`

> This is the **GPL/MANO stage** — it lives in `reconstruction`, not here,
> because it uses manotorch forward kinematics. After writing it, register the
> class in `reconstruction/modules/v2d_task_library_loader/lib/loader_registry.py`
> so `run_loader.py --dataset newdata` can dispatch to it. The loader imports the
> Parquet schema and MANO constants from `robotic_grounding` (which reconstruction
> depends on), so the `ManoSharpaData` contract stays single-sourced.

This script reads raw motion data and converts it to the `ManoSharpaData` Parquet schema (MANO hand parameters + object poses per frame).

### Subclass DatasetLoaderBase

The base class is at `reconstruction/modules/v2d_task_library_loader/lib/dataset_loader_base.py`. Implement these required abstract methods:

```python
class NewDatasetLoader(DatasetLoaderBase):

    def list_sequences(self, args) -> list[SequenceInfo]:
        """Discover and return all sequences from the raw dataset directory.
        Returns a list of SequenceInfo(sequence_id, raw_motion_file,
        object_name, object_body_names, source)."""
        ...

    def load_mano_data(self, sequence_info, device) -> dict:
        """Load MANO hand parameters for a sequence.
        Returns dict with keys: H (num_frames),
        right/left_global_orient (H,3), right/left_finger_pose (H,45),
        right/left_trans (H,3), right/left_betas (10,),
        right/left_fitting_err (H,)."""
        ...

    def load_object_data(self, sequence_info) -> dict:
        """Load object poses. Returns dict mapping body_name ->
        (pose_Nx4x4, root_position_Nx3, root_axis_angle_Nx3,
        articulation_N_or_None)."""
        ...

    def load_object_meshes(self, sequence_info, device) -> tuple:
        """Load trimesh objects. Returns (meshes, verts, faces,
        surface_points, surface_normals, compute_tips_dist)."""
        ...

    def get_mano_kwargs(self) -> dict:
        """Return {'flat_hand_mean': ..., 'center_idx': ...}."""
        ...

    def get_fps(self) -> float:
        """Return the dataset's frame rate."""
        ...
```

### Override these optional methods

```python
    def get_object_mesh_paths(self, sequence_info) -> list[str]:
        """Return absolute paths to mesh files (one per object body)."""
        ...

    def get_object_urdf_paths(self, sequence_info) -> list[str]:
        """Return paths where rigid URDFs will be generated.
        Convention: assets/urdfs/{dataset}/{id}_rigid.urdf"""
        ...

    def get_frame_range(self, num_frames) -> tuple[int, int]:
        """Override to trim frames (e.g., skip first/last N)."""
        ...
```

### CLI entry points (required pattern)

Every loader must expose `parse_args()` and `main(args)` for the generic dispatch to work:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load <dataset> sequences.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save", action="store_true")
    # Add dataset-specific args (e.g., --data_root)
    DatasetLoaderBase.add_filter_args(parser)  # adds --sequence_id, --sequence_pattern, etc.
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    loader = NewDatasetLoader()
    loader.run(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
```

### Reference implementations

Use these as templates, ordered from simplest to most complex (all in
`reconstruction/modules/v2d_task_library_loader/lib/`):

| Loader | Best for | Key patterns |
|--------|----------|-------------|
| `taco_loader.py` | OBJ meshes in centimeters, pickle/NPY | Simple two-object (tool+target) structure |
| `h2o_loader.py` | Per-frame text-format labels, bimanual with validity flags | `_extract_archives_if_needed`, per-frame `flag=0` skip, multi-camera |
| `dexycb_loader.py` | Per-frame `.npz`, PCA MANO, single-hand | `_extract_archives_if_needed`, PCA→AA expansion with `hands_mean`, idle-hand zero-fill, `os.walk`-based enumeration |
| `grab_loader.py` | Per-sequence `.npz` with SMPL-X + MANO, bimanual, full body | SMPL-X full-body context, rotational world-frame correction via betas-aware `new_transl`, filename split-from-left |
| `hot3d_loader.py` | GLB meshes, JSONL/CSV source, PCA MANO (ncomps=15) | Coordinate frame transforms, timestamp dedup, PCA expansion |
| `oakink2_loader.py` | Variable object count, deferred discovery | Y-up to Z-up transforms, frame filtering |
| `arctic_loader.py` | Articulated objects | MuJoCo FK for articulated parts |

### Common pitfalls

- **Coordinate frames**: If your data is Y-up or camera-frame (OpenCV/OpenGL), rotate to Z-up world before storing. See `hot3d_loader.QUEST3_WORLD_TO_ZUP`, `h2o_loader.CAM_TO_WORLD_ZUP`, `grab_loader.Y_UP_TO_Z_UP`.
- **MANO PCA expansion**: If hand poses are stored as PCA coefficients (`pose_m[:, 3:48]` on DexYCB, 15 components on Hot3D), expand to 45-DOF axis-angle via `ManoLayer.th_selected_comps`. Remember to **add `th_hands_mean`** when the dataset was fitted with `flat_hand_mean=False` — omitting it leaves the hand in an implausible "flat" baseline. See `dexycb_loader._expand_pca_to_aa`.
- **Single-hand datasets**: If each sequence only has one hand annotated (H2O frame-level, DexYCB session-level), **zero-fill the idle hand** (`global_orient=0`, `finger_pose=0`, `trans=0`, `betas=0`). MANO FK on zeros produces a neutral hand at the origin, IK tracks it harmlessly, and the `tip_distance` quality check uses `min(left, right)` so the idle hand doesn't skew rejection. If the active flag is per-frame (H2O), raising `ValueError` from `load_mano_data` skips the whole sequence; the base loader catches it.
- **Filename parsing**: Datasets use irregular take suffixes like `camera_takepicture_3_Retake.npz`. Split from the **left** (`stem.partition('_')`) to extract the object name — `rsplit` will misidentify it because multiple trailing tokens aren't uniformly `{action}_{take}`.
- **Auto-extract archives in the loader**: If the dataset ships as per-subject tarballs on CSS (Step 0 tar-pack strategy), add an `_extract_archives_if_needed(dataset_dir)` helper to the loader (see `h2o_loader` / `dexycb_loader`). It: (1) globs `*.tar.gz` at the dataset root, (2) skips tarballs whose target dir already exists, (3) falls back to a `/tmp` cache if the mount is read-only, (4) runs once per workflow before `list_sequences`. Without this, the OSMO workflow sees zero files after the CSS download.
- **Mesh scale**: If meshes are in centimeters, set `mesh_vertex_scale=0.01` in the registry AND apply the same scale in `load_object_meshes()`.
- **Object body names**: These become USD prim names — avoid special characters. Use `make_usd_safe()` from `dataset_loader_base.py`.
- **Underscore normalization**: Names in per-frame files and mesh filenames sometimes differ (GRAB: npz says `alarmclock`, ContactDB STL says `alarm_clock`). Use the npz-side name as the canonical identifier; meshes that don't match should fall back cleanly or skip the sequence.

## Step 3: Write the Retarget Script

**New file:** `scripts/retarget/newdata_to_sharpa.py`

This script reads loaded Parquet data and runs inverse kinematics (IK) to produce robot joint trajectories.

### Template structure

```python
from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.retarget.sharpa_kinematics import SharpaWaveKinematics

LINK_TO_SITE_QUAT_XYZW = None  # or dataset-specific quaternion

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget <dataset> to Sharpa.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    DatasetLoaderBase.add_filter_args(parser)
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    """Read loaded Parquet, run IK per frame, save retargeted Parquet."""
    device = torch.device(args.device)
    # 1. Read loaded parquet
    # 2. Initialize SharpaWaveKinematics
    # 3. Per-frame IK: mano wrist/finger -> robot joints
    # 4. Save retargeted ManoSharpaData parquet
    ...

if __name__ == "__main__":
    args = parse_args()
    main(args)
```

### Reference implementations

Use `hot3d_to_sharpa.py` as the closest template — it's the most recent and follows the cleanest pattern. The IK loop structure is nearly identical across all datasets; what varies is:

- `LINK_TO_SITE_QUAT_XYZW` — Arctic uses `(−0.5, 0.5, 0.5, 0.5)`, others use `None`
- Visualization helpers (optional, dataset-specific)

## After All 3 Steps — Mostly Automatic

Most scripts pick up the new dataset automatically via the registry:

| What | How it works |
|------|-------------|
| **OSMO workflow** | Stage 1 Load: reconstruction `v2d_{dataset}_load` (`lib/run_loader.py` dispatch). Stages 1.5+: RG `retarget.yaml` (`run_retarget.py` dispatch via registry), consuming the `{dataset}_loaded` dataset |
| **CSS browse/sync** | `list_css_sequences.py` and `sync_css_data.py` pick up the new name from `get_all_dataset_names()` |
| **Stage 4 viz** | `vis_retargeted.py` pulls `--dataset` choices + filter args from the registry |
| **Stage 3 reconstruct** | `reconstruct_support_surfaces.py` pulls `--dataset` choices from the registry |
| **Stage 6 metrics** | `data_assessor.py` is dataset-agnostic |
| **Training** | `SceneConfig` resolves URDFs; contact data handled gracefully if missing |
| **Validation** | `validate_training_assets.py --dataset <name>` works |

**Not automatic** — you still need a small, dataset-specific branch:

### Step 3.5: Register object discovery in `generate_rigid_urdfs.py`

If your dataset ships meshes on CSS (not committed to the repo), add a
`_discover_<name>_objects()` function and wire it into `_discover_objects`.

```python
# scripts/generate_rigid_urdfs.py
def _discover_<name>_objects() -> dict[str, tuple[Path, Path]]:
    # Prefer a committed canonical copy, fall back to the runtime CSS mount.
    canonical_dir = ASSET_DIR / "meshes" / "<name>"
    runtime_dir = HUMAN_MOTION_DATA_DIR / "<name>" / "dataset" / ".../contact_meshes"
    urdf_out_dir = URDF_DIR / "<name>"
    ...
    return {object_id: (mesh_path, urdf_path) for ...}

def _discover_objects(dataset: str) -> dict[str, tuple[Path, Path]]:
    ...
    elif dataset == "<name>":
        return _discover_<name>_objects()
```

**Critical**: `generate_rigid_urdfs.py` must `from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR` — **do not redefine it locally**. The OSMO workflow sets `HUMAN_MOTION_DATA_DIR=/tmp/human_motion_data`; a local redefinition silently reverts to the in-image path and Stage 1.5 produces 0 URDFs without error (it's swallowed by `|| true` in the workflow).

## Step 4: Iterate locally on the small sample

**Don't debug on OSMO.** The build+push+submit cycle is ~20 min/iteration.
The repo is volume-mounted into the Docker container (via `./workflow/run.sh`),
so Python edits take effect on the next invocation — no rebuild needed.

Start the container once (separate terminal):
```bash
./workflow/run.sh start latest 0
```

Then iterate on load → retarget → visualize:

```bash
# Loader (CPU is fine for MANO FK). Runs from the RECONSTRUCTION loader module
# (it has manotorch); iterate there, not in the RG container. The data-root flag
# name follows the dataset: --h2o_dir, --grab_dir, --dexycb_dir, etc. Check your
# loader's parse_args() for the exact flag.
#   cd reconstruction/modules/v2d_task_library_loader
python -m v2d.task_library_loader.lib.run_loader --dataset <dataset> \
  --dataset_root <raw_dataset_dir> \
  --output_dir <out>/<dataset>_loaded \
  --device cpu --save --max_sequences 1

# Retarget (GPU needed for IK) — back in the robotic_grounding container
docker exec robotic-grounding-latest-gpu0 python scripts/retarget/<dataset>_to_sharpa.py \
  --input_dir .../<dataset>_loaded \
  --output_dir .../<dataset>_processed \
  --device cuda:0 --save --max_sequences 1

# Save Viser recording, copy to host, serve for browser playback
docker exec robotic-grounding-latest-gpu0 python scripts/retarget/vis_retargeted.py \
  --input_dir .../<dataset>_processed \
  --show_mano --save_html --html_dir /tmp/viz
docker cp robotic-grounding-latest-gpu0:/tmp/viz /tmp/claude/viz
cd /tmp/claude/viz && python3 -m http.server 8765 &
# Open: http://localhost:8765/viser-client/?playbackPath=http://localhost:8765/recordings/<seq_id>.viser
```

Common bugs to watch for in the recording:
- **Hands swapped** — flip the left/right split in the loader
- **Hands floating or upside down** — the coordinate-frame rotation (e.g. `CAM_TO_WORLD_ZUP`) is wrong
- **Object shows as a sphere** — `get_object_mesh_paths` points to a file that doesn't exist
- **Idle hand glued at wrong origin** — for single-hand datasets, the zero-filled hand should sit at world origin; if it's drifting, the retarget isn't honoring the zero pose (check the loader produces strictly zero `finger_pose` and `trans` for the idle side)

## Step 5: Scale to OSMO

Once the Viser recording looks right (hands grip the object, motion is smooth,
no obvious flips or floating), submit to OSMO. By this point Track A should have
finished uploading the full raw dataset to CSS. This is now **two chained
workflows** (MANO migration):

**(a) Load** — run the reconstruction loader to produce the `{dataset}_loaded`
OSMO dataset (this is the GPL/MANO stage, in the reconstruction repo):

```bash
osmo workflow submit reconstruction/workflows/task_library_load/osmo/load.yaml \
  --set dataset=<dataset> --pool <pool>
```

**(b) Retarget+** — once `{dataset}_loaded` exists, run the RG workflow, which
consumes it and runs `process + reconstruct` across all sequences:

```bash
python scripts/run_osmo.py \
  --experiment-name retarget-<dataset> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=<dataset> \
  --pool groot-l40-01
```

Each step builds + pushes its own image (~15 min) before the workflow runs.

Monitor:
```bash
osmo workflow logs robotic_grounding_retarget-<dataset>-1
```

## Step 6: Training smoke test

```bash
# Sync a couple processed sequences from CSS
python scripts/sync_css_data.py --dataset <dataset> --component processed --max_sequences 2

# Inside the Docker container:
docker exec robotic-grounding-latest-gpu0 python scripts/generate_rigid_urdfs.py --dataset <dataset>
docker exec robotic-grounding-latest-gpu0 python scripts/validate_training_assets.py --dataset <dataset>

# 3-iteration training run
docker exec robotic-grounding-latest-gpu0 python scripts/rsl_rl/train.py \
  --task Sharpa-V2P-v0 --headless \
  --motion_file <dataset>/<dataset>_processed/<seq_id>/sharpa_wave \
  --max_iterations 3 --num_envs 8
```
