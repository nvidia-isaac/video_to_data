# Robotics Visualizer

A lightweight gallery server for browsing motion-capture retargeting datasets.
Each sequence shows a split-screen view: **viser 3D playback** (top) and
**MP4 camera feed** (bottom).

No external Python dependencies — stdlib only.

---

## Downloading Dataset Recordings

Recordings are stored in OSMO (`isaac/v2d_{name}_retarget_exp_200`). The sync
script wraps `osmo dataset download` for all six datasets.

```bash
# All datasets → robotic_grounding/visualizer/datasets/
python robotic_grounding/visualizer/sync_visualizer_data.py

# Specific datasets
python robotic_grounding/visualizer/sync_visualizer_data.py --dataset arctic h2o

# Narrow to sequences matching a pattern (regex applied within {name}_html/)
python robotic_grounding/visualizer/sync_visualizer_data.py --dataset arctic --pattern arctic_s01

# Download all datasets in parallel (3 concurrent osmo processes)
python robotic_grounding/visualizer/sync_visualizer_data.py --jobs 3

# Dry run — prints the osmo command without executing
python robotic_grounding/visualizer/sync_visualizer_data.py --dry-run
```

Requires the `osmo` CLI to be on PATH and authenticated:
```bash
osmo login https://us-west-2-aws.osmo.nvidia.com/
```
---

## Quick Start

```bash
# Point at a directory containing v2d_* dataset folders
python robotic_grounding/visualizer/serve.py
# → http://<server-ip>:8080  (serves from visualizer/datasets/ by default)
```

Options:
```bash
python serve.py --port 9000
python serve.py --host 127.0.0.1
python serve.py --data-dir /path/to/external/data   # override data directory
python serve.py --html-dir /path/to/v2d_arctic_retarget_exp_200   # serve a single html output dir directly
```

`--html-dir` accepts a directory that has `recordings/` and `viser-client/` **directly inside** (i.e. the output of `vis_retargeted.py`). It can be repeated to mount multiple directories alongside the standard datasets. Each mount appears as an additional dataset in the sidebar.

---

## Data Directory Layout

The server auto-discovers any directory matching `v2d_{name}_retarget*/`:

```
<data-dir>/
  v2d_arctic_retarget_exp_200/
    arctic_html/
      recordings/
        *.viser          ← 3D playback file (required)
        *.mp4            ← camera feed video (optional)
      viser-client/      ← static viser SPA (copy from any existing dataset)
  v2d_taco_retarget_exp_200/
    ...
```

Restart the server to pick up new datasets — no code changes needed.

---

## Generating `.viser` Files from Retargeted Parquets

`.viser` files are recorded viser sessions — 3D animations of the robot hand
and object playing back a retargeted sequence.

Run `scripts/retarget/vis_retargeted.py` **inside the Docker container**:

```bash
# Start and enter the container (from repo root)
./workflow/run.sh start

# Inside the container —————————————————————————————————————

# 1. Retarget if not already done (writes Parquet to arctic_processed/)
python scripts/retarget/run_retarget.py --dataset arctic --save

# 2. Record .viser (and optionally .mp4) for every sequence in the dataset
python scripts/retarget/vis_retargeted.py \
    --dataset arctic \
    --save_html \
    --save_mp4 \
    --html_dir /workspace/video_to_data/robotic_grounding/visualizer/datasets/v2d_arctic_retarget_exp_200/arctic_html

# Or a single sequence:
python scripts/retarget/vis_retargeted.py \
    --dataset arctic \
    --sequence_id arctic_s01_scissors_use_01 \
    --save_html \
    --save_mp4 \
    --html_dir /workspace/video_to_data/robotic_grounding/visualizer/datasets/v2d_arctic_retarget_exp_200/arctic_html
```

`--save_html` records the animation to a `.viser` file and copies the viser JS
client. `--save_mp4` also renders an offline MP4 via pyrender (no display needed).

The output lands in a flat layout inside `--html_dir`:

```
<html_dir>/
  recordings/
    arctic_s01_scissors_use_01.viser
    arctic_s01_scissors_use_01.mp4
    ...
  viser-client/
```

**Serving from a custom path (outside the standard datasets layout)**

If you write to a path that is not under `visualizer/datasets/`, point the server
at it directly with `--html-dir` **on the host**:

```bash
python robotic_grounding/visualizer/serve.py \
    --html-dir /path/to/arctic_html
# → the dataset appears in the sidebar alongside any downloaded datasets
```

---

## Adding a New Dataset

1. Create the directory structure under your `--data-dir`:
   ```
   v2d_{name}_retarget/
     {name}_html/
       recordings/       ← put .viser and .mp4 files here
       viser-client/     ← copy from any existing dataset directory
   ```

2. Copy the `viser-client/` folder from an existing dataset (they all share the
   same build — the server deduplicates the JS bundle automatically).

3. Restart the server — the new dataset appears in the sidebar immediately.

---

## Current Datasets

| Dataset | Sequences | Avg .viser size |
|---------|-----------|----------------|
| arctic  | 200       | 8.5 MB         |
| dexycb  | 200       | 7.1 MB         |
| grab    | 200       | 14.0 MB        |
| h2o     | 110       | 9.2 MB         |
| hot3d   | 200       | 73.3 MB        |
| taco    | 200       | 10.6 MB        |

> hot3d sequences are 8–10× larger due to data generation differences.
> Each sequence is cached in the browser for 7 days so the slow first load
> only happens once per week.

---

## Technical Notes

- **Loop mechanism:** viser has hardcoded internal looping. The server covers
  the ~30 ms blank flash at each loop boundary with a CSS overlay — no iframe
  reload, so camera position is always preserved.
- **Shared viser-client:** All datasets with the same viser build share one
  canonical URL, so the browser downloads the 2.7 MB JS bundle once regardless
  of how many datasets are loaded.
- **HTTP 206 range requests:** Supported for MP4 video seeking.
- **Cache:** Recordings cached 7 days; hashed JS/CSS/WASM bundles cached 1 year.
