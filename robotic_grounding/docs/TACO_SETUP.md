# TACO end-to-end from scratch (original source)

TACO is a rigid tool-use dataset (MANO hands + two rigid objects, `tool` and `target`).
Its object meshes ship as a separate model set, so they live under `object_assets/`
(not runtime-mesh), and there is no segmentation. Same two images / two patterns as the
rest of the pipeline (see `SETUP.md` §1).

```
raw TACO (+ MANO) ─[LOAD]→ taco_loaded ─[URDF→PROCESSED→SUPPORT→VIS]→ outputs
```
`HMD=~/datasets/human_motion_data`

---

## 1. MANO hand models
License-gated; download and place the two `.pkl` as described in `SETUP.md` §5, so that
`$HMD/mano/models/MANO_{LEFT,RIGHT}.pkl` exist. The loader reads only these two files.

## 2. Download the dataset (original Dropbox release)

> **Downloading by hand?** In the Dropbox folder, open and download only these 3 sub-folders:
> **`Object_Poses`** (~34 MB), **`Object_Models`** (~0.42 GB), **`Hand_Poses`** (~15.2 GB).
> **Do NOT** download the whole top-level folder or the RGB/depth video folders
> (`Allocentric_RGB_Videos`, `Egocentric_RGB_Videos`, `Egocentric_Depth_Videos`, …) — hundreds
> of GB, unused. Dropbox zips a folder server-side; `Hand_Poses` (~15 GB) often fails the
> in-browser zip, so download it per-triplet sub-folder (or via the per-file `dl=1` trick
> below) and `zip -T` each archive.

Source: **https://github.com/leolyliu/TACO-Instructions** → Dropbox folder (full V1).
Only **3** of the folders are used (the rest is multi-view RGB video we don't need):

| Folder | Size | Used for |
|---|---|---|
| `Object_Poses` | 34 MB | `(triplet)/{seq}/tool_NNN.npy`, `target_NNN.npy` (4×4 poses) |
| `Object_Models` | 0.42 GB | `object_models_released/NNN_cm.obj` (206 meshes, cm) |
| `Hand_Poses` | **15.2 GB** | `(triplet)/{seq}/{left,right}_hand.pkl` + `*_hand_shape.pkl` (MANO) |

Dropbox **folder** links (`scl/fo`) only serve the JS web app to `curl`. The trick
that works: hit a **per-file** link on the raw-content host `dl.dropboxusercontent.com`
with `dl=1` (returns real `application/zip`, supports resume):
```bash
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
HOST="https://dl.dropboxusercontent.com/scl/fo/8w7xir110nbcnq8uo1845"
RL="rlkey=xnhajvn71ua5i23w75la1nidx&dl=1"
# resumable, stall-retrying download (Hand_Poses is 15 GB and Dropbox throttles):
dl(){ curl -sL -C - --retry 30 --retry-delay 5 --retry-all-errors \
        --speed-limit 2048 --speed-time 30 -A "$UA" -o "$2" "$1"; }
dl "$HOST/AAWfef.../Object_Poses.zip?$RL"  ~/Object_Poses.zip     # per-file IDs differ
dl "$HOST/AAWCslPnomJ2V31dZaON1kY/Object_Models.zip?$RL" ~/Object_Models.zip
dl "$HOST/AAYWfef_wNSfy2XsC7CxFAo/Hand_Poses.zip?$RL"     ~/Hand_Poses.zip
for z in ~/Object_Poses.zip ~/Object_Models.zip ~/Hand_Poses.zip; do zip -T "$z"; done  # verify
```
(Each file's link id is taken from the Dropbox web UI; a truncated zip fails `zip -T`.)

## 3. Set up the dataset directory

> **No fetch/sync script needed** — download the data directly from its original source (above);
> there's no automated fetcher to run. The pipeline only needs the files arranged in the
> directory hierarchy shown below — match this layout and the loader finds everything.

Everything lives under `<HMD>/taco/` — raw poses/hands under `dataset/` (exactly what the
loader reads), object meshes under `object_assets/meshes/taco/` (the data-root location the
orchestrator bind-mounts over the package's `MESHES_DIR` at runtime, so generated STLs/URDFs
stay out of the repo):
```bash
# raw poses/hands -> <HMD>/taco/dataset:
mkdir -p "$HMD/taco/dataset"
unzip -q ~/Object_Poses.zip -d "$HMD/taco/dataset"     # -> Object_Poses/
unzip -q ~/Hand_Poses.zip   -d "$HMD/taco/dataset"     # -> Hand_Poses/

# object meshes -> data-root object_assets (already named NNN_cm.obj == what the
# loader expects; the urdf stage writes NNN_rigid.urdf + STLs alongside):
MESH="$HMD/taco/object_assets/meshes/taco"; mkdir -p "$MESH"
unzip -j -o ~/Object_Models.zip 'object_models_released/*_cm.obj' -d "$MESH"   # 206 meshes
```
Resulting structure:
```
$HMD/taco/
├── dataset/Object_Poses/(triplet)/{seq}/{tool_NNN,target_NNN}.npy
├── dataset/Hand_Poses/(triplet)/{seq}/{left,right}_hand.pkl + *_hand_shape.pkl
└── object_assets/meshes/taco/NNN_cm.obj          # 206 object meshes (cm)
```
Object URDFs are **not** needed for LOAD (the loader only records their path strings;
they're generated in the `urdf` stage). Remove the zips after `zip -T` passes (Hand_Poses
is ~15 GB extracted).

## 4. Run the pipeline (host orchestrator, Pattern A)
```bash
cd <repo>/robotic_grounding
python scripts/run_pipeline_docker.py taco \
    --hmd $HMD --mano-dir $HMD/mano --max-sequences 2
```
- TACO's recommended stages are `load,urdf,processed,support,vis` (the default when
  `--stages auto`). `urdf` generates the rigid object URDFs into `object_assets/urdfs/taco/`.
- No `segment` (TACO clips are already atomic), and no shipped URDFs (the objects are rigid).
- Outputs under `$HMD/taco/`: `taco_loaded`, `taco_processed`, `reconstructed_stage`,
  `taco_html`, and `object_assets/{meshes,urdfs}/taco`.
- Useful knobs: `--stages a,b,c`, `--gpu N`, `--no-mp4` (HTML only), `--with-dummy`
  (heavy Isaac playback MP4), `--dry-run`. Quality filtering is opt-in via `--assess`
  (report-only CSV), `--reject` (also writes `taco_rejected.txt`), `--filter-penetration`
  (runs only the `hand_penetration` check) and `--penetration-threshold CM`.

**Pattern B (shell-in).** LOAD on the host, then enter the persistent container:
```bash
MANO_DIR=$HMD/mano HMD=$HMD bash run_load_local.sh taco
HUMAN_MOTION_DATA_DIR=$HMD ./workflow/run.sh start
# inside the container:
bash run_retarget_local.sh taco                 # core; WITH_VIS=1 / WITH_DUMMY=1 to add stages
```

## 5. Gotchas
- `taco` meshes are in **cm** → the loader scales vertices ×0.01 (`vertex_scale=0.01`); mesh
  basenames must match the `tool_NNN`/`target_NNN` pose stems (TACO already ships `NNN_cm.obj`,
  so no renaming).
- `MESHES_DIR`/`ASSETS_DIR` are package-relative (not env-overridable); only
  `HUMAN_MOTION_DATA_DIR` is. The orchestrator works around this by bind-mounting
  `$HMD/taco/object_assets/{meshes,urdfs}/taco` **over** those in-repo paths at runtime, so
  object meshes + generated STLs/URDFs live under `$HMD` (data root), not the repo.
- The LOAD stage bakes container-absolute mesh paths (`/data/object_assets/meshes/taco/NNN_cm.obj`)
  into the parquet. Downstream stages that trust that path (e.g. `vis_retargeted` MP4 object
  rendering) need the object-assets dir mounted at the same `/data/object_assets` —
  `run_pipeline_docker.py` does this on every retarget stage. Symptom if missing:
  `[mp4] skip object …: string is not a file`. (Support-surface reconstruction is unaffected
  — it resolves meshes via `MESHES_DIR`.)
- `--with-dummy` (Isaac) is RAM-bound (~10 GB/process); concurrency is capped via `--jobs`.

## Citation
If you use this dataset, please cite the original work:

```bibtex
@inproceedings{liu2024taco,
  title     = {{TACO}: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding},
  author    = {Liu, Yun and Yang, Haolin and Si, Xu and Liu, Ling and Li, Zipeng and Zhang, Yuxiang and Liu, Yebin and Yi, Li},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {21740--21751},
  year      = {2024}
}
```
