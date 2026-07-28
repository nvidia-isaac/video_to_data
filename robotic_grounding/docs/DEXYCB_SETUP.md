# DexYCB end-to-end from scratch (original source)

DexYCB (NVIDIA, CVPR 2021) is a runtime-mesh dataset: the YCB object meshes ship in the
download, and rigid object URDFs are generated from them by the `urdf` stage (run by
default). MANO is PCA (expanded by the loader). It is a grasping benchmark of 1000 short
single-grasp sequences (each ~72 frames at 30 Hz). No segmentation. Distributed on
**Google Drive**, CC BY-NC 4.0, **no registration**. (Two images / two patterns — see
`SETUP.md` §1.)

```
DexYCB <subject>.tar.gz + models.tar.gz + calibration.tar.gz ─[place in dataset dir]→
  <HMD>/dexycb/dataset/  ─[LOAD auto-extracts tarballs, MANO FK]→ dexycb_loaded
  ─[URDF→PROCESSED→SUPPORT→VIS]→ outputs
```
`HMD=~/datasets/human_motion_data`

---

## 1. Download (Google Drive)

> **Downloading by hand?** (browser, signed into a Google account) Get **`models.tar.gz`**
> (1.4 GB) + **`calibration.tar.gz`** (16 KB) + **one** subject tarball
> (`2020NNNN-subject-NN.tar.gz`, ~12 GB; all 10 for the full set). **Skip** `bop.tar.gz`
> (1.2 GB, unused) and the 119 GB single archive `dex-ycb-20210415.tar.gz` (only grab that if
> you'd rather download one file). If a subject hits Drive's quota, open it signed in and
> **"Make a copy"** to your Drive, then download your copy.

Source: **https://dex-ycb.github.io/** (CC BY-NC 4.0, no registration). For a smoke test we
need **one subject** + `models` (YCB meshes) + `calibration` (the per-subject pose archives
bundle RGB/depth, so each is ~12 GB):

| File | Google Drive id | Size |
|---|---|---|
| `models.tar.gz` (21 YCB meshes) | `1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM` | 1.4 GB |
| `calibration.tar.gz` | `1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ` | 16 KB |
| **any one** subject below | (see table) | ~12 GB |

Subject archives (any single one is enough for a test):

| subject | file | Google Drive id |
|---|---|---|
| 01 | `20200709-subject-01.tar.gz` | `1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk` |
| 02 | `20200813-subject-02.tar.gz` | `1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62` |
| 03 | `20200820-subject-03.tar.gz` | `1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq` |
| 04 | `20200903-subject-04.tar.gz` | `14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA` |
| 05 | `20200908-subject-05.tar.gz` | `1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS` |
| 06 | `20200918-subject-06.tar.gz` | `1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ` |
| 07 | `20200928-subject-07.tar.gz` | `1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0` |
| 08 | `20201002-subject-08.tar.gz` | `1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv` |
| 09 | `20201015-subject-09.tar.gz` | `1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH` |
| 10 | `20201022-subject-10.tar.gz` | `1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-` |

(Single-archive `dex-ycb-20210415.tar.gz` = 119 GB has everything.)

Download all three files with **authenticated rclone**. It counts against your own Google
account, so it avoids the anonymous-download quota (*"Too many users have downloaded this file
recently"*) that the ~12 GB subject tarballs reliably trip — that cap blocks `wget`/`curl`/the
browser too, often for every subject at once:
```bash
# one-time auth (any Google account; headless: it prints an `rclone authorize "drive"`
# step to run on a machine with a browser, then paste the token back):
rclone config            # new remote "gdrive", storage "drive", scope 1, blank client_id
rclone lsd gdrive:       # confirm auth works

mkdir -p ~/dexycb_download
# fetch each file by its Drive ID (no progress bar — watch the .partial grow):
rclone -v backend copyid gdrive: 1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM ~/dexycb_download/models.tar.gz
rclone -v backend copyid gdrive: 1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ ~/dexycb_download/calibration.tar.gz
rclone -v backend copyid gdrive: 1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ ~/dexycb_download/20200918-subject-06.tar.gz
```
Swap the subject ID/name (from the table above) for a different subject, or add more lines for
the full set. If you can't use rclone: retry later (the cap clears within ~24 h), or open the
file in a browser signed into Google and **"Make a copy"** to your Drive (a copy you own has
its own quota), then download your copy.

## 2. Set up the dataset directory

> **No fetch/sync script needed** — download the data directly from its original source (above);
> there's no automated fetcher to run. The pipeline only needs the files arranged in the
> directory hierarchy shown below — match this layout and the loader finds everything.

The loader (`DEFAULT_DEXYCB_DIR = <HMD>/dexycb/dataset`) **auto-extracts** any `*.tar.gz`
placed in that dir (`_extract_archives_if_needed`), restoring the canonical layout — so you
just drop the three tarballs in:
```bash
DST=$HMD/dexycb/dataset; mkdir -p "$DST"
cp ~/dexycb_download/{20200709-subject-01,models,calibration}.tar.gz "$DST/"
```
After the loader (or a manual `tar xzf … -C "$DST"`) expands them:
```
$HMD/dexycb/dataset/
├── 20200709-subject-01/<session>/<camera>/labels_*.npz   # per-frame pose_y (object) + pose_m (MANO PCA)
│                       <session>/meta.yml                 # ycb_ids, mano_side, mano_calib
├── calibration/ (intrinsics, extrinsics, mano_<calib>/mano.yml)
└── models/<ycb_name>/textured_simple.obj                  # 21 YCB object meshes
```
**No `object_assets/`** — DexYCB is runtime-mesh: meshes are read from `models/`.

## 3. Run the pipeline (host orchestrator, Pattern A)
```bash
cd <repo>/robotic_grounding
python scripts/run_pipeline_docker.py dexycb \
    --hmd $HMD --mano-dir $HMD/mano \
    --max-sequences 2          # 'auto' stages: load,processed,support,vis
```
- Orchestrator treats dexycb as **runtime-mesh** (`RUNTIME_MESH_DATASETS`): LOAD omits
  `--object_assets_dir`; retarget stages mount the data root at `/data/human_motion_data`.
- **No `segment`**. `urdf` runs by default — `generate_rigid_urdfs` builds the rigid object
  URDFs from `models/` into `object_assets/urdfs/dexycb/` (used downstream for sim/RL).
- Outputs under `$HMD/dexycb/`: `dexycb_loaded`, `dexycb_processed`, `reconstructed_stage`, `dexycb_html`.

Pattern B (shell into the persistent container): `bash run_load_local.sh dexycb` on the host,
then inside the container `bash run_retarget_local.sh dexycb`.

## 4. Gotchas
- MANO is **PCA** (`pose_m[:,0:48]` = 3 global + 45 PCA; `flat_hand_mean=False`) — the loader
  expands to 45-DOF (adds `hands_mean`). Per-subject betas from `calibration/mano_<calib>/mano.yml`.
- Object pose `pose_y` is in **camera frame** per the labels; the loader handles the transform.
- The per-subject tarball bundles RGB/depth (~12 GB) — there's no pose-only subset; that's the
  download cost. Only the `labels_*.npz` + `meta.yml` are consumed.
- Google Drive quota (see §1) is the main friction; `models`/`calibration` are reliable.

## Citation
If you use this dataset, please cite the original work:

```bibtex
@inproceedings{chao2021dexycb,
  title     = {{DexYCB}: A Benchmark for Capturing Hand Grasping of Objects},
  author    = {Chao, Yu-Wei and Yang, Wei and Xiang, Yu and Molchanov, Pavlo and Handa, Ankur and Tremblay, Jonathan and Narang, Yashraj S. and Van Wyk, Karl and Iqbal, Umar and Birchfield, Stan and Kautz, Jan and Fox, Dieter},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```
