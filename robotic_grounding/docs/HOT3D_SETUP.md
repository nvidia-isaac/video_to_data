# HOT3D end-to-end from scratch (original source)

HOT3D (Project Aria) is an egocentric hand-object dataset. Objects are **rigid**;
their meshes ship as a separate `.glb` library (meters). HOT3D sequences are long
egocentric recordings, so the pipeline **segments** them into atomic interaction
clips before retargeting. See `SETUP.md` §1 for the two-image / Pattern-A-vs-B
overview.

```
HOT3D GT download ─[arrange]→ <HMD>/hot3d/dataset/<seq>/ ─[LOAD]→ hot3d_loaded
  ─[SEGMENT]→ hot3d_loaded_segmented ─[URDF→PROCESSED→SUPPORT→VIS]→ outputs
```

```
HMD=~/datasets/human_motion_data
```

---

## 1. Download

> **Downloading by hand?** The portal gives 3 URL-manifest JSONs (links expire in 14 days).
> You need the **assets** library (`Hot3DAssets_download_urls.json`, fetched with `-l all` —
> the 33 object `.glb` + `instance.json`) and, per sequence, **two** data types:
> **`ground_truth`** (object/headset trajectories, metadata, masks) **and `hand_data`**
> (which holds `mano_hand_pose_trajectory.jsonl` — the MANO poses). **Skip** `main_vrs`
> (raw video — the bulk) and the `mps_*` streams. See §1's index table below — and note
> `-d 0` is the giant `main_vrs`, not what you want. Place files as shown in §2.

Register at **https://www.projectaria.com/datasets/hot3d/** (enter email →
"Access the Datasets") and download the URL files (links expire in 14 days):

```
Hot3DAria_download_urls.json     # Aria sequences
Hot3DQuest_download_urls.json    # Quest3 sequences
Hot3DAssets_download_urls.json   # object mesh library (.glb)
```

The downloader lives in the HOT3D toolkit
(`https://github.com/facebookresearch/hot3d`, at
`hot3d/data_downloader/dataset_downloader_base_main.py`; it only needs
`requests` + `tqdm`). `--data_types`/`-d` takes data-type **indices**;
`--sequence_names`/`-l` takes sequence IDs (or `all`).

**You need TWO data types — `ground_truth` AND `hand_data`:**
- `ground_truth` → `dynamic_objects.csv`, `headset_trajectory.csv`, `metadata.json`,
  `camera_models.json`, `masks/` (object + headset trajectories and metadata).
- `hand_data` → **`mano_hand_pose_trajectory.jsonl`** (the per-frame MANO hand poses the
  loader requires) + `umetrack_*`. **The MANO poses are here, NOT in `ground_truth`** — a
  `ground_truth`-only download will not load.

Both extract into the same `$RAW/<seq>/`.

> **⚠️ The `-d` indices are NOT fixed.** The downloader builds the list **dynamically per
> URL file** (`load_data_groups_from_cdn`), so they differ between the Aria and Quest
> manifests, and **index `0` is `main_vrs` — the ~344 GB (Aria) / ~385 GB (Quest) raw video
> you must NOT download.** Confirm by running the downloader once with `-d` omitted (it prints
> the data-type list for your file, downloads nothing). For the current release:
>
> | manifest | `ground_truth` | `hand_data` |
> |---|---|---|
> | `Hot3DAria_download_urls.json`  | **5** | **6** |
> | `Hot3DQuest_download_urls.json` | **1** | **2** |
>
> `main_vrs`/`mps_*` are raw video/SLAM — skip them. There is no VRS parsing or custom
> exporter; the loader reads HOT3D's native ground-truth + MANO format directly.

```bash
TOOLKIT=~/hot3d/hot3d/data_downloader     # cloned facebookresearch/hot3d
RAW=$HMD/hot3d/_raw;  ASSETS=$HMD/hot3d/_assets
# (optional) confirm the indices for YOUR file — prints the list, downloads nothing:
python "$TOOLKIT/dataset_downloader_base_main.py" -c ~/Hot3DAria_download_urls.json -o /tmp/probe

# all Aria sequences — ground_truth (-d 5) AND hand_data (-d 6):
python "$TOOLKIT/dataset_downloader_base_main.py" -c ~/Hot3DAria_download_urls.json  -o "$RAW" -l all -d 5
python "$TOOLKIT/dataset_downloader_base_main.py" -c ~/Hot3DAria_download_urls.json  -o "$RAW" -l all -d 6
# all Quest sequences — ground_truth (-d 1) AND hand_data (-d 2):
python "$TOOLKIT/dataset_downloader_base_main.py" -c ~/Hot3DQuest_download_urls.json -o "$RAW" -l all -d 1
python "$TOOLKIT/dataset_downloader_base_main.py" -c ~/Hot3DQuest_download_urls.json -o "$RAW" -l all -d 2
# object mesh library (.glb) — single data type, needs -l all:
python "$TOOLKIT/dataset_downloader_base_main.py" -c ~/Hot3DAssets_download_urls.json -o "$ASSETS" -l all
```
(Swap `-l all` for explicit IDs, e.g. `-l P0001_10a27bf7`, for a smoke test. `ground_truth` +
`hand_data` total ~1.1 GB + ~2 GB; the asset library is ~159 MB.)

Each sequence unpacks to `$RAW/<seq>/` (`ground_truth` and `hand_data` files merge into one
dir). Assets unpack to `$ASSETS/assets/<uid>.glb` plus `instance.json`.

---

## 2. Set up the dataset directory

> **No fetch/sync script needed** — download the data directly from its original source (above);
> there's no automated fetcher to run. The pipeline only needs the files arranged in the
> directory hierarchy shown below — match this layout and the loader finds everything.


The loader reads sequences from `<HMD>/hot3d/dataset/<seq>/` (any subdir containing
`mano_hand_pose_trajectory.jsonl`) and object meshes from
`<HMD>/hot3d/object_assets/meshes/hot3d/<uid>.glb` (the data-root location the
orchestrator bind-mounts over the package mesh dir at runtime, so the repo stays
clean):

```bash
cd $HMD/hot3d
mkdir -p dataset object_assets/meshes/hot3d
mv _raw/*/ dataset/                     # all downloaded sequences (Aria + Quest)
cp _assets/assets/*.glb _assets/assets/instance.json object_assets/meshes/hot3d/
```
Each `dataset/<seq>/` must contain `mano_hand_pose_trajectory.jsonl` (from `hand_data`) plus
`dynamic_objects.csv` / `headset_trajectory.csv` / `metadata.json` (from `ground_truth`); a
sequence missing the MANO jsonl was downloaded without `hand_data` and won't load.

Resulting layout:

```
$HMD/hot3d/
  dataset/<seq>/{mano_hand_pose_trajectory.jsonl,dynamic_objects.csv,
                 headset_trajectory.csv,metadata.json,masks/}
  object_assets/meshes/hot3d/{<uid>.glb, instance.json}
```

`metadata.json` ships with each sequence (`have_hand_object_pose_gt`, `headset`,
`object_uids`, `object_names`) — nothing to generate.

---

## 3. Run the pipeline

Pattern A (host orchestrator, recommended) — see `SETUP.md` §3 for the one-time
image build:

```bash
cd <repo>/robotic_grounding
python scripts/run_pipeline_docker.py hot3d \
    --hmd $HMD --mano-dir $HMD/mano --max-sequences 2
```

HOT3D notes:
- The default stages are `load,urdf,processed,support,vis`.
- `segment` (Stage 1.6) **auto-inserts** for hot3d (long egocentric → atomic
  clips); `processed`/`support` then read `hot3d_loaded_segmented`, not
  `hot3d_loaded`.
- `urdf` generates the rigid object URDFs into
  `object_assets/urdfs/hot3d` (from the `.glb` meshes, via
  `generate_rigid_urdfs.py`).
- Outputs land under `<HMD>/hot3d/`: `hot3d_loaded`, `hot3d_loaded_segmented`,
  `hot3d_processed`, `reconstructed_stage`, `hot3d_html`, and
  `object_assets/{meshes,urdfs}/hot3d`.

Pattern B (shell-in) also works:

```bash
bash run_load_local.sh hot3d
# then, inside the robotic-grounding container:
bash run_retarget_local.sh hot3d
```

---

## 4. Gotchas

- Object meshes are **`.glb` in meters** — no cm scaling (the loader loads them
  with `vertex_scale=1.0`).
- MANO poses are stored as **15 PCA coefficients**; the loader expands them to
  full 45-DOF finger pose. `betas` uses the first 10 values; hand index `0` =
  left, `1` = right; wrist quaternions are **wxyz**.
- Aria world frames are **Z-up**; Quest3 frames are **Y-up** and are rotated to
  Z-up via `QUEST3_WORLD_TO_ZUP`. The loader also applies a per-sequence yaw
  normalization (from the first `headset_trajectory.csv` row) so every scene
  starts with the workspace in the +Y direction.
- The whole scene is lifted so the support surface (table) sits
  `HOT3D_TABLE_HEIGHT_M = 1.0` m above z=0; a table left at z=0 would be filtered
  as the ground plane, leaving no support surface.
- Sequence IDs are `P####_<hash>` (e.g. `P0001_10a27bf7`) and match across the
  Aria/Assets URL files.

## Citation
If you use this dataset, please cite the original work:

```bibtex
@inproceedings{banerjee2025hot3d,
  title     = {{HOT3D}: Hand and Object Tracking in 3D from Egocentric Multi-View Videos},
  author    = {Banerjee, Prithviraj and Shkodrani, Sindi and Moulon, Pierre and Hampali, Shreyas and Zhang, Fan and Fountain, Jade and Miller, Edward and Basol, Selen and Newcombe, Richard and Wang, Robert and Engel, Jakob Julian and Hodan, Tomas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```
