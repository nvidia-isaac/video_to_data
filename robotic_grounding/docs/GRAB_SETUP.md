# GRAB end-to-end from scratch (original source)

GRAB (ECCV 2020) is a runtime-mesh dataset: the object meshes ship in the download (no
separate URDF/mesh hunt), so rigid object URDFs are generated from them by the `urdf` stage
(run by default). It is MANO-native (betas are zeroed, so no SMPL-X body model is needed) and
has no images. No segmentation. Distributed via MPI login.

```
GRAB grab__sN.zip (.npz motion) + tools__object_meshes.zip (.ply) ─[place in dataset dir]→
  <HMD>/grab/dataset/{grab/sN, tools/object_meshes/contact_meshes}  ─[LOAD, MANO FK]→ grab_loaded
  ─[PROCESSED→SUPPORT→VIS]→ outputs
```
`HMD=~/datasets/human_motion_data`

---

## 1. Download (MPI login)

> **Downloading by hand?** On the Download page, get **`grab__s1.zip`** (Subject 1 motion,
> ~0.6 GB; add `grab__s2…s10` for the full set) and **`tools__object_meshes.zip`** (~32 MB
> object meshes — this item may require accepting an extra access grant on the site).
> **Skip** Rendered Videos, Subject Shape Templates, Virtual Markers, and the SMPL-X→MANO
> mapping (the loader uses MANO fullpose directly). Place files as shown in §2 — the
> pipeline only needs them on disk, not any particular download method.

Source: **https://grab.is.tue.mpg.de** (register, verify email, accept the license on the
Download page). It is a separate MPI account from MANO/SMPL-X, but you can reuse the same
email. For the pipeline we need only **two** archives (skip Rendered Videos, Subject Shape
Templates, Virtual Markers, and the SMPL-X→MANO mapping):

| Portal item | `sfile` | Size | Holds |
|---|---|---|---|
| **Subject 1** (under "GRAB dataset") | `grab__s1.zip` | ~0.6 GB | `s1/<obj>_<action>_<take>.npz` (motion) |
| **Objects** | `tools__object_meshes.zip` | 32 MB | `object_meshes/<obj>.ply` (57 meshes) |

MPI's `download.php` authenticates by POSTing username/password to the file URL (plain `curl`
fails — the 302 drops the session — so use `requests` with TLS verify on):
```python
# download_grab.py — run with GRAB_USERNAME / GRAB_PASSWORD in the env
import os, requests
U, P = os.environ["GRAB_USERNAME"], os.environ["GRAB_PASSWORD"]
for sfile in ["grab__s1.zip", "tools__object_meshes.zip"]:   # add grab__s2..s10 for more subjects
    url = f"https://download.is.tue.mpg.de/download.php?domain=grab&resume=1&sfile={sfile}"
    r = requests.post(url, data={"username": U, "password": P}, stream=True)  # verify ON
    assert r.status_code == 200 and r.headers["content-type"] != "text/html", (sfile, r.status_code)
    with open(os.path.expanduser(f"~/grab_download/{sfile}"), "wb") as f:
        for c in r.iter_content(1 << 20): f.write(c)
```

> **Accept the GRAB license in a browser before running the script.** Log in on the GRAB
> Download page and accept the license there first. Until you do, the credentialed POST above
> returns an empty-body `403` for every file — the login succeeds, but downloads are license-gated
> and the acceptance does not carry over to a scripted session. Accept in the browser, then run
> the script.
>
> If a single file still returns a `403` while the others download, open that file on the Download
> page and accept it (or just download that one file in the browser), then place it in
> `~/grab_download/`.

## 2. Set up the dataset directory

> **No fetch/sync script needed** — download the data directly from its original source (above);
> there's no automated fetcher to run. The pipeline only needs the files arranged in the
> directory hierarchy shown below — match this layout and the loader finds everything.

The loader globs `<HMD>/grab/dataset/grab/sN/*.npz` and resolves each `.npz`'s
`object.object_mesh` (a path like `tools/object_meshes/contact_meshes/airplane.ply`) relative
to `<HMD>/grab/dataset`. The subject zip extracts to `sN/`, the object zip to
`object_meshes/` — re-home both into that layout:
```bash
DST=$HMD/grab/dataset
mkdir -p "$DST/grab" "$DST/tools/object_meshes/contact_meshes"
unzip -q ~/grab_download/grab__s1.zip -d /tmp/grab_s              # -> s1/*.npz
cp -r /tmp/grab_s/s1 "$DST/grab/"                                 # -> grab/s1/
unzip -q ~/grab_download/tools__object_meshes.zip -d /tmp/grab_m   # -> object_meshes/*.ply
cp /tmp/grab_m/object_meshes/*.ply "$DST/tools/object_meshes/contact_meshes/"
```
Resulting structure:
```
$HMD/grab/dataset/
├── grab/s1/<obj>_<action>_<take>.npz                     # 198 sequences (Subject 1)
└── tools/object_meshes/contact_meshes/<obj>.ply          # 57 object meshes (meters)
```
**No `object_assets/`** — GRAB is runtime-mesh: meshes are read from the dataset itself.

## 3. Run the pipeline (host orchestrator, Pattern A)
```bash
cd <repo>/robotic_grounding
python scripts/run_pipeline_docker.py grab \
    --hmd $HMD --mano-dir $HMD/mano \
    --max-sequences 2          # 'auto' stages: load,urdf,processed,support,vis
```
- Orchestrator treats grab as **runtime-mesh** (`RUNTIME_MESH_DATASETS`): LOAD omits
  `--object_assets_dir`; the retarget stages mount the data root at `/data/human_motion_data`
  so the loader-baked dataset-relative mesh paths resolve.
- **No `segment`** (GRAB clips are already atomic, so `--max-sequences 2` = 2 sequences with no
  fan-out). `urdf` runs by default — `generate_rigid_urdfs.py` builds the rigid object URDFs
  from the `.ply` into `object_assets/urdfs/grab/` (used downstream for sim/RL).
- Outputs under `$HMD/grab/`: `grab_loaded`, `grab_processed`, `reconstructed_stage`, `grab_html`.

Pattern B (shell into the persistent container): `bash run_load_local.sh grab` on the host,
then inside the container `bash run_retarget_local.sh grab`.

## 4. Gotchas
- GRAB's stored world frame is already **Z-up** — the loader applies **no** axis remap
  (`Y_UP_TO_Z_UP` is the identity matrix). An earlier R_x(90°) "Y-up→Z-up" rotation was wrong
  and tipped the whole scene 90° onto its side.
- Hand params are MANO **PCA** coeffs, but the loader reads the already-expanded 45-DOF
  `fullpose`; betas are **zeroed** (subject identity is in `vtemp`, which the IK retarget does
  not need) — so MANO `.pkl` is the only body model required.
- Object meshes are **`.ply`** in **meters** (no scaling), read from
  `dataset/tools/object_meshes/contact_meshes/`.
- MPI `download.php`: POST creds with `requests` (TLS verify on); plain `curl` fails because the
  302 drops the POST session. If you get `403`, accept the GRAB license on the website first.

## Citation
If you use this dataset, please cite the original work:

```bibtex
@inproceedings{GRAB:2020,
  title     = {{GRAB}: A Dataset of Whole-Body Human Grasping of Objects},
  author    = {Taheri, Omid and Ghorbani, Nima and Black, Michael J. and Tzionas, Dimitrios},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020},
  url       = {https://grab.is.tue.mpg.de}
}
```
