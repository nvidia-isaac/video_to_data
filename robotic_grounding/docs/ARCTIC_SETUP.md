# ARCTIC end-to-end from scratch (original source)

ARCTIC's objects are **articulated** (two parts, `bottom`+`top`, joined by a revolute
`rotation` joint), so the loader uses **MuJoCo** forward kinematics over **articulated
URDFs**. Those URDFs are **REQUIRED at LOAD** and are **shipped, not generated** — the
ARCTIC release has no URDFs, so they come from the published **ArtiGrasp** project. ARCTIC
is an **object_assets** dataset (meshes + URDFs under `<HMD>/arctic/object_assets`, not the
dataset download). There is **no `urdf` generation stage** and no segmentation.

```
ARCTIC raw_seqs (.mano/.object.npy) ┐
ArtiGrasp rsc/arctic (URDFs+meshes) ┘─[arrange]→ <HMD>/arctic/{dataset,object_assets}
   ─[LOAD, MuJoCo art-FK]→ arctic_loaded ─[PROCESSED→SUPPORT→VIS]→ outputs
```
`HMD=~/datasets/human_motion_data`

---

## 1. Download

> **Downloading by hand?** From the portal's Download page you only need **`raw_seqs.zip`**
> (~215 MB, "Raw GT sequences in world coordinate") — the per-frame MANO + object motion.
> **Do NOT download the image archives** (full 2K images ~649 GB, cropped ~116 GB, image
> features ~14 GB) or the splits/meta/backgrounds; they are unused. Object URDFs + meshes
> come from ArtiGrasp (next subsection), not the portal. Place files as shown in §2 — the
> pipeline only needs them on disk, not any particular download method.

**Motion (MPI portal — credentialed).** Register at **https://arctic.is.tue.mpg.de**
(plus MANO/SMPL-X accounts). MPI's `download.php` authenticates by **POSTing
username/password to the file URL** — plain `curl` fails because the 302 drops the session,
so use `requests` (TLS verify ON). Only `raw_seqs` (the per-frame MANO + object motion) is
needed:
```python
# download_arctic.py — run with ARCTIC_USERNAME / ARCTIC_PASSWORD in the env
import os, requests
U, P = os.environ["ARCTIC_USERNAME"], os.environ["ARCTIC_PASSWORD"]
REL = "arctic_release/<release_hash>/v1_0/data"   # from the portal's Download page
for fn in ["raw_seqs.zip"]:                       # raw_seqs = motion (MANO + object)
    url = f"https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile={REL}/{fn}"
    r = requests.post(url, data={"username": U, "password": P}, stream=True)  # verify ON
    assert r.status_code == 200 and r.headers["content-type"] != "text/html", (fn, r.status_code)
    with open(os.path.expanduser(f"~/arctic_download/{fn}"), "wb") as f:
        for c in r.iter_content(1 << 20): f.write(c)
```

**Articulated URDFs + meshes (ArtiGrasp — public, no login).** The published ArtiGrasp
project ships the articulated URDFs **and** the matching decimated meshes
(`{top,bottom}_watertight_tiny.{obj,stl}`) the loader reads:
```bash
git clone --no-checkout --depth 1 --filter=blob:none https://github.com/zdchan/artigrasp.git ~/artigrasp
cd ~/artigrasp && git sparse-checkout set rsc/arctic && git checkout
```
> **If the sparse clone doesn't work** (old git, no `--filter` support, firewalled, etc.),
> just grab the whole repo by hand — it's public, no login. Either a full clone
> (`git clone https://github.com/zdchan/artigrasp.git ~/artigrasp`) or, in a browser, open
> **https://github.com/zdchan/artigrasp** → **Code ▸ Download ZIP**, unzip, and use only the
> **`rsc/arctic/`** folder. You only need, per object, `rsc/arctic/<obj>/<obj>.urdf` and
> `rsc/arctic/<obj>/{top,bottom}_watertight_tiny.{obj,stl}` — everything else in the repo is
> unused. (You can also download those per-object files individually from the GitHub web UI.)

## 2. Set up the dataset directory

> **No fetch/sync script needed** — download the data directly from its original source (above);
> there's no automated fetcher to run. The pipeline only needs the files arranged in the
> directory hierarchy shown below — match this layout and the loader finds everything.

**Motion.** The loader globs `<HMD>/arctic/dataset/*/*.mano.npy`, pairs each `.object.npy`,
and **excludes `scissor`**:
```bash
unzip -q ~/arctic_download/raw_seqs.zip -d /tmp/arctic_raw
mkdir -p $HMD/arctic/dataset
cp -r /tmp/arctic_raw/raw_seqs/* $HMD/arctic/dataset/   # s01..s10/<obj>_<action>_<seq>.{mano,object}.npy
```

**Object assets.** ArtiGrasp lays out each object as
`rsc/arctic/<obj>/{<obj>.urdf, {top,bottom}_watertight_tiny.{obj,stl}}` with the URDF
referencing meshes by **bare filename** (co-located). The loader wants the split layout
`object_assets/{urdfs/arctic/<obj>.urdf, meshes/arctic/<obj>/…}`. Because `mujoco<3.7`
resolves a mesh `filename` to its **basename in the URDF's directory** (it ignores any
relative/absolute path prefix), don't rewrite the filenames — inject a
`<mujoco><compiler meshdir=…/></mujoco>` directive so the relocated meshes resolve:
```bash
ARTI=~/artigrasp/rsc/arctic
DEST=$HMD/arctic/object_assets
OBJECTS="box capsulemachine espressomachine ketchup laptop microwave mixer notebook phone waffleiron"
mkdir -p "$DEST/urdfs/arctic"
for obj in $OBJECTS; do
  mkdir -p "$DEST/meshes/arctic/$obj"
  cp "$ARTI/$obj"/{top,bottom}_watertight_tiny.obj "$ARTI/$obj"/{top,bottom}_watertight_tiny.stl \
     "$DEST/meshes/arctic/$obj/"
  # inject <mujoco><compiler meshdir="../../meshes/arctic/<obj>"/></mujoco> after <robot ...>
  python - "$ARTI/$obj/$obj.urdf" "$DEST/urdfs/arctic/$obj.urdf" "$obj" <<'PY'
import re, sys
src, dst, obj = sys.argv[1:4]
xml = open(src).read()  # ArtiGrasp original (bare mesh filenames)
inj = f'<mujoco><compiler meshdir="../../meshes/arctic/{obj}"/></mujoco>'
open(dst, "w").write(re.sub(r'(<robot[^>]*>)', r'\1'+inj, xml, count=1))
PY
done
```

Resulting structure (objects: the 10 above; `scissor` is excluded by the loader):
```
$HMD/arctic/
├── dataset/s01..s10/<obj>_<action>_<seq>.{mano,object}.npy        # ARCTIC motion
└── object_assets/
    ├── urdfs/arctic/<obj>.urdf                                    # ArtiGrasp + meshdir directive
    └── meshes/arctic/<obj>/{top,bottom}_watertight_tiny.{obj,stl} # ArtiGrasp
```

## 3. Run the pipeline (host orchestrator, Pattern A)
```bash
cd <repo>/robotic_grounding
python scripts/run_pipeline_docker.py arctic \
    --hmd $HMD --mano-dir $HMD/mano --max-sequences 2
```
- The default (`--stages auto`) selects **`load,processed,support,vis`** for arctic.
- **No `urdf` stage**: the objects are articulated, so their URDFs can't be generated —
  they are REQUIRED at LOAD (MuJoCo FK) and SHIPPED via ArtiGrasp. Asking for the `urdf`
  stage on arctic is a no-op: `generate_rigid_urdfs` refuses articulated datasets.
- **No `segment`**: ARCTIC sequences are already clip-sized.
- `load` runs MuJoCo articulated FK (`bottom`/`top` via the `rotation` joint) → `arctic_loaded`.
- Outputs under `$HMD/arctic/`: `arctic_loaded`, `arctic_processed`, `reconstructed_stage`,
  `arctic_html`.
- Pattern B (shell-in) also works: `bash run_load_local.sh arctic`, then inside the
  container `bash run_retarget_local.sh arctic`.

## 4. Gotchas
- ARCTIC ships **no URDFs** — they MUST come from ArtiGrasp (or be hand-crafted). The
  dataset cannot be loaded from the ARCTIC download alone.
- `mujoco<3.7` resolves a mesh `filename` to its **basename in the URDF's directory** —
  hence the `<compiler meshdir>` directive (3.7+ changed this; the loader image pins
  `mujoco<3.7`).
- Loader contract: each object's URDF must expose body **`top`** and joint **`rotation`**
  (ArtiGrasp's URDFs match); object part names are `bottom` (root) and `top`.
- Support reconstruction uses only the **root body** for articulated objects.
- MPI `download.php`: POST creds with `requests` (TLS verify ON); plain `curl` fails
  because the 302 drops the POST session.

## Citation
If you use this dataset, please cite the original work:

```bibtex
@inproceedings{fan2023arctic,
  title     = {{ARCTIC}: A Dataset for Dexterous Bimanual Hand-Object Manipulation},
  author    = {Fan, Zicong and Taheri, Omid and Tzionas, Dimitrios and Kocabas, Muhammed and Kaufmann, Manuel and Black, Michael J. and Hilliges, Otmar},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {12943--12954},
  year      = {2023}
}
```
