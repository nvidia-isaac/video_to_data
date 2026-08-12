# ARCTIC end-to-end from scratch (original source)

ARCTIC's objects are **articulated** (two parts, `bottom`+`top`, joined by a revolute
`rotation` joint), so the loader uses forward kinematics over **articulated
URDFs**. Those URDFs are **REQUIRED at LOAD** and are **shipped, not generated** — the
ARCTIC release has no URDFs, so they come from the published **ArtiGrasp** project. ARCTIC
is an **object_assets** dataset (meshes + URDFs under `<HMD>/arctic/object_assets`, not the
dataset download). There is **no `urdf` generation stage** and no segmentation.

```
ARCTIC raw_seqs (.mano/.object.npy) ┐
ArtiGrasp rsc/arctic (URDFs+meshes) ┘─[arrange]→ <HMD>/arctic/{dataset,object_assets}
   ─[LOAD, art-FK]→ arctic_loaded ─[PROCESSED→SUPPORT→VIS]→ outputs
```
`HMD=~/datasets/human_motion_data`

---

## 1. Download

You need two things: the **raw motion sequences** (`raw_seqs`, ~215 MB — per-frame MANO +
object motion; the **only** ARCTIC file the pipeline uses) and the **articulated URDFs +
meshes** (from ArtiGrasp, not ARCTIC — next subsection). **Do NOT download ARCTIC's image
archives** (full 2K ~649 GB, cropped ~116 GB, features ~14 GB) or the splits/meta/backgrounds;
they are unused. Place files as shown in §2 — the pipeline only needs them on disk.

**Motion (`raw_seqs`) — via the ARCTIC repo's download scripts (credentialed).**
First register at **https://arctic.is.tue.mpg.de** (plus MANO/SMPL-X accounts) to obtain a
username/password. ARCTIC has **no post-login "Download page"** — the portal redirects to the
public GitHub repo, whose `bash/` scripts are the actual download path. They POST your
credentials to MPI's `download.php` (a plain `curl` fails — the 302 drops the POST session).
Fetch just the small archives (which include `raw_seqs`), **not** the images:
```bash
# Host, OUTSIDE the container — this is third-party ARCTIC tooling in its own venv.
git clone --no-checkout --depth 1 https://github.com/zc-alexfan/arctic.git ~/arctic
cd ~/arctic
git sparse-checkout init --cone
git sparse-checkout set bash docs scripts_data
git checkout master
python3 -m venv ~/venvs/arctic && source ~/venvs/arctic/bin/activate
pip install setuptools loguru requests tqdm
export ARCTIC_USERNAME=<email> ARCTIC_PASSWORD=<password>
chmod +x ./bash/*.sh
./bash/download_misc.sh            # fetches raw_seqs (+ small misc) — NOT the image archives
python scripts_data/checksum.py    # verifies each zip is real, not an HTML login/error page
python scripts_data/unzip_download.py
mv unpack data
```
Raw sequences land at
`~/arctic/data/arctic_data/data/raw_seqs/s01..s10/<obj>_<action>_<seq>.{mano,object}.npy`;
§2 copies them into `$HMD/arctic/dataset/`.

> **Why not download `raw_seqs.zip` by hand from the portal?** You can't — there is no
> post-login Download page: `arctic.is.tue.mpg.de` redirects to the GitHub repo, which does
> **not** contain `raw_seqs.zip`. The `bash/` scripts above are the only working path, and
> their `checksum.py` step catches a bad-credential HTML response instead of silently writing
> it to disk as a corrupt zip.

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
# raw_seqs were fetched by §1's ./bash/download_misc.sh (there is no downloadable raw_seqs.zip):
mkdir -p $HMD/arctic/dataset
cp -r ~/arctic/data/arctic_data/data/raw_seqs/* $HMD/arctic/dataset/   # s01..s10/<obj>_<action>_<seq>.{mano,object}.npy
```

**Object assets.** ArtiGrasp lays out each object as
`rsc/arctic/<obj>/{<obj>.urdf, {top,bottom}_watertight_tiny.{obj,stl}}` with the URDF
referencing meshes by **bare filename** (co-located). The loader wants the split layout
`object_assets/{urdfs/arctic/<obj>.urdf, meshes/arctic/<obj>/…}`, which separates the URDF from
its meshes, so each URDF needs **both** mesh-path rewrites below — one for plain URDF parsers,
one for MuJoCo — pointing at `../../meshes/arctic/<obj>` (relative to `urdfs/arctic/<obj>.urdf`):
```bash
ARTI=~/artigrasp/rsc/arctic
DEST=$HMD/arctic/object_assets
# array (not a space-string) so this works in zsh too — zsh does NOT word-split an
# unquoted scalar, so `for obj in $OBJECTS` would iterate once over the whole string.
OBJECTS=(box capsulemachine espressomachine ketchup laptop microwave mixer notebook phone waffleiron)
mkdir -p "$DEST/urdfs/arctic"
for obj in "${OBJECTS[@]}"; do
  mkdir -p "$DEST/meshes/arctic/$obj"
  cp "$ARTI/$obj"/{top,bottom}_watertight_tiny.obj "$ARTI/$obj"/{top,bottom}_watertight_tiny.stl \
     "$DEST/meshes/arctic/$obj/"
  # 1. prefix every bare mesh filename with "../../meshes/arctic/<obj>/"
  # 2. inject <mujoco><compiler meshdir="../../meshes/arctic/<obj>"/></mujoco> after <robot ...>
  python - "$ARTI/$obj/$obj.urdf" "$DEST/urdfs/arctic/$obj.urdf" "$obj" <<'PY'
import re, sys
src, dst, obj = sys.argv[1:4]
xml = open(src).read()  # ArtiGrasp original (bare mesh filenames)
rel = f'../../meshes/arctic/{obj}'
xml = xml.replace('<mesh filename="', f'<mesh filename="{rel}/')
inj = f'<mujoco><compiler meshdir="{rel}"/></mujoco>'
open(dst, "w").write(re.sub(r'(<robot[^>]*>)', r'\1' + inj, xml, count=1))
PY
done
```
> **Why both, and why they don't double up.** Generic URDF parsers (urdfpy/PyBullet/Isaac) resolve
> `<mesh filename>` relative to the URDF, so they need the `../../meshes/arctic/<obj>/` prefix.
> MuJoCo ignores that prefix — for URDF input its compiler defaults to `strippath="true"`, keeping
> only the mesh **basename** and resolving it against `meshdir`, which is unset in a stock URDF, so
> without the `<mujoco>` block it looks next to the URDF in `urdfs/arctic/` and fails. Because
> MuJoCo strips the directory first, the prefix and `meshdir` never concatenate.

Resulting structure (objects: the 10 above; `scissor` is excluded by the loader):
```
$HMD/arctic/
├── dataset/s01..s10/<obj>_<action>_<seq>.{mano,object}.npy        # ARCTIC motion
└── object_assets/
    ├── urdfs/arctic/<obj>.urdf                                    # ArtiGrasp + rewritten mesh paths
    └── meshes/arctic/<obj>/{top,bottom}_watertight_tiny.{obj,stl} # ArtiGrasp
```

## 3. Run the pipeline (host orchestrator, Pattern A)

> **Complete the one-time host setup first** ([`SETUP.md`](SETUP.md) §2–3). Without the host
> `v2d` packages the orchestrator aborts immediately with
> `ModuleNotFoundError: No module named 'v2d'` (`command failed (exit 1); aborting.`):
> ```bash
> python3 -m venv ~/venvs/v2d
> source ~/venvs/v2d/bin/activate
> cd <repo>/reconstruction
> pip install -e modules/v2d_common -e modules/v2d_docker -e modules/v2d_task_library_loader/docker
> cd ../robotic_grounding
> python scripts/run_pipeline_docker.py --build-only    # builds BOTH Docker images
> ```

This command is the **actual pipeline execution** (LOAD → PROCESSED → SUPPORT → VIS), not a
data-layout check:
```bash
cd <repo>/robotic_grounding
# --sequence-pattern pins the exact sequence the downstream RL examples reference
# (arctic/arctic_processed/dataset_s07_box_grab_01/sharpa_wave). --max-sequences N would
# instead sample the first N sorted sequences (all s01*), so the s07 examples would not resolve.
python scripts/run_pipeline_docker.py arctic \
    --hmd $HMD --mano-dir $HMD/mano --sequence-pattern s07_box_grab_01
```
- The default (`--stages auto`) selects **`load,processed,support,vis,assess`** for arctic
  (the trailing `assess` is a report-only quality pass → `arctic_quality.csv`).
- **No `urdf` stage**: the objects are articulated, so their URDFs can't be generated —
  they are REQUIRED at LOAD (FK) and SHIPPED via ArtiGrasp. Asking for the `urdf`
  stage on arctic is a no-op: `generate_rigid_urdfs` refuses articulated datasets.
- **No `segment`**: ARCTIC sequences are already clip-sized.
- `load` runs articulated FK (`bottom`/`top` via the `rotation` joint) → `arctic_loaded`.
- Outputs under `$HMD/arctic/`: `arctic_loaded`, `arctic_processed`, `reconstructed_stage`,
  `arctic_html`.
- Pattern B (shell-in) also works: `bash run_load_local.sh arctic`, then inside the
  container `bash run_retarget_local.sh arctic`.

## 4. Gotchas
- ARCTIC ships **no URDFs** — they MUST come from ArtiGrasp (or be hand-crafted). The
  dataset cannot be loaded from the ARCTIC download alone.
- ArtiGrasp's URDFs reference meshes by **bare filename** (meshes sit next to the URDF); the
  split `urdfs/` + `meshes/` layout breaks that, so §2 applies **two** rewrites: the
  `../../meshes/arctic/<obj>/` prefix on each `<mesh filename>` (for URDF parsers) **and** a
  `<mujoco><compiler meshdir="../../meshes/arctic/<obj>"/></mujoco>` block (for MuJoCo, which
  strips the path from URDF mesh filenames and uses `meshdir` instead). Doing only one of the
  two leaves the other consumer unable to find the meshes.
- Loader contract: each object's URDF must expose body **`top`** and joint **`rotation`**
  (ArtiGrasp's URDFs match); object part names are `bottom` (root) and `top`.
- Support reconstruction uses only the **root body** for articulated objects.
- ARCTIC has **no post-login Download page** — fetch `raw_seqs` with the ARCTIC repo's
  `bash/download_misc.sh` (§1), which POSTs your creds to MPI `download.php`. A plain `curl`
  fails because the 302 drops the POST session; `scripts_data/checksum.py` catches an HTML
  login/error page before it's mistaken for a valid zip.

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
