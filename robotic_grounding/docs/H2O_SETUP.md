# H2O end-to-end from scratch (original source)

H2O (Hand-Two-Object, ICCV 2021, ETH Zürich) is a runtime-mesh dataset: the object meshes ship
in the download and rigid object URDFs are generated from them by the `urdf` stage (run by
default). MANO is 48-DOF axis-angle (3 global + 45 finger, `flat_hand_mean=True`). Takes are
continuous **single-object** interactions, so there is **no segmentation**. Requires
registration (HTTP-basic-auth creds emailed by the site).

```
H2O subjectN_pose_v1_1.tar.gz + object.zip + label_split.zip ─[place in dataset dir]→
  <HMD>/h2o/dataset/  ─[LOAD auto-extracts archives, MANO FK]→ h2o_loaded
  ─[PROCESSED→SUPPORT→VIS]→ outputs
```
`HMD=~/datasets/human_motion_data`

---

## 1. Register + download

> **Downloading by hand?** Get **`object.zip`** (~5.6 MB meshes), **`label_split.zip`**
> (~0.3 MB), and **`subject1_pose_v1_1.tar.gz`** (~78 MB; add `subject{2,3,4}_pose_v1_1.tar.gz`
> for the full set). **Do NOT download `manolabel_v1.1.tar.gz`** (~473 MB — the pose archives
> already contain `hand_pose_mano/`) **or any RGB-D image archives** (unused). Place files as
> shown in §2 — the pipeline only needs them on disk, not any particular download method.

1. Open **https://h2odataset.ethz.ch**, fill the registration form (name, email,
   affiliation) and click **"I Agree"**. The site then **emails you a username +
   password** — these are HTTP-basic-auth credentials, **valid for 7 days**, so download
   promptly.
2. Download the **pose** set (poses + object meshes, no RGB-D — the light option) via plain
   HTTP basic auth against `https://h2odataset.ethz.ch/data/dataset/<file>`:
   ```bash
   U='<user>'; P='<pass>'; base='https://h2odataset.ethz.ch/data/dataset'
   mkdir -p ~/h2o_download
   for f in object.zip label_split.zip subject1_pose_v1_1.tar.gz; do   # add subject{2,3,4} for the full set
     curl -sL -C - --retry 5 -u "$U:$P" "$base/$f" -o ~/h2o_download/"$f"
   done
   ```
   Sizes: `object.zip` 5.6 MB, `label_split.zip` 0.3 MB, each `subjectN_pose` ~78 MB.
   > **Do not** download `manolabel_v1.1.tar.gz` (473 MB): the `subjectN_pose_v1_1.tar.gz`
   > archives already contain `hand_pose_mano/`, which is all the loader reads — and the
   > loader's idempotent-extraction check never matches `manolabel`, so it would re-extract
   > on every run.

## 2. Set up the dataset directory

> **No fetch/sync script needed** — download the data directly from its original source (above);
> there's no automated fetcher to run. The pipeline only needs the files arranged in the
> directory hierarchy shown below — match this layout and the loader finds everything.

**Simplest: just drop the archives in `<HMD>/h2o/dataset/` — the loader auto-extracts them on
first run** (`_extract_archives_if_needed`, idempotent, under a file lock):
```bash
DST=$HMD/h2o/dataset; mkdir -p "$DST"
mv ~/h2o_download/{subject1_pose_v1_1.tar.gz,object.zip,label_split.zip} "$DST/"
```
Or pre-extract by hand if you prefer:
```bash
tar xzf "$DST"/subject1_pose_v1_1.tar.gz -C "$DST"   # repeat for subject{2,3,4} for the full set
unzip -q "$DST"/object.zip      -d "$DST"            # -> object/
unzip -q "$DST"/label_split.zip -d "$DST"
```
Resulting layout (the loader globs `subjectN/<action>/<take>/cam{0-4}/` and reads, per camera:
`hand_pose_mano/*.txt`, `obj_pose_rt/*.txt`, `cam_pose/*.txt`; meshes from `object/`):
```
$HMD/h2o/dataset/
├── subject{1..4}/<action>/<take>/cam{0..4}/{hand_pose_mano,obj_pose_rt,cam_pose}/*.txt
└── object/<name>/                       # object meshes
```
**No `object_assets/`** — H2O is runtime-mesh: meshes are read from the dataset itself.

## 3. Run the pipeline (host orchestrator, Pattern A)
```bash
cd <repo>/robotic_grounding
python scripts/run_pipeline_docker.py h2o \
    --hmd $HMD --mano-dir $HMD/mano --max-sequences 2     # 'auto' stages: load,urdf,processed,support,vis,assess
```
- Orchestrator treats h2o as **runtime-mesh** (`RUNTIME_MESH_DATASETS`): LOAD omits
  `--object_assets_dir`; retarget stages mount the data root at `/data/human_motion_data` so the
  loader-baked mesh paths resolve. `urdf` runs by default — `generate_rigid_urdfs` builds the
  rigid object URDFs from the dataset meshes into `object_assets/urdfs/h2o/` (used downstream for sim/RL).
- **No `segment` stage.** H2O is not in `SEGMENT_DATASETS`. Each take is a continuous
  *single-object* interaction (one take can grab→place→pour→close, but always the **same one
  rigid object**), processed whole rather than chopped into atomic clips.
- **Support surfaces use a hand-release gate (h2o-only).** Because the object is often held
  steady mid-manipulation, plain stillness would mistake a held pause for a resting surface.
  `reconstruct_support_surfaces.py` auto-enables `require_hand_release` for `--dataset h2o`: a
  still run only yields a support disk if the hand has released the object during it. Knobs:
  `HAND_RELEASE_CONTACT_THRESHOLD_M` (0.02 m), `MIN_RELEASED_STILL_FRAMES` (5) in
  `support_recon.py`. Other datasets are unaffected.
- Outputs under `$HMD/h2o/`: `h2o_loaded`, `h2o_processed`, `reconstructed_stage`, `h2o_html`.

Pattern B (in-container shell): `bash run_load_local.sh h2o`, then inside the container
`bash run_retarget_local.sh h2o`.

## 4. Gotchas
- **Registration required**; the emailed HTTP-basic-auth credentials **expire after 7 days**.
  Download the **pose** set only — it skips the large RGB-D images.
- **Skip `manolabel_v1.1.tar.gz`** — the pose archives already carry `hand_pose_mano/`, and
  leaving manolabel in the dataset dir makes the loader re-extract 473 MB every run.
- Each `hand_pose_mano/*.txt` is 124 floats = 2 hands × 62 (`H2O_PER_HAND_FLOATS`): flag(1),
  trans(3), pose(48 axis-angle = 3 global + 45 finger), shape(10). MANO is axis-angle (not PCA),
  `flat_hand_mean=True` per the official format.
- Hand/object poses are stored **per (moving) camera frame**; the loader composes each frame's
  `cam_pose` extrinsic to output world-frame poses. World = initial head-mounted camera (OpenCV).
- **Head-pitch tilt:** the `cam4`-only pose release is *ego-normalized* — every sequence's
  `cam_pose` starts at identity, so the world frame is the subject's initial head pose, which is
  pitched down toward the table. There is no static-rig gravity reference, so the scene would
  render tilted. The loader corrects this with a fixed pitch `H2O_HEAD_PITCH_DEG` (−45°, composed
  into `H2O_WORLD_TO_ZUP` in `h2o_loader.py`) about the camera-X axis — one constant levels the
  whole dataset (per-take roll averages to ~0 and is not constant-correctable). Tune it in the
  visualizer (re-run `load,processed,vis` with `--dev`); flip the sign if the table tilts the
  wrong way; rebuild the loader image to bake the final value.
- `cam4` is the egocentric view (loader default); any cam view of a take can be used as a sequence.

## Citation
If you use this dataset, please cite the original work:

```bibtex
@inproceedings{Kwon_2021_ICCV,
  title     = {{H2O}: Two Hands Manipulating Objects for First Person Interaction Recognition},
  author    = {Kwon, Taein and Tekin, Bugra and St{\"u}hmer, Jan and Bogo, Federica and Pollefeys, Marc},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {10138--10148},
  year      = {2021}
}
```
