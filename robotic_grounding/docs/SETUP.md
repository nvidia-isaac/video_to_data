# Hand → Robot retargeting: setup & run

This guide reproduces the hand-object → robot retargeting pipeline on a clean machine,
**downloading every dataset from its original public source** (the academic dataset
portals). You end up with, per dataset: retargeted robot motion (Parquet), reconstructed
support surfaces, and a viser/MP4 visualization.

---

## 1. The pipeline uses two Docker images

```
                   raw dataset  +  MANO models
                              │
                              ▼
        ╔═══════════════════════════════════════════════╗
        ║  IMAGE 1 · v2d_task_library_loader              ║
        ║  stage: load   —   MANO forward-kinematics      ║
        ╚═══════════════════════════════════════════════╝
                              │
                              ▼
                         <ds>_loaded
            Parquet — per-frame MANO joints + object poses
                              │
                              ▼
        ╔═══════════════════════════════════════════════╗
        ║  IMAGE 2 · robotic-grounding                    ║
        ║  segment → urdf → processed → support → vis     ║
        ║  (IK retarget · support surfaces · viz)         ║
        ╚═══════════════════════════════════════════════╝
                              │
                              ▼
                           outputs
          <ds>_processed                retargeted robot motion
          object_assets/urdfs/<ds>      object URDFs
          reconstructed_stage/*.usda    support surfaces
          <ds>_html/                    viser recording + MP4
```
Top → bottom: **IMAGE 1** turns the raw dataset (+ MANO) into `<ds>_loaded`; that file is
handed to **IMAGE 2**, which retargets it to the robot and produces the URDFs, support
surfaces, and visualization. (`segment` runs only for hot3d.) The orchestrator runs the
right image for each stage automatically.

- **Loader image** `v2d_task_library_loader` — reads the raw dataset + MANO models and
  writes `<ds>_loaded` (per-frame MANO joints + object poses). Built from a public
  PyTorch base.
- **Retarget image** `robotic-grounding` — IK-retargets hands to the robot, reconstructs
  support surfaces, and renders the visualization. Built locally on top of the **Isaac Lab
  2.3.1** container base (pinned in `workflow/Dockerfile`).

You drive both from the host with a single script — **`scripts/run_pipeline_docker.py`** —
which builds the right `docker run` for each stage. You never shell into a container.

## 2. Prerequisites (host)
- **Docker** + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html); a recent NVIDIA driver + an NVIDIA GPU.
- **Isaac Lab 2.3.1** — the base the retarget image builds on (pinned in
  `workflow/Dockerfile`). Follow the
  [Isaac Lab installation docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
  to obtain the base container; everything else is built locally.
- **Git LFS**, then pull the **robot** assets:
  ```bash
  cd robotic_grounding
  git lfs pull        # fetches the robot URDFs + meshes (sharpa_wave, g1) — these, and only
                      # these, are stored in the repo via LFS. Object/dataset meshes are NOT
                      # in the repo; you download them per dataset (below).
  ```
- A Python env on the host for the orchestrator (it only builds/launches Docker; no ML deps).

## 3. Build the images
Build **both** images with one command:
```bash
cd robotic_grounding
python scripts/run_pipeline_docker.py --build-only     # builds loader + robotic-grounding
```
Or run the equivalent build commands directly:
```bash
# retarget image (robotic-grounding) — from robotic_grounding/:
./workflow/run.sh build                                   # -> robotic-grounding:latest

# loader image (v2d_task_library_loader) — from reconstruction/:
pip install -e modules/v2d_common -e modules/v2d_docker -e modules/v2d_task_library_loader/docker
python -m v2d.task_library_loader.docker.build            # -> v2d_task_library_loader
```
(To build-and-run in one shot, add `--build` to a normal run (§7) — it builds only the
image(s) that run's stages need.)

While iterating on `reconstruction/` loader code you don't need to rebuild the loader image —
pass `--dev` to a run to mount your live code into the container; rebuild once when you're done.

## 4. Directory layout
Everything — MANO and every dataset — lives under one root you choose (`--hmd <HMD>`,
e.g. `~/datasets/human_motion_data`). **[D]** = you download/provide it; **[G]** = the
pipeline generates it:
```
<HMD>/
├── mano/models/                          # [D] MANO hand models (§5)
│   ├── MANO_LEFT.pkl
│   └── MANO_RIGHT.pkl
└── <dataset>/                            # one per dataset: taco, hot3d, arctic, grab, …
    ├── dataset/                          # [D] raw data, exactly as downloaded from source (§6)
    ├── object_assets/
    │   ├── meshes/<dataset>/             # [D] object meshes*  +  [G] visual STLs
    │   └── urdfs/<dataset>/              # [G] object URDFs   (arctic: [D] — from ArtiGrasp)
    ├── <dataset>_loaded/                 # [G] load     — MANO FK + object poses (Parquet)
    ├── <dataset>_loaded_segmented/       # [G] segment  — atomic clips (hot3d only)
    ├── <dataset>_processed/              # [G] processed — robot joints via IK
    ├── reconstructed_stage/              # [G] support  — support-surface .usda
    └── <dataset>_html/                   # [G] vis      — viser recordings + MP4
```
**You download only** `mano/models/` and each dataset's `dataset/` (plus, for **arctic**,
its articulated URDFs from ArtiGrasp). **The pipeline generates everything else.**

\* Object meshes: for **taco / hot3d / arctic** you place them under
`object_assets/meshes/<dataset>/` (from the dataset's source — see its doc). For
**grab / h2o / dexycb** the meshes ship inside `dataset/`, so `object_assets/meshes/`
holds only the generated visual STLs. (See §8 for the two object-asset conventions.)

## 5. MANO hand models (LOAD only, license-gated)
1. Register + accept the license at **https://mano.is.tue.mpg.de/** and download `mano_v1_2.zip`.
2. Extract just the two `.pkl` into `<HMD>/mano/models/` (the manotorch layout):
   ```
   <HMD>/mano/models/MANO_LEFT.pkl
   <HMD>/mano/models/MANO_RIGHT.pkl
   ```
3. Pass `--mano-dir <HMD>/mano` at runtime — MANO is never committed and is only used by LOAD.

## 6. Download a dataset (from its original source)
Pick a dataset and follow its from-scratch doc — each lists the **original portal/registration,
the exact files, and how to lay them out** under `<HMD>/<dataset>/` (per §4):

> **No fetch/sync script is involved.** You download each dataset directly from its public source
> and arrange the files to match the directory hierarchy in §4 (and its per-dataset doc); the
> pipeline reads them straight from there. There is no automated fetcher to run.

| Dataset | Original source | Doc |
|---|---|---|
| taco | github.com/leolyliu/TACO-Instructions (Dropbox) | [`TACO_SETUP.md`](TACO_SETUP.md) |
| hot3d | projectaria.com/datasets/hot3d | [`HOT3D_SETUP.md`](HOT3D_SETUP.md) |
| arctic | arctic.is.tue.mpg.de (+ ArtiGrasp URDFs) | [`ARCTIC_SETUP.md`](ARCTIC_SETUP.md) |
| grab | grab.is.tue.mpg.de | [`GRAB_SETUP.md`](GRAB_SETUP.md) |
| dexycb | dex-ycb.github.io (Google Drive) | [`DEXYCB_SETUP.md`](DEXYCB_SETUP.md) |
| h2o | h2odataset.ethz.ch | [`H2O_SETUP.md`](H2O_SETUP.md) |

## 7. Run the pipeline
```bash
cd robotic_grounding
python scripts/run_pipeline_docker.py <dataset> \
    --hmd <HMD> --mano-dir <MANO_DIR> --max-sequences 2     # small smoke test
```
- `--stages auto` (default) picks the recommended stages for the dataset and prints why.
  Override with `--stages load,urdf,processed,support,vis` (or a subset).
- `--build` builds the needed image(s) first; `--max-sequences N` samples N sequences.
- Data filtering is opt-in: `--assess --reject --filter-penetration --penetration-threshold 2.0`.
- Inspect a result: `python visualizer/serve.py --port 8081 --html-dir <HMD>/<dataset>/<dataset>_html`.

## 7b. Run the RL example sequences
To reproduce the specific example sequences, run the ready-made script.
It uses a **self-contained workspace** `<HMD>/example_sequences/` so the example inputs+outputs
stay isolated from your main datasets — download the relevant sequences (arctic, hot3d, taco)
per §6 but lay them out under `<HMD>/example_sequences/<ds>/` (per §4), then:
```bash
cd robotic_grounding
HMD=<HMD> ./run_example_sequences.sh     # load -> [segment] -> [urdf] -> processed -> support -> vis
```
It runs the pipeline only on the listed sequences (ARCTIC `s01_mixer_use_01`, `s07_box_grab_01`,
`s01_espressomachine_use_01`; HOT3D `P0002_59a84a3a` → `seg025`; TACO
`empty__cup__bowl_20231006_280`, `skim_off__spoon__pan_20230926_011`) and writes RL-ready
motion parquets to
`<HMD>/example_sequences/<ds>/<ds>_processed/sequence_id=.../robot_name=sharpa_wave/`.

See **[EXAMPLE_SEQUENCES.md](EXAMPLE_SEQUENCES.md)** for the full sequence list, the workspace
layout, knobs (`ONLY`, `DRY_RUN`, `EXAMPLE_DIR`, `MANO_DIR`), and how to feed the outputs to RL.

## 8. Object assets & URDFs — two conventions
- **`object_assets` datasets** (taco, hot3d, arctic): object meshes live under
  `<HMD>/<ds>/object_assets/meshes/<ds>/`; the orchestrator bind-mounts them into the
  container, so generated STLs/URDFs land there too.
- **runtime-mesh datasets** (grab, h2o, dexycb): object meshes ship inside the dataset
  download; the loader reads them from `dataset/` directly. The generated STLs/URDFs still
  go to `<HMD>/<ds>/object_assets/` (mounted in at runtime — never the repo).
- **URDFs**: the `urdf` stage runs **by default** for every rigid dataset — it generates the
  object URDFs from the meshes into `object_assets/urdfs/<ds>/` (used downstream for sim/RL).
  The one exception is **arctic**, whose objects are *articulated*: their URDFs can't be
  generated, are required at LOAD (MuJoCo FK), and are downloaded (see the arctic doc).

## 9. Gotchas
- Use **`python`**, never `python3`, inside the containers (it's the Isaac wrapper).
- Robot URDFs/meshes are **Git-LFS** — run `git lfs pull` or they'll be broken pointers.
- Editing a `*_loader.py` under `reconstruction/` requires rebuilding the loader image
  (or pass `--dev` to mount your live code while iterating). Edits under `robotic_grounding/`
  take effect immediately (the orchestrator mounts the repo live).
- `taco` object meshes are in **cm** (scaled ×0.01); `hot3d` needs the `segment` stage
  (auto-inserted); `arctic` objects are articulated.
- `dummy_agent` (optional RL playback) is RAM-bound (~10 GB/process) and must run headless.
