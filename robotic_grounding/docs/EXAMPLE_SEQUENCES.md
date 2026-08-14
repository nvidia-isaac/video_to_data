# Example sequences (`run_example_sequences.sh`)

A one-command reproduction of the **RL example sequences** — a small, fixed set of
hand-object clips across arctic / hot3d / taco. It runs the full retargeting pipeline
(`load → [segment] → [urdf] → processed → support → vis`) on each listed sequence and
writes RL-ready robot-motion parquets (plus a viser/MP4 visualization) you can hand straight
to the RL scripts.

It's a thin wrapper around [`scripts/run_pipeline_docker.py`](../scripts/run_pipeline_docker.py)
(host orchestrator, Pattern A) — each call spins up the right Docker image per stage.

## What it runs

| Dataset | Sequence (`--sequence-pattern`) | What it exercises | Stages |
|---|---|---|---|
| **arctic** | `s01_mixer_use_01` | mixer use — **articulated** object | `load,processed,support,vis` |
| **arctic** | `s07_box_grab_01` | box grab — single rigid object | `load,processed,support,vis` |
| **arctic** | `s01_espressomachine_use_01` | espresso machine use (hand) | `load,processed,support,vis` |
| **hot3d** | `P0002_59a84a3a` → `seg025` | multi-object atomic clip (auto-segmented) | `load,urdf,processed,support,vis` |
| **taco** | `empty__cup__bowl_20231006_280` | rigid tool use | `load,urdf,processed,support,vis` |
| **taco** | `skim_off__spoon__pan_20230926_011` | rigid tool use | `load,urdf,processed,support,vis` |

Notes:
- **arctic** skips `urdf` — its objects are *articulated*, so URDFs are downloaded (from
  ArtiGrasp), not generated. See [`ARCTIC_SETUP.md`](ARCTIC_SETUP.md).
- **hot3d** auto-inserts a `segment` stage that splits the source recording into atomic clips;
  the example targets `seg025` of `P0002_59a84a3a`.
- **taco** patterns are regex over the derived sequence id (the loader maps a triplet
  `"(skim off, spoon, pan)"` → `skim_off__spoon__pan`); the two examples are pinned to a
  specific sequence by appending its capture id (e.g. `_20230926_011`).

## Self-contained workspace

The script runs in an **isolated example workspace** so it never mixes with your main
`<HMD>` datasets:

```
$EXAMPLE_DIR/                 # default: <HMD>/example_sequences
└── <ds>/                     # arctic, hot3d, taco
    ├── dataset/              # [D] raw data you download (input)
    ├── object_assets/        # [D] object meshes (input) + [G] generated STLs/URDFs
    ├── <ds>_loaded/          # [G] load output
    ├── <ds>_loaded_segmented/# [G] segment output (hot3d only)
    ├── <ds>_processed/       # [G] RL-ready motion parquets (output)
    ├── reconstructed_stage/  # [G] support-surface .usda (output)
    └── <ds>_html/            # [G] viser recordings + MP4 (vis output)
```
**[D]** = you download/provide · **[G]** = the pipeline generates. MANO models are shared
from `$MANO_DIR` (default `<HMD>/mano`) — mounted independently of the workspace, so they're
**not** duplicated here.

## Prerequisites

The script does **not** download anything. First obtain MANO and the three datasets from
their original public sources, per the umbrella guide [`SETUP.md`](SETUP.md) and the
per-dataset guides ([`ARCTIC_SETUP.md`](ARCTIC_SETUP.md), [`HOT3D_SETUP.md`](HOT3D_SETUP.md),
[`TACO_SETUP.md`](TACO_SETUP.md)) — but lay each one out under
`$EXAMPLE_DIR/<ds>/` (the workspace above) instead of `<HMD>/<ds>/`. You only need the
specific sequences in the table, not the full datasets.

Also build the two pipeline images once (see [`SETUP.md` §3](SETUP.md#3-build-the-images)):
```bash
python scripts/run_pipeline_docker.py --build-only
```

## Run

From `robotic_grounding/`:
```bash
HMD=~/datasets/human_motion_data ./run_example_sequences.sh
```
Outputs land under `$EXAMPLE_DIR/<ds>/<ds>_processed/sequence_id=.../robot_name=sharpa_wave/`,
and the visualization under `$EXAMPLE_DIR/<ds>/<ds>_html/`. Browse it with:
```bash
python visualizer/serve.py --html-dir "$EXAMPLE_DIR/<ds>/<ds>_html"
```

### Knobs (environment variables)
| Var | Default | Effect |
|---|---|---|
| `HMD` | `~/datasets/human_motion_data` | Your data root; sets the defaults below. |
| `EXAMPLE_DIR` | `$HMD/example_sequences` | The self-contained example workspace (inputs + outputs). |
| `MANO_DIR` | `$HMD/mano` | MANO models directory (shared, mounted separately). |
| `ONLY` | _(unset)_ | Run just one dataset's examples: `arctic`, `hot3d`, or `taco`. |
| `DRY_RUN` | _(unset)_ | Print the `docker run` commands without executing them. |

Examples:
```bash
ONLY=taco    HMD=~/datasets/human_motion_data ./run_example_sequences.sh   # taco examples only
DRY_RUN=1    HMD=~/datasets/human_motion_data ./run_example_sequences.sh   # validate, don't run
EXAMPLE_DIR=/scratch/v2d_examples ./run_example_sequences.sh               # custom workspace
```

## Next: use the outputs in RL

Point the RL scripts at a produced partition (see the main
[README](../README.md#agent-smoke-tests)), e.g.:
```bash
python scripts/rsl_rl/dummy_agent.py --task Sharpa-V2D-v0-Play \
  --motion_file $EXAMPLE_DIR/arctic/arctic_processed/sequence_id=dataset_s07_box_grab_01/robot_name=sharpa_wave \
  --num_envs 1 --use_primitive_urdfs
```
(Edit the `run …` lines in the script to add or change sequences.)
