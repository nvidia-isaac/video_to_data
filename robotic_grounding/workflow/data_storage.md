# Task Library Data Storage

All task-library data — raw inputs, the `{dataset}_loaded` Parquet, and the
retarget outputs — lives on the NVIDIA CSS (PDX) swift bucket under a single
per-dataset prefix:

```
swift://pdx.s8k.io/AUTH_team-isaac/datasets/v2d/human_motion_data/{dataset}/
```

| | Where | Produced by | How it's accessed |
|---|---|---|---|
| **Raw inputs** (`dataset/`) | swift `…/{dataset}/dataset/` | uploaded once per dataset | auto-downloaded by the workflows; browse with `list_css_sequences.py` |
| **Loaded** (`{dataset}_loaded/`) | swift `…/{dataset}/{dataset}_loaded/` | **reconstruction** `v2d_{dataset}_load` workflow (MANO FK; GPL-contained) | consumed by `retarget.yaml`; pull with `sync_css_data.py` |
| **Retarget outputs** (`{dataset}_processed/`, `{dataset}_urdfs/`, `reconstructed_stage/`, `{dataset}_html/`, `{dataset}_videos/`, `{dataset}_quality.csv`) | swift `…/{dataset}/…` | `retarget.yaml` (writes to `output_url`) | pull with `sync_css_data.py` / `aws s3` |

Object meshes + URDFs (arctic, taco, oakink2, hot3d) live on swift under
`…/{dataset}/object_assets/{meshes,urdfs}/{dataset}/` — **not committed** to the
repo. Fetch them locally with `scripts/fetch_object_assets.py --dataset <name>`
(upload with `scripts/upload_object_assets.py`). h2o/grab/dexycb keep meshes in
the raw dataset. Robot meshes (g1, sharpa_wave, …) remain committed under
`assets/meshes/`.

## Why swift URLs (not OSMO datasets)

Both workflows publish via swift `url:` outputs rather than versioned OSMO
`dataset:` outputs:

- The OSMO **`isaac` dataset bucket is read-only** (`dataset:` writes are
  rejected), so workflows write to the team's swift `AUTH_team-isaac` path
  instead.
- It keeps the two stages **consistent** — the reconstruction load workflow
  (`loaded_output_url`) and the RG retarget workflow (`output_url`) both write
  swift prefixes with the same `…/{dataset}/…` layout.

**Tradeoff — no automatic per-run versioning.** A swift `url:` output overwrites
the prefix in place; there's no rollback-by-version like an OSMO dataset gave.
For experiments, **override `output_url` to a scratch prefix** so you don't
clobber the canonical `{dataset}_processed/` (mirrors how the load workflow's
`loaded_output_url` defaults to canonical and is overridden for tests):

```bash
python scripts/run_osmo.py --experiment-name retarget-taco-test \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=taco --set stages=process \
  --set output_url=swift://pdx.s8k.io/AUTH_team-isaac/datasets/v2d/human_motion_data/taco/scratch_taco_test/
```

(`retarget.yaml`'s `dataset_suffix` is deprecated/unused now that output is a URL;
override `output_url` instead.)

## Storage layout (swift)

```
swift://pdx.s8k.io/AUTH_team-isaac/datasets/v2d/
  human_motion_data/
    {dataset}/
      dataset/                # Raw data (downloaded into each workflow run)
      {dataset}_loaded/       # reconstruction load workflow (Parquet: MANO + object poses)
      {dataset}_urdfs/        # Stage 1.5 (per-object rigid URDFs)
      {dataset}_processed/    # Stage 2 (Parquet: IK-retargeted robot trajectories)
      reconstructed_stage/    # Stage 3 (support-surface USDs; absent when every object is held)
      {dataset}_html/         # Stage 4 (Viser recordings + pyrender MP4s)
      {dataset}_videos/       # Stage 5 (Isaac Sim MP4s via dummy_agent)
      {dataset}_quality.csv   # Stage 6 (per-sequence quality metrics)
```

Supported datasets include `taco`, `arctic`, `oakink2`, `hot3d`, `h2o`, `dexycb`, `grab`.

## Browsing available raw sequences on CSS

`list_css_sequences.py` lists what's on the CSS input prefix — useful for picking
a sequence to target with `sequence_id` or `sequence_pattern`:

```bash
# Set CSS credentials
source scripts/setup_css_env.sh

# List all datasets
python scripts/list_css_sequences.py

# List a specific dataset / filter by regex
python scripts/list_css_sequences.py --dataset taco
python scripts/list_css_sequences.py --dataset taco --pattern '.*screw.*'
```

## Pulling retarget outputs locally

After a workflow completes, pull its swift outputs into the local repo so
training / visualization scripts find them under
`source/robotic_grounding/robotic_grounding/assets/human_motion_data/{dataset}/`.

```bash
source scripts/setup_css_env.sh

# Pull a component (loaded / processed / support_surfaces)
python scripts/sync_css_data.py --dataset taco --component processed

# Or pull a prefix directly with the aws CLI (any component, incl. urdfs/html/videos)
aws s3 sync \
  s3://datasets/v2d/human_motion_data/taco/taco_processed/ \
  source/robotic_grounding/robotic_grounding/assets/human_motion_data/taco/taco_processed/ \
  --endpoint-url ${CSS_ENDPOINT_URL} --region us-east-1
```

The downloaded layout matches what the training scripts expect — the
`motion_file` arg to `scripts/rsl_rl/train.py` resolves as
`{dataset}/{dataset}_processed/{sequence_id}/sharpa_wave` relative to
`HUMAN_MOTION_DATA_DIR`.

> `sync_css_data.py` currently knows `loaded` / `processed` / `support_surfaces`;
> for `urdfs` / `html` / `videos` use the `aws s3 sync` form above until a
> `--component` is added.
