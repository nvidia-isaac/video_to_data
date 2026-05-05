# Task Library Data Storage

Input data and retarget outputs live in two different places:

| | Where | How it's accessed |
|---|---|---|
| **Raw inputs** (`dataset/`) | NVIDIA CSS (PDX) swift bucket | Auto-downloaded by the workflow; browsable from the host via `list_css_sequences.py` |
| **Retarget outputs** (`*_loaded/`, `*_processed/`, `reconstructed_stage/`, `*_urdfs/`, `*_videos/`, `*_html/`) | **OSMO dataset** `v2d_{dataset}_retarget_exp_200` | One new version per workflow run; pull with `osmo dataset download` |

Object meshes are baked into the Docker image under `assets/meshes/` — they're committed to the repo via git-lfs and available at build time.

## Why OSMO datasets for outputs

Outputs used to be uploaded back to CSS at sibling prefixes (`{dataset}_loaded/`, `{dataset}_processed/`, etc.). Switching to an OSMO dataset gives us:

- **Versioning** — every workflow run bumps the dataset version, so regressions are rollback-able by version rather than overwriting production data.
- **Atomic publishes** — the whole run snapshot uploads as one manifest; partial uploads don't leave the CSS prefix in a half-written state.
- **Experiment isolation** — the `_exp_200` suffix (see `retarget.yaml:97`) keeps experimental small-sample runs out of the production `v2d_{dataset}_retarget` history. Flip the suffix back for full-dataset production runs.

The Swift output uploads in `retarget.yaml` are intentionally commented out while the `_exp_200` experiment is active (`retarget.yaml:73–90`). Re-enable them for production runs if you also want the swift mirror.

## Raw-input storage layout (CSS)

```
swift://pdx.s8k.io/AUTH_team-isaac/datasets/v2d/
  human_motion_data/
    {dataset}/
      dataset/              # Raw data (downloaded into each workflow run)
```

Supported datasets include `taco`, `arctic`, `oakink2`, `hot3d`, `h2o`, `dexycb`, `grab`.

## Retarget-output storage layout (OSMO dataset)

Each workflow run writes a single OSMO dataset version containing:

```
v2d_{dataset}_retarget_exp_200:<version>
  {dataset}_loaded/          # Stage 1 output (Parquet: MANO + object poses)
  {dataset}_urdfs/           # Stage 1.5 output (per-object rigid URDFs)
  {dataset}_processed/       # Stage 2 output (Parquet: IK-retargeted robot trajectories)
  reconstructed_stage/       # Stage 3 output (support surface USDs; may be absent for datasets where every object is held)
  {dataset}_html/            # Stage 4 output (Viser recordings + pyrender MP4s)
  {dataset}_videos/          # Stage 5 output (Isaac Sim MP4s via dummy_agent)
```

## Browsing available raw sequences on CSS

`list_css_sequences.py` lists what's on the CSS input prefix — useful for picking a sequence to target with `sequence_id` or `sequence_pattern`:

```bash
# Set CSS credentials
source scripts/setup_css_env.sh

# List all datasets
python scripts/list_css_sequences.py

# List a specific dataset
python scripts/list_css_sequences.py --dataset taco

# Filter by regex
python scripts/list_css_sequences.py --dataset taco --pattern '.*screw.*'
```

## Browsing OSMO dataset versions (outputs)

```bash
# List all versions of a retarget output dataset (most recent first)
osmo dataset info v2d_taco_retarget_exp_200 --order desc

# Inspect the file tree of a specific version
osmo dataset inspect v2d_taco_retarget_exp_200:<version>
```

## Pulling retarget outputs locally

After a workflow completes, pull its OSMO dataset version to the local repo so training / visualization scripts can find it under `source/robotic_grounding/robotic_grounding/assets/human_motion_data/{dataset}/`.

```bash
# Pick the version you want from:
osmo dataset info v2d_taco_retarget_exp_200 --order desc

# Download it (regex limits bandwidth to the components you need)
osmo dataset download v2d_taco_retarget_exp_200:<version> \
  source/robotic_grounding/robotic_grounding/assets/human_motion_data/taco/

# Just the processed Parquets (skip videos/html/etc. to save bandwidth)
osmo dataset download v2d_taco_retarget_exp_200:<version> \
  source/robotic_grounding/robotic_grounding/assets/human_motion_data/taco/ \
  --regex 'taco_processed/.*'

# Processed + URDFs + support surfaces (everything training needs)
osmo dataset download v2d_taco_retarget_exp_200:<version> \
  source/robotic_grounding/robotic_grounding/assets/human_motion_data/taco/ \
  --regex '(taco_processed|taco_urdfs|reconstructed_stage)/.*'
```

The downloaded layout matches what the training scripts expect — the `motion_file` arg to `scripts/rsl_rl/train.py` resolves as `{dataset}/{dataset}_processed/{sequence_id}/sharpa_wave` relative to `HUMAN_MOTION_DATA_DIR`.

### Legacy CSS outputs (pre-OSMO-dataset migration)

`sync_css_data.py` still works for pulling older outputs from the CSS swift prefix if you're looking at a run that pre-dates the OSMO-dataset switch (or a production run where the swift uploads were re-enabled in `retarget.yaml`):

```bash
source scripts/setup_css_env.sh
python scripts/sync_css_data.py --dataset taco --component processed
```

For current experiment runs the CSS output prefix will be stale — prefer `osmo dataset download` above.
