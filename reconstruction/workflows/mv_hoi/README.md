# MV HOI Workflow Tooling

Host-side scripts for building/pushing pipeline images and submitting/querying
OSMO workflows for the MV HOI reconstruction and calibration pipelines.

## Files

| File                        | Purpose                                                            |
|-----------------------------|--------------------------------------------------------------------|
| `build_images.sh`           | Build all `v2d_*` Docker images locally                            |
| `push_images.sh`            | Tag + push images to `nvcr.io/nvstaging/isaac-amr` and record version |
| `submit.py`                 | Submit workflows (single sequence or auto-scan)                    |
| `export.py`                 | Submit batched post-QC export workflows                            |
| `query.py`                  | Show workflow status / summaries; owns OSMO read helpers + refresh |
| `mark_ready.py`             | Create HITL `ready_for_processing` markers                         |
| `blacklist.py`              | Add/list/remove dataset-scoped sequence blacklist entries          |
| `db.py`                     | SQLite schema + CRUD for versions and pipeline rows                |
| `config.yaml`               | Dataset configs (pipelines, workflow YAMLs, paths, thresholds)     |
| `osmo/*.yaml`               | OSMO workflow definitions                                          |
| `processing.db`             | SQLite DB for normal runs (git-ignored)                            |
| `pipelines_test`            | Test-mode table inside `processing.db`                              |

## Prerequisites

The workflow management scripts shell out to the `osmo` CLI. Ensure `osmo` is
installed, on `PATH`, and authenticated before running `submit.py`, `query.py`,
or commands that cancel/resubmit workflows. The scripts also expect the Swift
credentials used by `config.yaml` to be available in the environment.

Install host-side Python dependencies in the local environment that runs these
scripts:

```bash
python3 -m pip install -r requirements.txt
```

Credential templates live under `reconstruction/scripts/`; copy them to a
private location such as `~/bin/`, fill in real secrets, and source the private
copies before running the corresponding scripts:

```bash
source ~/bin/setup_css_env.sh          # submit/export Swift CSS access
source ~/bin/setup_databricks_env.sh   # export.py Kratos queries
source ~/bin/setup_hitl_aws_env.sh     # mark_ready.py / mark_ready_cron.sh
```

## Database

Core tables in `processing.db`:

### `pipeline_versions`
One row per published image set. Version is semver and strictly increasing.

| Column       | Notes                                  |
|--------------|----------------------------------------|
| `version`    | PK, semver `X.Y.Z`                     |
| `message`    | Optional release note                  |
| `created_at` | Timestamp                              |

### `pipelines`
One row per calibration or reconstruction pipeline submission. Existing DBs
with legacy `workflows` / `workflows_test` tables are migrated in place by
`init_db()`.

| Column             | Notes                                                         |
|--------------------|---------------------------------------------------------------|
| `id`               | PK                                                            |
| `sequence_name`    | Swift sequence directory name                                 |
| `dataset`          | Key from `config.yaml` → `datasets`                           |
| `pipeline_type`    | `mv_calibration` or `mv_hoi_reconstruction`                   |
| `pipeline_version` | Which `pipeline_versions.version` was used                    |
| `workflow_name`    | Locally-generated primary OSMO workflow name                  |
| `osmo_workflow_id` | ID returned by primary `osmo workflow submit`                 |
| `osmo_export_workflow_id` | Batched export workflow ID for `WAITING_EXPORT` / `PASS` rows |
| `status`           | See state machine below                                       |
| `details`          | Free-form context (e.g. `task_failed: eval_chamfer_object`)   |
| `created_at`       | Submission time                                               |
| `updated_at`       | Last status change                                            |

### `blacklisted_sequences`
Dataset-scoped sequence names that `submit.py` should ignore by default.

| Column          | Notes                                      |
|-----------------|--------------------------------------------|
| `dataset`       | Key from `config.yaml` → `datasets`; part of PK |
| `sequence_name` | Swift sequence directory name; part of PK  |
| `reason`        | Optional free-form reason                  |
| `blacklisted_at` | Timestamp when the row was first blacklisted |

## State machine

```
       submit
         │
         ▼
    ┌──────────┐  OSMO FAILED*    ┌──────┐
    │WAITING_WF├─────────────────▶│ FAIL │
    └────┬─────┘                  └──────┘
         │ OSMO COMPLETED             ▲
         ▼                            │
    ┌──────────┐  QC fails (manual)   │
    │WAITING_QC├──────────────────────┘
    └────┬─────┘
         │ Kratos completed + export submitted
         ▼
  ┌──────────────┐  export failed ┌──────┐
  │WAITING_EXPORT├───────────────▶│ FAIL │
  └──────┬───────┘                └──────┘
         │ export completed
         ▼
      ┌──────┐
      │ PASS │
      └──────┘
```

\* `FAIL` also used for `cancelled_for_resubmit` when a manual submit cancels
a running workflow.

`refresh_waiting(dataset, pipeline)` (defined in `query.py`) polls OSMO for
every `WAITING_WF` row. Completed reconstruction rows advance to `WAITING_QC`,
completed calibration rows advance directly to `PASS`, and failed rows advance
to `FAIL`. It runs automatically at the top of every `query.py` and `submit.py`
invocation, so the DB is fresh before any read or write. OSMO status queries
are run through a bounded thread pool because they are mostly CLI/network I/O;
DB updates remain serial. Use `--refresh-workers` or `MV_HOI_REFRESH_WORKERS`
to tune concurrency (default: CPU core count). Refresh progress is printed as
simple counts so cron logs remain readable.

`_failure_detail` only reports root-cause `FAILED` tasks in `details`;
`FAILED_UPSTREAM` / `FAILED_CANCELED` tasks are excluded.

If refresh sees that a sequence's two most recent runs for the same pipeline
both failed with identical `details`, it automatically adds a dataset-scoped
blacklist entry using those `details` as the reason and prints a message.

`export.py` handles the `WAITING_QC` → `WAITING_EXPORT` → `PASS` path. It
queries Kratos/Databricks for completed human QC rows, writes generated OSMO
workflow files under `osmo/generated/`, and submits at most one export workflow
at a time with up to the configured export `batch_size`. The launcher checks
only the oldest sequence-name batch of `WAITING_QC` rows, so not-yet-completed
QC for older rows will hold newer rows back instead of being skipped.

## Configuration

`config.yaml` groups settings by real pipeline type. Each pipeline owns one or
more OSMO workflow configs:

```yaml
pipelines:
  mv_calibration:
    input_path: calibration
    output_path: calibration_output
    max_concurrent: 30
    workflows:
      calibration:
        workflow_yaml: osmo/mv_calibration.yaml

  mv_hoi_reconstruction:
    input_path: data
    output_path: data_output
    export_path: data_export
    max_concurrent: 30
    workflows:
      reconstruction:
        workflow_yaml: osmo/mv_hoi_reconstruction.yaml
      export:
        workflow_yaml: osmo/mv_hoi_export.yaml
        batch_size: 30
```

## Pipeline versioning

Images are tagged with both `:latest` and `:X.Y.Z`. The semver string is
enforced strictly increasing and stored in `pipeline_versions`; each
submitted workflow records the version used.

```bash
./push_images.sh                             # auto-bump patch
./push_images.sh -m "fix OOM"                # with release note
./push_images.sh 1.2.0                       # explicit version
./push_images.sh 1.2.0 -m "initial release"
```

`submit.py` refuses to submit if `pipeline_versions` is empty — run
`push_images.sh` first.

## Build & push

```bash
# Build all images (run from this directory or anywhere — paths are resolved
# relative to the script):
./build_images.sh

# Build a single module:
./build_images.sh sam2
./build_images.sh v2d_foundation_pose

# Push (requires docker login to nvcr.io):
./push_images.sh
```

## Submitting

Auto mode scans Swift for sequences and submits up to `max_concurrent`:

```bash
# By default, skips sequences whose latest run is PASS / WAITING_WF /
# WAITING_QC / FAIL. Use --retry_failed to include failed sequences.
python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction
python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --retry_failed
python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --refresh-workers 16

# Dry run (prints osmo submit command without executing):
python submit.py ... --dry_run
```

If `osmo workflow submit` returns an ambiguous transient error such as a read
timeout, `submit.py` records a `WAITING_WF` placeholder with details starting
`submit_ambiguous:` and assumes the OSMO ID is `<workflow_name>-1`. Auto mode
then stops for that run to avoid duplicate submissions while OSMO catches up.

Manual mode submits one named sequence. The confirmation dialog kicks in
based on the latest row's status:

| Latest status | Behavior                                                          |
|---------------|-------------------------------------------------------------------|
| (no row)      | Submit immediately.                                                |
| `FAIL`        | Submit immediately.                                                |
| `WAITING_WF`  | Ask to confirm; on yes, `osmo workflow cancel` the running one, mark it `FAIL` with details `cancelled_for_resubmit`, then submit. |
| `WAITING_QC`  | Ask to confirm; on yes, submit anyway.                             |
| `PASS`        | Ask to confirm; on yes, submit anyway.                             |

```bash
python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --sequence <name>
```

`--force` bypasses all confirmation/cancel logic. When a forced non-dry-run
submission succeeds for a blacklisted sequence, the matching blacklist entry is
removed.

Blacklisted sequences are skipped in both auto and manual mode. The blacklist
is scoped by dataset, so the same sequence name can be blocked for one dataset
and still submitted for another. `--force` submits blacklisted sequences anyway
and removes the blacklist entry after a successful non-dry-run submit.

```bash
# Add or update an entry:
python blacklist.py --dataset sc_office_4exo_1 --sequence <name> --reason "bad capture"

# Add without a reason:
python blacklist.py --dataset sc_office_4exo_1 --sequence <name>

# Remove an entry:
python blacklist.py --dataset sc_office_4exo_1 --sequence <name> --remove

# List entries for a dataset:
python blacklist.py --dataset sc_office_4exo_1 --list
```

### Test mode

Pass `--test` to `submit.py` / `query.py` to route everything to an isolated
test location:

- DB table: `pipelines_test` (instead of `pipelines`)
- Outputs: `_test` is appended to calibration output, reconstruction output,
  reconstruction export output, and `mesh_base` from `config.yaml`

Inputs (`calibration` / `data`) and HITL settings are unchanged.

```bash
python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --test
python query.py  --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --test --summary
```

## Exporting

After reconstruction rows reach `WAITING_QC`, run the export launcher:

```bash
source ~/bin/setup_css_env.sh
source ~/bin/setup_databricks_env.sh
python export.py --dataset sc_office_4exo_1
```

The launcher leaves rows in `WAITING_QC` until Kratos has a completed
`<workflow_name>.json` item. It marks rows `FAIL` when human QC has more than
five failure annotations, when merged failure coverage exceeds 30% of the
sequence frame count, or when a failure annotation has an invalid range. Passed
QC rows are batched into one OSMO export workflow and stamped with
`osmo_export_workflow_id`. Configure the human-QC failure gates under the
reconstruction pipeline's `workflows.export` config with
`max_failure_annotations` and `max_failure_coverage`.

## Marking HITL Ready

After a HITL batch has been staged, create the `ready_for_processing` marker:

```bash
source ~/bin/setup_hitl_aws_env.sh
python mark_ready.py --dataset sc_office_4exo_1 --batch batch_YYYYMMDD
```

Use `mark_ready_cron.sh` for the scheduled previous-day batch flow; it sources
both CSS and HITL AWS credential files so it can discover the batch name and
write the marker with the dedicated HITL S3 credentials.

## Querying

```bash
# Single sequence (latest row):
python query.py --dataset <d> --pipeline <p> --sequence <name>

# Aggregate summary (counts + failure reasons):
python query.py --dataset <d> --pipeline <p> --summary

# Summary of only the latest row per sequence (dedupes retries):
python query.py --dataset <d> --pipeline <p> --summary --latest

# Table of all rows:
python query.py --dataset <d> --pipeline <p>

# Table of latest row per sequence:
python query.py --dataset <d> --pipeline <p> --latest

# Include all pipelines (calibration + reconstruction) in summary/list:
python query.py --dataset <d> --pipeline <p> --all-pipelines

# Tune parallel OSMO polling during the automatic WAITING_WF refresh:
python query.py --dataset <d> --pipeline <p> --summary --refresh-workers 16
```

`--latest` composes with both `--summary` and the default list view.

## Common maintenance

Inspect the DB directly:

```bash
sqlite3 processing.db "SELECT sequence_name, status, details FROM pipelines ORDER BY created_at DESC LIMIT 20;"
sqlite3 processing.db "SELECT version, message, created_at FROM pipeline_versions ORDER BY created_at;"
```

Hand-edit a stale row:

```bash
sqlite3 processing.db "UPDATE pipelines SET status='FAIL', details='...' WHERE workflow_name='...';"
```
