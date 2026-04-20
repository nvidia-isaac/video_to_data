# MV HOI Workflow Tooling

Host-side scripts for building/pushing pipeline images and submitting/querying
OSMO workflows for the MV HOI reconstruction and calibration pipelines.

## Files

| File                        | Purpose                                                            |
|-----------------------------|--------------------------------------------------------------------|
| `build_images.sh`           | Build all `v2d_*` Docker images locally                            |
| `push_images.sh`            | Tag + push images to `nvcr.io/nvstaging/isaac-amr` and record version |
| `submit.py`                 | Submit workflows (single sequence or auto-scan)                    |
| `query.py`                  | Show workflow status / summaries; owns OSMO read helpers + refresh |
| `db.py`                     | SQLite schema + CRUD for versions and workflows                    |
| `config.yaml`               | Dataset configs (swift paths, pools, QC thresholds)                |
| `osmo/*.yaml`               | OSMO workflow definitions (`mv_calibration`, `mv_hoi_reconstruction`) |
| `processing.db`             | SQLite DB for normal runs (git-ignored)                            |
| `processing_test.db`        | SQLite DB for `--test` runs (git-ignored)                          |

## Database

Two tables in `processing.db`:

### `pipeline_versions`
One row per published image set. Version is semver and strictly increasing.

| Column       | Notes                                  |
|--------------|----------------------------------------|
| `version`    | PK, semver `X.Y.Z`                     |
| `message`    | Optional release note                  |
| `created_at` | Timestamp                              |

### `workflows`
One row per OSMO workflow submission.

| Column             | Notes                                                         |
|--------------------|---------------------------------------------------------------|
| `id`               | PK                                                            |
| `sequence_name`    | Swift sequence directory name                                 |
| `dataset`          | Key from `config.yaml` → `datasets`                           |
| `pipeline_type`    | `mv_calibration` or `mv_hoi_reconstruction`                   |
| `pipeline_version` | Which `pipeline_versions.version` was used                    |
| `workflow_name`    | Locally-generated, unique                                     |
| `osmo_workflow_id` | ID returned by `osmo workflow submit`                         |
| `status`           | See state machine below                                       |
| `details`          | Free-form context (e.g. `task_failed: eval_chamfer_object`)   |
| `created_at`       | Submission time                                               |
| `updated_at`       | Last status change                                            |

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
          │ QC passes (manual)
          ▼
       ┌──────┐
       │ PASS │
       └──────┘
```

\* `FAIL` also used for `cancelled_for_resubmit` when a manual submit cancels
a running workflow.

`refresh_waiting(dataset, pipeline)` (defined in `query.py`) polls OSMO for
every `WAITING_WF` row and advances it to `WAITING_QC` or `FAIL`. It runs
automatically at the top of every `query.py` and `submit.py` invocation, so
the DB is fresh before any read or write.

`_failure_detail` only reports root-cause `FAILED` tasks in `details`;
`FAILED_UPSTREAM` / `FAILED_CANCELED` tasks are excluded.

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

# Dry run (prints osmo submit command without executing):
python submit.py ... --dry_run
```

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

`--force` bypasses all confirmation/cancel logic.

### Test mode

Pass `--test` to `submit.py` / `query.py` to route everything to an isolated
test location:

- DB: `processing_test.db` (instead of `processing.db`)
- Outputs: `_test` is appended to `calibration_output_path`,
  `data_output_path`, and `mesh_base` from `config.yaml`

Inputs (`calibration_path`, `data_path`) and HITL settings are unchanged.

```bash
python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --test
python query.py  --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --test --summary
```

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
```

`--latest` composes with both `--summary` and the default list view.

## Common maintenance

Inspect the DB directly:

```bash
sqlite3 processing.db "SELECT sequence_name, status, details FROM workflows ORDER BY created_at DESC LIMIT 20;"
sqlite3 processing.db "SELECT version, message, created_at FROM pipeline_versions ORDER BY created_at;"
```

Hand-edit a stale row:

```bash
sqlite3 processing.db "UPDATE workflows SET status='FAIL', details='...' WHERE workflow_name='...';"
```
