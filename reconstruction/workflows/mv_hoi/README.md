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
| `publish_status.py`         | Publish pipeline status and summaries to Google Sheets             |
| `*_cron.sh`                 | Scheduled submit/export/HITL-ready/status-publish wrappers         |
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
private location such as `~/secrets/`, fill in real secrets, and source the private
copies before running the corresponding scripts:

```bash
source ~/secrets/setup_css_env.sh          # submit/export Swift CSS access
source ~/secrets/setup_databricks_env.sh   # export.py Kratos queries
source ~/secrets/setup_hitl_aws_env.sh     # mark_ready.py / mark_ready_cron.sh
source ~/secrets/setup_mv_hoi_status_publish_env.sh  # Google Sheets status publishing
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
            ├─ missing prerequisite ──────────────▶ SKIPPED
            │
            ▼
       ┌──────────┐  OSMO FAILED*                  ┌──────┐
       │WAITING_WF├───────────────────────────────▶│ FAIL │
       └────┬─────┘                                └──────┘
            │ OSMO COMPLETED
            ├─ calibration ───────────────────────▶ PASS
            │
            ▼ reconstruction
       ┌──────────┐  human QC fails / invalid QC   ┌──────┐
       │WAITING_QC├───────────────────────────────▶│ FAIL │
       └────┬─────┘                                └──────┘
            │ human QC completed + export submitted
            ▼
     ┌──────────────┐  export failed               ┌──────┐
     │WAITING_EXPORT├─────────────────────────────▶│ FAIL │
     └──────┬───────┘                              └──────┘
            │ export completed
            ▼
         ┌──────┐
         │ PASS │
         └──────┘
```

\* `FAIL` is also used for `cancelled_for_resubmit` when `--force` cancels a
running `WAITING_WF` workflow before resubmitting it.

`refresh_waiting(dataset, pipeline)` (defined in `query.py`) polls OSMO for
every `WAITING_WF` row. Completed reconstruction rows advance to `WAITING_QC`,
completed calibration rows advance directly to `PASS`, and failed rows advance
to `FAIL`. `refresh_waiting_exports()` polls export OSMO workflows for
`WAITING_EXPORT` rows and advances each source row to `PASS` or `FAIL`.
These refreshes run automatically in `submit.py`, `query.py`, `export.py`, and
`publish_status.py` unless explicitly disabled where supported, so the DB is
fresh before normal reads and writes. OSMO status queries are run through a
bounded thread pool because they are mostly CLI/network I/O; DB updates remain
serial. Use `--refresh-workers` or `MV_HOI_REFRESH_WORKERS` to tune concurrency
(default: CPU core count). Refresh progress is printed as simple counts so cron
logs remain readable.

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

`SKIPPED` rows are created by `submit.py` when reconstruction prerequisites are
missing. Current skip reasons include missing `hoi_metadata.yaml`, missing
`calib_seq_name`, missing calibration output for the referenced calibration
sequence, missing object id, and missing object mesh. A repeated identical
`SKIPPED` reason for the latest row is not inserted again.

## Workflow Structure

### Calibration

The calibration pipeline scans `swift_base/calibration/<sequence>/`, writes to
`swift_base/calibration_output/<sequence>/`, and runs
`osmo/mv_calibration.yaml`:

```
rosbag_to_edex -> calibrate_extrinsics
```

`rosbag_to_edex` extracts camera images and intrinsics from the calibration
rosbag. `calibrate_extrinsics` writes the calibrated EDEX output consumed by
reconstruction sequences through `hoi_metadata.yaml`'s `calib_seq_name`.

### Reconstruction

The reconstruction pipeline scans `swift_base/data/<sequence>/`, writes to
`swift_base/data_output/<sequence>/`, and runs `osmo/mv_hoi_reconstruction.yaml`.
Before submission, `submit.py` requires:

- `hoi_metadata.yaml` in the input sequence
- `calib_seq_name` in the metadata and existing calibration output
- object id from `object.id`, `object_id`, or `object_name`
- object mesh under `mesh_base/<object_id>/.../output_aligned.glb`

The OSMO workflow is grouped as:

- Input/preprocess: `rosbag_to_edex`, then `mv_preprocess`
- Object branch: `grounding_dino`, `sam2_object_masks`, `check_object_mask`, `foundation_pose`
- Human branch: `detectron2`, `sam2_human_masks`, `sam3d_body`
- Derived exports/evals: `export_soma`, `estimate_ground_plane`, `export_fused_pointcloud`, object/human chamfer, and report-only object/human silhouette-mask metrics
- Final gates/output: `check_accuracy`, `render_hoi_overlay`, then `upload_hitl`

`check_accuracy` gates reconstruction completion inside OSMO using chamfer
thresholds plus object-mask containment. `upload_hitl` only runs after that gate
passes and stages review artifacts under the configured HITL S3 batch.

### Export

The export pipeline does not submit one OSMO workflow per sequence. `export.py`
collects eligible `WAITING_QC` rows, checks human QC in Kratos/Databricks, and
generates a batched OSMO workflow from `osmo/mv_hoi_export.yaml` under
`osmo/generated/<export_name>/workflow.yaml`. Each accepted sequence gets a
generated export task and a generated `copy_failure_segments` task. Source rows
are marked `WAITING_EXPORT` until the generated export workflow completes.

## Configuration

`config.yaml` groups settings by real pipeline type. Each pipeline owns one or
more OSMO workflow configs:

```yaml
datasets:
  sc_office_4exo_1:
    swift_base: swift://pdx.s8k.io/AUTH_team-isaac/recordings/v2d/multiview/sc_office_4exo_1
    mesh_base: swift://pdx.s8k.io/AUTH_team-isaac/recordings/v2d/mesh
    pipelines:
      mv_calibration:
        input_path: calibration
        output_path: calibration_output
        max_concurrent: 20
        workflows:
          calibration:
            workflow_yaml: osmo/mv_calibration.yaml
      mv_hoi_reconstruction:
        input_path: data
        output_path: data_output
        export_path: data_export
        max_concurrent: 40
        workflows:
          reconstruction:
            workflow_yaml: osmo/mv_hoi_reconstruction.yaml
            qc_thresholds:
              max_chamfer_object: 40.0
              max_chamfer_human: 40.0
              min_mask_containment: 0.8
              mask_bbox_padding: 0.1
            hitl_s3_base: s3://hitl-intake-testing/production-folder/isaac-v2d-multiview/data-factory-production/video-to-data-v2d-multiview-data-collection
            hitl_batch_name_template: "batch_{date}"
          export:
            workflow_yaml: osmo/mv_hoi_export.yaml
            batch_size: 20
            kratos_table: llmdf_admin.project_285164_annotations
            kratos_status_table: llmdf_admin.item_status_transition_metrics
            kratos_project_id: 285164
            max_failure_annotations: 6
            max_failure_coverage: 0.5
    osmo_pool: isaac-dev-h100-01
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
# WAITING_QC / WAITING_EXPORT / FAIL. Use --retry_failed to include failed
# sequences. Use --force to ignore latest status and blacklist checks.
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
On a later refresh, if OSMO reports the assumed workflow ID as missing, the
placeholder is marked `FAIL` with details `submit_ambiguous_not_found`.

Manual mode submits one named sequence. It does not prompt for confirmation.
If the latest row is active or already complete, it skips unless `--force` is
passed:

| Latest status | Behavior                                                          |
|---------------|-------------------------------------------------------------------|
| (no row)      | Submit immediately.                                                |
| `FAIL`        | Submit immediately.                                                |
| `SKIPPED`     | Submit immediately if prerequisites are now present.               |
| `WAITING_WF`  | Skip unless `--force`; with `--force`, cancel OSMO, mark the old row `FAIL` with details `cancelled_for_resubmit`, then submit. |
| `WAITING_QC`  | Skip unless `--force`; with `--force`, submit a new row without canceling the old one. |
| `WAITING_EXPORT` | Skip unless `--force`; with `--force`, submit a new row without canceling the export workflow. |
| `PASS`        | Skip unless `--force`; with `--force`, submit a new row.            |

```bash
python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --sequence <name>
```

`--force` bypasses latest-status and blacklist skip checks. It only cancels
remote OSMO work for a previous `WAITING_WF` row. When a forced non-dry-run
submission succeeds for a blacklisted sequence, the matching blacklist entry is
removed.

Blacklisted sequences are skipped in both auto and manual mode. The blacklist
is scoped by dataset, so the same sequence name can be blocked for one dataset
and still submitted for another. `--force` submits blacklisted sequences anyway
and removes the blacklist entry after a successful non-dry-run submit.
Refresh also auto-blacklists a dataset/sequence when its two most recent runs
for the same pipeline both failed with identical `details`.

Missing prerequisites are not treated as OSMO failures. In reconstruction
submit, they are recorded as `SKIPPED` rows with details such as
`skipped: no hoi_metadata.yaml`, `skipped: no calib_seq_name in hoi_metadata`,
`skipped: calibration not found for <calib_seq>`, `skipped: no object_id in
hoi_metadata`, or `skipped: no mesh for object <object_id>`. `SKIPPED` is not
part of auto-submit's latest-status skip set, so future auto-submit runs will
recheck the sequence. If the same prerequisite is still missing, the existing
latest `SKIPPED` row is left as-is instead of adding a duplicate.

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
source ~/secrets/setup_css_env.sh
source ~/secrets/setup_databricks_env.sh
python export.py --dataset sc_office_4exo_1
```

The launcher leaves rows in `WAITING_QC` until the latest
`llmdf_admin.item_status_transition_metrics` partition has a latest
`status_to` beginning with `Completed` for the project and
`<workflow_name>.json` item, then reads human-QC annotation rows from the
project annotation table. With the current config, it marks rows `FAIL` when
human QC has more than 6 failure annotations, when merged failure coverage
exceeds 50% of the sequence frame count, or when a failure annotation has an
invalid frame range. Passed QC rows are batched into one OSMO export workflow
and stamped with `osmo_export_workflow_id`. Configure the Kratos tables, project
ID, and human-QC failure gates under the reconstruction pipeline's
`workflows.export` config with `kratos_table`, `kratos_status_table`,
`kratos_project_id`, `max_failure_annotations`, and `max_failure_coverage`.

Use `export_cron.sh` for scheduled export checks. It runs after the top-of-hour
submit jobs, sources CSS and Databricks credentials, refreshes waiting states,
and submits no new batch while any `WAITING_EXPORT` rows are still active.

## Marking HITL Ready

After a HITL batch has been staged, create the `ready_for_processing` marker:

```bash
source ~/secrets/setup_hitl_aws_env.sh
python mark_ready.py --dataset sc_office_4exo_1 --batch batch_YYYYMMDD
```

Use `mark_ready_cron.sh` for the scheduled previous-day batch flow; it sources
both CSS and HITL AWS credential files so it can discover the batch name and
write the marker with the dedicated HITL S3 credentials.

## Scheduled Operations

The cron wrappers live next to this README, source credentials from `~/secrets/`
by default, activate `reconstruction/.venv`, take a lock in `/tmp`, and append
logs under `reconstruction/workflows/mv_hoi/logs/`.

| Script | Default schedule in script comment | Purpose |
|--------|------------------------------------|---------|
| `submit_calibration_cron.sh` | hourly, minute 0, Pacific | Submit calibration sequences with `--retry_failed` up to `max_concurrent`. |
| `submit_reconstruction_cron.sh` | hourly, minute 0, Pacific | Submit reconstruction sequences with `--retry_failed` up to `max_concurrent`. Prefer this name for new crontabs. |
| `submit_cron.sh` | hourly, minute 0, Pacific | Legacy/equivalent reconstruction submit wrapper. |
| `export_cron.sh` | hourly, minute 30, Pacific | Check completed human QC and submit one batched export workflow when eligible. |
| `mark_ready_cron.sh` | daily 12:00 Pacific | If the previous-day HITL batch exists, create its `ready_for_processing` marker. |
| `publish_status_cron.sh` | daily 09:00 Pacific | Publish calibration and reconstruction status/summary worksheets to Google Sheets. |

Example crontab shape:

```cron
CRON_TZ=America/Los_Angeles
0 */1 * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/submit_calibration_cron.sh
0 * * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/submit_reconstruction_cron.sh
30 * * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/export_cron.sh
0 12 * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/mark_ready_cron.sh
0 9 * * * /path/to/video_to_data/reconstruction/workflows/mv_hoi/publish_status_cron.sh
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

# Tune parallel OSMO polling during the automatic WAITING_WF refresh:
python query.py --dataset <d> --pipeline <p> --summary --refresh-workers 16
```

`--latest` composes with both `--summary` and the default list view.

## Publishing Status To Google Sheets

After creating a Google service account key and sharing the target spreadsheet
with the service account email as Editor, configure the publisher env:

```bash
cp ../../scripts/setup_mv_hoi_status_publish_env.sh ~/secrets/
$EDITOR ~/secrets/setup_mv_hoi_status_publish_env.sh
source ~/secrets/setup_css_env.sh
source ~/secrets/setup_mv_hoi_status_publish_env.sh
```

Preview what would be published without contacting Google Sheets:

```bash
python publish_status.py --dataset sc_office_4exo_1 --all-pipelines --dry-run
```

Publish latest per-sequence status rows and a summary tab:

```bash
python publish_status.py --dataset sc_office_4exo_1 --all-pipelines
```

Use `publish_status_cron.sh` for scheduled daily updates. It sources CSS and
Google Sheets publisher credentials from `~/secrets/`, refreshes workflow state,
and rewrites separate worksheet pairs for calibration and reconstruction:
`calibration_status`, `calibration_summary`, `reconstruction_status`, and
`reconstruction_summary`.

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
