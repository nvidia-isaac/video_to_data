"""Revoke known exports that slipped through a temporary QC FPS mismatch.

This is a narrow one-off cleanup tool. It only targets the sequence list below,
defaults to dry-run, and requires --apply before deleting Swift data or mutating
the local pipeline DB.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_utils import (
    EXPORT_WORKFLOW,
    RECON_PIPELINE,
    get_pipeline_export_path,
    get_workflow_cfg,
    load_config as _load_config,
)
from db import (
    PIPELINES_TABLE,
    get_blacklisted_sequence,
    get_latest_workflow,
    init_db,
    update_workflow,
    upsert_blacklisted_sequence,
)
from export import (
    DB_PATH,
    export_qc_thresholds,
    get_s3_client,
    normalize_failure_annotations,
    qc_failure_reason,
    query_completed_kratos_annotations,
    resolve_frame_count,
)


DEFAULT_DATASET = "sc_office_4exo_1"
EXPECTED_FAILURE_PREFIX = "qc_fail: failure_coverage>50%"
TARGET_SEQUENCES = (
    "2026-03-06_11-38-44_tennis_ball_pick_dribble_02",
    "2026-03-10_14-00-37_tennis_ball_roll_04",
    "2026-03-10_15-17-40_basketball_leg_lowers_01",
    "2026-03-10_15-21-24_basketball_leg_lowers_02",
    "2026-03-10_15-29-04_basketball_reach_crunch_01",
    "2026-03-10_15-30-38_basketball_reach_crunch_02",
    "2026-03-10_17-10-02_basketball_ball_pass_ankles_02",
)


def load_config() -> dict:
    return _load_config(SCRIPT_DIR)


@dataclass
class RevokeResult:
    sequence_name: str
    workflow: dict[str, Any] | None = None
    item_name: str | None = None
    frame_count: int | None = None
    failure_reason: str | None = None
    export_prefix: str | None = None
    keys: list[str] = field(default_factory=list)
    eligible: bool = False
    skip_reason: str | None = None
    already_failed: bool = False
    deleted_count: int = 0
    db_action: str = "none"
    blacklist_action: str = "none"


def export_prefix_for_sequence(
    base_prefix: str,
    dataset_cfg: dict,
    sequence_name: str,
) -> str:
    return (
        f"{base_prefix}/{get_pipeline_export_path(dataset_cfg, RECON_PIPELINE)}"
        f"/{sequence_name}"
    ).strip("/")


def list_swift_prefix_keys(client, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    folder_prefix = prefix.rstrip("/") + "/"
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def delete_swift_keys(client, bucket: str, keys: list[str]) -> int:
    batch_size = 1000
    deleted = 0
    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )
        deleted += len(batch)
    return deleted


def _is_matching_failed_row(workflow: dict[str, Any], failure_reason: str) -> bool:
    return workflow.get("status") == "FAIL" and workflow.get("details") == failure_reason


def evaluate_sequence(
    sequence_name: str,
    dataset_name: str,
    dataset_cfg: dict,
    annotations_by_item: dict[str, list[dict]],
    client,
    bucket: str,
    base_prefix: str,
    *,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> RevokeResult:
    result = RevokeResult(sequence_name=sequence_name)
    workflow = get_latest_workflow(
        sequence_name,
        dataset_name,
        RECON_PIPELINE,
        db_path=db_path,
        table=table,
    )
    result.workflow = workflow
    if not workflow:
        result.skip_reason = "no latest reconstruction workflow row"
        return result

    result.item_name = f"{workflow['workflow_name']}.json"
    annotations = annotations_by_item.get(result.item_name)
    if annotations is None:
        result.skip_reason = f"no completed Kratos annotations for {result.item_name}"
        return result

    frame_count = resolve_frame_count(
        client,
        bucket,
        base_prefix,
        dataset_cfg,
        sequence_name,
    )
    result.frame_count = frame_count
    if frame_count is None:
        result.skip_reason = "no frame_count found"
        return result

    failure_segments, invalid_reason = normalize_failure_annotations(
        annotations,
        frame_count,
    )
    if invalid_reason:
        result.skip_reason = invalid_reason
        return result

    max_failure_annotations, max_failure_coverage = export_qc_thresholds(dataset_cfg)
    failure_reason = qc_failure_reason(
        failure_segments,
        frame_count,
        max_failure_annotations=max_failure_annotations,
        max_failure_coverage=max_failure_coverage,
    )
    result.failure_reason = failure_reason
    if not failure_reason:
        result.skip_reason = "corrected QC annotations do not fail export gate"
        return result
    if not failure_reason.startswith(EXPECTED_FAILURE_PREFIX):
        result.skip_reason = f"corrected QC failure is not 50% coverage: {failure_reason}"
        return result

    status = workflow.get("status")
    result.already_failed = _is_matching_failed_row(workflow, failure_reason)
    if status != "PASS" and not result.already_failed:
        details = workflow.get("details") or ""
        result.skip_reason = (
            f"latest row status is {status!r}, details={details!r}; "
            "expected PASS or matching FAIL"
        )
        return result

    result.export_prefix = export_prefix_for_sequence(
        base_prefix,
        dataset_cfg,
        sequence_name,
    )
    result.keys = list_swift_prefix_keys(client, bucket, result.export_prefix)
    result.eligible = True
    return result


def _target_workflows(
    dataset_name: str,
    sequences: tuple[str, ...],
    *,
    db_path: str,
    table: str,
) -> dict[str, dict[str, Any] | None]:
    return {
        sequence: get_latest_workflow(
            sequence,
            dataset_name,
            RECON_PIPELINE,
            db_path=db_path,
            table=table,
        )
        for sequence in sequences
    }


def _query_target_annotations(
    workflow_by_sequence: dict[str, dict[str, Any] | None],
    export_cfg: dict,
) -> dict[str, list[dict]]:
    item_names = [
        f"{workflow['workflow_name']}.json"
        for workflow in workflow_by_sequence.values()
        if workflow
    ]
    kratos_kwargs = {}
    if "kratos_status_table" in export_cfg:
        kratos_kwargs["kratos_status_table"] = export_cfg["kratos_status_table"]
    if "kratos_project_id" in export_cfg:
        kratos_kwargs["kratos_project_id"] = export_cfg["kratos_project_id"]
    return query_completed_kratos_annotations(
        export_cfg["kratos_table"],
        item_names,
        **kratos_kwargs,
    )


def apply_revoke_results(
    results: list[RevokeResult],
    client,
    bucket: str,
    *,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> None:
    for result in results:
        if not result.eligible or not result.workflow or not result.failure_reason:
            continue

        if result.keys:
            result.deleted_count = delete_swift_keys(client, bucket, result.keys)
        else:
            result.deleted_count = 0

        if result.already_failed:
            result.db_action = "already FAIL"
        else:
            update_workflow(
                result.workflow["workflow_name"],
                status="FAIL",
                details=result.failure_reason,
                db_path=db_path,
                table=table,
            )
            result.db_action = "marked FAIL"

        existing = get_blacklisted_sequence(
            result.workflow["dataset"],
            result.sequence_name,
            db_path=db_path,
        )
        upsert_blacklisted_sequence(
            result.workflow["dataset"],
            result.sequence_name,
            reason=result.failure_reason,
            db_path=db_path,
        )
        result.blacklist_action = "updated" if existing else "inserted"


def print_summary(results: list[RevokeResult], *, apply: bool) -> None:
    mode = "apply" if apply else "dry-run"
    print(f"Revoke misaligned QC exports summary ({mode}):")
    for result in results:
        workflow_status = result.workflow.get("status") if result.workflow else "missing"
        if not result.eligible:
            print(
                f"  {result.sequence_name}: skipped; status={workflow_status!r}; "
                f"reason={result.skip_reason}"
            )
            continue

        delete_summary = (
            f"deleted={result.deleted_count}"
            if apply
            else f"would_delete={len(result.keys)}"
        )
        db_action = result.db_action
        blacklist_action = result.blacklist_action
        if not apply:
            db_action = "already FAIL" if result.already_failed else "would mark FAIL"
            blacklist_action = "would upsert"

        print(
            f"  {result.sequence_name}: eligible; status={workflow_status!r}; "
            f"reason={result.failure_reason}; export_objects={len(result.keys)}; "
            f"{delete_summary}; db={db_action}; blacklist={blacklist_action}; "
            f"prefix={result.export_prefix}"
        )


def run_revoke(
    dataset_name: str = DEFAULT_DATASET,
    *,
    apply: bool = False,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    sequences: tuple[str, ...] = TARGET_SEQUENCES,
) -> list[RevokeResult]:
    init_db(db_path)
    config = load_config()
    dataset_cfg = config["datasets"][dataset_name]
    export_cfg = get_workflow_cfg(dataset_cfg, RECON_PIPELINE, EXPORT_WORKFLOW)

    workflow_by_sequence = _target_workflows(
        dataset_name,
        sequences,
        db_path=db_path,
        table=table,
    )
    annotations_by_item = _query_target_annotations(workflow_by_sequence, export_cfg)
    client, bucket, base_prefix = get_s3_client(dataset_cfg["swift_base"])

    results = [
        evaluate_sequence(
            sequence,
            dataset_name,
            dataset_cfg,
            annotations_by_item,
            client,
            bucket,
            base_prefix,
            db_path=db_path,
            table=table,
        )
        for sequence in sequences
    ]

    if apply:
        apply_revoke_results(results, client, bucket, db_path=db_path, table=table)
    print_summary(results, apply=apply)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Revoke known exports that passed QC due to 15/30 FPS mismatch",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete Swift data_export objects and mutate processing.db",
    )
    args = parser.parse_args()

    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete Swift objects and update DB.")
    run_revoke(args.dataset, apply=args.apply, db_path=args.db_path)


if __name__ == "__main__":
    main()
