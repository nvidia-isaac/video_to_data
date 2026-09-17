#!/usr/bin/env python3
"""Retire one canceled backlog campaign without deleting its audit history.

The command defaults to an audit. Applying the report requires submit
authority, an exact campaign ID, a canceled campaign with no remote/ambiguous
requests, and an unchanged audit identity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
MV_HOI_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MV_HOI_DIR))

from orchestration import db  # noqa: E402
from orchestration.runtime import require_submit_authority, state_path  # noqa: E402


SCHEMA = "v2d.mv_hoi.backlog_campaign_reset.v1"
REMOTE_ACTIVE = ("RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def audit(
    campaign: int | str,
    *,
    expected_campaign_id: int,
    db_path: str,
) -> dict:
    campaign_row = db.get_campaign(campaign, db_path=db_path)
    if not campaign_row:
        raise ValueError(f"Unknown campaign: {campaign}")
    if int(campaign_row["id"]) != int(expected_campaign_id):
        raise ValueError(
            f"Campaign ID mismatch: expected {expected_campaign_id}, "
            f"found {campaign_row['id']}"
        )
    if campaign_row["campaign_type"] not in (
        "BACKLOG_REPROCESSING", "REMEDIATION",
    ):
        raise ValueError("Only backlog/remediation campaigns can be reset")

    connection = db.get_connection(db_path)
    try:
        request_rows = connection.execute(
            """SELECT sr.id, sr.sequence_id, sr.stage, sr.status,
                      s.sequence_name
               FROM stage_requests sr
               JOIN sequences s ON s.id=sr.sequence_id
               WHERE sr.campaign_id=?
               ORDER BY sr.id""",
            (campaign_row["id"],),
        ).fetchall()
        run_rows = []
        for stage, table in (
            ("preprocess", "preprocess_runs"),
            ("reconstruction", "reconstruction_runs"),
        ):
            rows = connection.execute(
                f"""SELECT run.id, run.sequence_id, run.request_id,
                           s.sequence_name, run.output_uri
                    FROM {table} run
                    JOIN stage_requests sr ON sr.id=run.request_id
                    JOIN sequences s ON s.id=run.sequence_id
                    WHERE sr.campaign_id=? AND run.is_current=1
                    ORDER BY run.id""",
                (campaign_row["id"],),
            ).fetchall()
            run_rows.extend({
                "stage": stage,
                "run_id": int(row["id"]),
                "sequence_id": int(row["sequence_id"]),
                "sequence": row["sequence_name"],
                "request_id": int(row["request_id"]),
                "output_uri": row["output_uri"],
            } for row in rows)
        blacklist_rows = connection.execute(
            """SELECT b.sequence_id, s.sequence_name, b.reason, b.created_by
               FROM blacklisted_sequences b
               JOIN sequences s ON s.id=b.sequence_id
               WHERE b.created_by=?
                 AND EXISTS (
                     SELECT 1 FROM stage_requests sr
                     WHERE sr.campaign_id=? AND sr.sequence_id=b.sequence_id
                 )
               ORDER BY s.sequence_name""",
            (
                f"campaign:{campaign_row['name']}",
                campaign_row["id"],
            ),
        ).fetchall()
    finally:
        connection.close()

    counts: dict[str, int] = {}
    for row in request_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    remote = [
        {
            "request_id": int(row["id"]),
            "sequence": row["sequence_name"],
            "stage": row["stage"],
            "status": row["status"],
        }
        for row in request_rows if row["status"] in REMOTE_ACTIVE
    ]
    membership = sorted({row["sequence_name"] for row in request_rows})
    records = {
        "campaign_id": int(campaign_row["id"]),
        "campaign_name": campaign_row["name"],
        "campaign_status": campaign_row["status"],
        "dataset": campaign_row["dataset"],
        "membership_count": len(membership),
        "request_counts": dict(sorted(counts.items())),
        "remote_active_requests": remote,
        "current_runs_to_retire": sorted(
            run_rows, key=lambda item: (item["stage"], item["run_id"]),
        ),
        "campaign_blacklists_to_remove": [
            {
                "sequence_id": int(row["sequence_id"]),
                "sequence": row["sequence_name"],
                "reason": row["reason"],
                "created_by": row["created_by"],
            }
            for row in blacklist_rows
        ],
    }
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **records,
        "audit_sha256": hashlib.sha256(_canonical(records).encode()).hexdigest(),
    }


def apply(report: dict, *, db_path: str) -> dict:
    if report["campaign_status"] != "CANCELED":
        raise RuntimeError("Campaign must be canceled before lineage reset")
    if report["remote_active_requests"]:
        raise RuntimeError("Remote or ambiguous campaign requests remain")

    now = datetime.now(timezone.utc).isoformat()
    expected_creator = f"campaign:{report['campaign_name']}"
    retired = {"preprocess": 0, "reconstruction": 0}
    connection = db.get_connection(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        campaign_row = connection.execute(
            """SELECT id, name, status FROM processing_campaigns
               WHERE id=?""",
            (report["campaign_id"],),
        ).fetchone()
        if (
            campaign_row is None
            or campaign_row["name"] != report["campaign_name"]
            or campaign_row["status"] != "CANCELED"
        ):
            raise RuntimeError("Campaign identity or status changed after audit")

        active_count = connection.execute(
            """SELECT COUNT(*) FROM stage_requests
               WHERE campaign_id=?
                 AND status IN ('RESERVED','SUBMITTED','RUNNING','UNKNOWN')""",
            (report["campaign_id"],),
        ).fetchone()[0]
        if active_count:
            raise RuntimeError("Campaign acquired remote/ambiguous work after audit")

        for item in report["current_runs_to_retire"]:
            table = {
                "preprocess": "preprocess_runs",
                "reconstruction": "reconstruction_runs",
            }[item["stage"]]
            cursor = connection.execute(
                f"""UPDATE {table}
                    SET is_current=0,
                        superseded_at=COALESCE(superseded_at, ?),
                        updated_at=?
                    WHERE id=? AND sequence_id=? AND request_id=?
                      AND is_current=1""",
                (
                    now, now, item["run_id"], item["sequence_id"],
                    item["request_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Current {item['stage']} run changed after audit: "
                    f"{item['run_id']}"
                )
            retired[item["stage"]] += 1

        blacklist_ids = [
            int(item["sequence_id"])
            for item in report["campaign_blacklists_to_remove"]
        ]
        removed_blacklists = 0
        for sequence_id in blacklist_ids:
            cursor = connection.execute(
                """DELETE FROM blacklisted_sequences
                   WHERE sequence_id=? AND created_by=?""",
                (sequence_id, expected_creator),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Campaign blacklist changed after audit: {sequence_id}"
                )
            removed_blacklists += 1

        for stage, table in (
            ("preprocess", "preprocess_runs"),
            ("reconstruction", "reconstruction_runs"),
        ):
            remaining = connection.execute(
                f"""SELECT COUNT(*) FROM {table} run
                    JOIN stage_requests sr ON sr.id=run.request_id
                    WHERE sr.campaign_id=? AND run.is_current=1""",
                (report["campaign_id"],),
            ).fetchone()[0]
            if remaining:
                raise RuntimeError(
                    f"{remaining} current {stage} campaign run(s) remain"
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {
        "retired_current_runs": retired,
        "removed_campaign_blacklists": removed_blacklists,
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--expected-campaign-id", type=int, required=True)
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    db.init_db(args.db)
    report = audit(
        args.campaign,
        expected_campaign_id=args.expected_campaign_id,
        db_path=args.db,
    )
    if args.apply:
        require_submit_authority("reset canceled backlog campaign lineage")
        report["apply_result"] = apply(report, db_path=args.db)
        report["applied_at"] = datetime.now(timezone.utc).isoformat()
    report_path = args.report or state_path("manifests") / (
        f"backlog_campaign_reset_{report['campaign_id']}_"
        f"{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    _write_report(report_path, report)
    print(json.dumps({
        "report": str(report_path),
        "campaign_id": report["campaign_id"],
        "membership_count": report["membership_count"],
        "request_counts": report["request_counts"],
        "remote_active_count": len(report["remote_active_requests"]),
        "current_runs_to_retire": len(report["current_runs_to_retire"]),
        "campaign_blacklists_to_remove": len(
            report["campaign_blacklists_to_remove"]
        ),
        "applied": bool(report.get("apply_result")),
        "apply_result": report.get("apply_result"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
