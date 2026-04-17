"""Query workflow status, metrics, and aggregate summaries.

Single sequence:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --sequence <name>

Aggregate summary:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --summary

List all:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction
"""

import argparse
import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from db import (
    get_latest_workflow,
    get_summary,
    get_workflows_by_dataset,
    init_db,
)

DB_PATH = os.path.join(SCRIPT_DIR, "processing.db")


def load_config() -> dict:
    with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def show_sequence(dataset: str, sequence: str, pipeline_type: str) -> None:
    wf = get_latest_workflow(sequence, dataset, pipeline_type, db_path=DB_PATH)
    if not wf:
        print(f"No workflows found for {sequence}")
        return

    print(f"Sequence:  {wf['sequence_name']}")
    print(f"Dataset:   {wf['dataset']}")
    print(f"Pipeline:  {wf['pipeline_type']}")
    print(f"Version:   {wf['pipeline_version']}")
    print(f"Workflow:  {wf['workflow_name']}")
    print(f"OSMO ID:   {wf.get('osmo_workflow_id', '')}")
    print(f"Status:    {wf['status']}")
    print(f"Details:   {wf['details']}")
    print(f"Created:   {wf['created_at']}")
    print(f"Updated:   {wf['updated_at']}")


def show_summary(dataset: str, pipeline_type: str | None = None) -> None:
    summary = get_summary(dataset, pipeline_type=pipeline_type, db_path=DB_PATH)

    total = sum(summary["counts"].values())
    pipeline_label = pipeline_type or "all pipelines"
    print(f"=== Summary for {dataset} ({pipeline_label}) ===")
    print(f"Total workflows: {total}")
    for status in ("IN_PROGRESS", "PASS", "FAIL"):
        count = summary["counts"].get(status, 0)
        print(f"  {status}: {count}")
    for status, count in sorted(summary["counts"].items()):
        if status not in ("IN_PROGRESS", "PASS", "FAIL"):
            print(f"  {status}: {count}")

    if summary["failure_reasons"]:
        print("\nFailure reasons:")
        for reason, count in summary["failure_reasons"].items():
            print(f"  [{count}] {reason or '(no details)'}")


def show_list(dataset: str, pipeline_type: str | None = None) -> None:
    workflows = get_workflows_by_dataset(
        dataset, pipeline_type=pipeline_type, db_path=DB_PATH,
    )
    if not workflows:
        print(f"No workflows found for {dataset}")
        return

    header = f"{'Sequence':<30} {'Pipeline':<25} {'Status':<12} {'Ver':<6} {'Details'}"
    print(header)
    print("-" * len(header))
    for wf in workflows:
        ver = wf["pipeline_version"] or "?"
        print(
            f"{wf['sequence_name']:<30} "
            f"{wf['pipeline_type']:<25} "
            f"{wf['status']:<12} "
            f"{ver:<6} "
            f"{wf['details'] or ''}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Query MV pipeline workflow status")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--pipeline", required=True,
                        help="Pipeline type (e.g. mv_calibration, mv_hoi_reconstruction)")
    parser.add_argument("--sequence", help="Show details for a specific sequence")
    parser.add_argument("--summary", action="store_true", help="Show aggregate summary")
    parser.add_argument("--all-pipelines", action="store_true",
                        help="Include all pipeline types in summary/list")
    args = parser.parse_args()

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    init_db(DB_PATH)

    pipeline_type = None if args.all_pipelines else args.pipeline

    if args.sequence:
        show_sequence(args.dataset, args.sequence, args.pipeline)
    elif args.summary:
        show_summary(args.dataset, pipeline_type=pipeline_type)
    else:
        show_list(args.dataset, pipeline_type=pipeline_type)


if __name__ == "__main__":
    main()
