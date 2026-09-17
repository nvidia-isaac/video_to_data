from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import database, db


@pytest.fixture
def postgresql_database() -> str:
    """Create an isolated schema only for explicitly enabled live tests."""

    if os.environ.get("MV_HOI_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set MV_HOI_RUN_POSTGRES_TESTS=1 for live PostgreSQL tests")
    base_url = os.environ.get("MV_HOI_POSTGRES_TEST_URL")
    if not base_url:
        pytest.skip("MV_HOI_POSTGRES_TEST_URL is not configured")

    schema = f"mvhoi_test_{uuid.uuid4().hex}"
    admin_engine = create_engine(base_url)
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    target_url = str(make_url(base_url).update_query_dict(
        # Exclude ``public`` so Alembic cannot resolve the dev copy's version
        # table while bootstrapping this isolated schema.
        {"options": f"-csearch_path={schema}"},
    ))
    try:
        database.upgrade_database(target_url)
        probe = database.create_database_engine(target_url)
        with probe.connect() as connection:
            assert connection.execute(text("SELECT current_schema()")).scalar_one() == schema
            assert connection.execute(text(
                """SELECT COUNT(*) FROM information_schema.tables
                   WHERE table_schema=current_schema()
                     AND table_type='BASE TABLE'"""
            )).scalar_one() == 13
            assert connection.execute(text(
                """SELECT COUNT(*) FROM information_schema.views
                   WHERE table_schema=current_schema()
                     AND table_name='sequence_status'"""
            )).scalar_one() == 1
        probe.dispose()
        yield target_url
    finally:
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin_engine.dispose()


def test_postgresql_request_reservations_are_disjoint(
    postgresql_database: str,
) -> None:
    """Concurrent dispatchers must never reserve the same work."""

    target = postgresql_database
    campaign = db.create_campaign(
        name="postgres-concurrency",
        campaign_type="BACKLOG_REPROCESSING",
        dataset="dataset",
        pipeline_version="1.0.0",
        output_uri="swift://example/data_export_2/",
        created_by="pytest",
        db_path=target,
    )
    campaign = db.freeze_campaign(
        campaign["id"],
        inventory_uri="swift://example/inventory.json",
        inventory_sha256="a" * 64,
        configuration_uri="swift://example/configuration.json",
        configuration_sha256="b" * 64,
        inventory_sequence_count=4,
        db_path=target,
    )
    expected_ids = {
        db.create_stage_request(
            sequence_name=f"sequence-{index}",
            dataset="dataset",
            stage="preprocess",
            pipeline_version="1.0.0",
            campaign=campaign["id"],
            cohort="BULK",
            db_path=target,
        )["id"]
        for index in range(4)
    }

    def reserve(dispatcher: str) -> set[int]:
        reserved = set()
        for request_id in sorted(expected_ids):
            reservation = db.reserve_request_with_execution(
                request_id,
                reserved_by=dispatcher,
                workflow_name=f"{dispatcher}-{request_id}",
                pipeline_version="1.0.0",
                workflow_spec_path="preprocess.yaml",
                pool="pool",
                db_path=target,
            )
            if reservation is not None:
                reserved.add(request_id)
        return reserved

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(reserve, ("dispatcher-a", "dispatcher-b"))

    assert first.isdisjoint(second)
    assert first | second == expected_ids
    statuses = db.get_sequence_status("dataset", db_path=target)
    assert len(statuses) == 4
    assert {row["preprocess_request_status"] for row in statuses} == {"SUBMITTED"}


def test_postgresql_allocates_stage_run_ids(postgresql_database: str) -> None:
    """The cross-table stage-run allocator must use PostgreSQL-valid SQL."""

    target = postgresql_database
    db.ensure_version_cached("1.0.0", db_path=target)
    first = db.create_stage_run(
        sequence_name="calibration-a",
        dataset="dataset",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="calibration-a",
        status="SUCCEEDED",
        trigger="MANUAL",
        source_uri="swift://recordings/calibration-a",
        output_uri="swift://output/calibration-a",
        db_path=target,
    )
    second = db.create_stage_run(
        sequence_name="calibration-b",
        dataset="dataset",
        stage=db.CALIBRATION_STAGE,
        pipeline_version="1.0.0",
        workflow_name="calibration-b",
        status="SUCCEEDED",
        trigger="MANUAL",
        source_uri="swift://recordings/calibration-b",
        output_uri="swift://output/calibration-b",
        db_path=target,
    )

    assert second["stage_run_id"] == first["stage_run_id"] + 1


def test_postgresql_updates_existing_blacklist(
    postgresql_database: str,
) -> None:
    """PostgreSQL must resolve the existing created_by column unambiguously."""

    target = postgresql_database
    db.upsert_blacklisted_sequence(
        "dataset",
        "sequence",
        "first failure",
        created_by="first-observer",
        db_path=target,
    )
    db.upsert_blacklisted_sequence(
        "dataset",
        "sequence",
        "new failure",
        db_path=target,
    )

    entry = db.get_blacklisted_sequence("dataset", "sequence", db_path=target)
    assert entry["reason"] == "new failure"
    assert entry["created_by"] == "first-observer"


def test_postgresql_reopens_campaign_execution_for_infrastructure_retry(
    postgresql_database: str,
) -> None:
    """The specialized retry path must atomically reopen terminal lineage."""

    target = postgresql_database
    campaign = db.create_campaign(
        name="postgres-retry",
        campaign_type="LEGACY_REVALIDATION",
        dataset="dataset",
        pipeline_version="1.0.0",
        output_uri="swift://example/data_export_2/",
        created_by="pytest",
        db_path=target,
    )
    campaign = db.freeze_campaign(
        campaign["id"],
        inventory_uri="swift://example/inventory.json",
        inventory_sha256="a" * 64,
        configuration_uri="swift://example/configuration.json",
        configuration_sha256="b" * 64,
        inventory_sequence_count=1,
        db_path=target,
    )
    request = db.create_stage_request(
        sequence_name="sequence",
        dataset="dataset",
        stage="revalidation",
        pipeline_version="1.0.0",
        campaign=campaign["id"],
        cohort="CANARY",
        db_path=target,
    )
    reservation = db.reserve_request_with_execution(
        request["id"],
        reserved_by="pytest",
        workflow_name="postgres-retry-workflow",
        pipeline_version="1.0.0",
        workflow_spec_path="workflow.yaml",
        pool="pool",
        db_path=target,
    )
    db.apply_execution_observation(
        reservation["execution_id"],
        execution_status="FAILED",
        details="task_failed: face_detector",
        request_outcomes=[{
            "request_id": request["id"],
            "status": "FAILED",
        }],
        db_path=target,
    )

    db.set_campaign_execution_retry_state(
        reservation["execution_id"],
        status="RUNNING",
        details="retryable_osmo_infrastructure_failure: restart 1/2",
        query_payload={"status": "FAILED", "task_details": {}},
        db_path=target,
    )

    assert db.get_workflow_execution(
        reservation["execution_id"], db_path=target,
    )["status"] == "RUNNING"
    assert db.get_stage_request(request["id"], db_path=target)["status"] == "RUNNING"
