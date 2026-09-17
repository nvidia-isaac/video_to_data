"""Copy an authoritative MV-HOI SQLite database into PostgreSQL.

The SQLite source is always opened read-only. PostgreSQL schema creation is
handled by Alembic, data insertion is one transaction, and every table plus
the derived ``sequence_status`` view is compared by row count and SHA-256.

The command is a dry run unless ``--apply`` is supplied. Replacing a nonempty
development target additionally requires ``--replace-target`` and the exact
database name through ``--confirm-target-database``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Iterable

from sqlalchemy import bindparam, inspect, text, update
from sqlalchemy.engine import make_url


SCRIPT_DIR = Path(__file__).resolve().parent
MV_HOI_DIR = SCRIPT_DIR.parent
if str(MV_HOI_DIR) not in sys.path:
    sys.path.insert(0, str(MV_HOI_DIR))

from orchestration.database import (  # noqa: E402
    ALEMBIC_HEAD,
    REVISION_ALIASES,
    connect_read_only,
    create_database_engine,
    current_database_revision,
    normalize_database_url,
    upgrade_database,
)
from orchestration.schema import METADATA  # noqa: E402


TABLE_ORDER = (
    "schema_metadata",
    "pipeline_versions",
    "sequences",
    "blacklisted_sequences",
    "workflow_executions",
    "processing_campaigns",
    "stage_requests",
    "calibration_runs",
    "preprocess_runs",
    "reconstruction_runs",
    "qc_reviews",
    "export_runs",
)
DEFERRED_REFERENCES = {
    "sequences": "calibration_sequence_id",
    "stage_requests": "fulfilled_by_request_id",
}


def _source_revision(connection) -> str | None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='alembic_version'"
    ).fetchone()
    if not exists:
        return None
    row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    return row[0] if row else None


def _validate_source(source: Path) -> dict:
    connection = connect_read_only(str(source))
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        revision = _source_revision(connection)
        normalized_revision = REVISION_ALIASES.get(revision, revision)
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = sorted(set(TABLE_ORDER) - tables)
        if integrity != "ok" or foreign_keys or normalized_revision != ALEMBIC_HEAD:
            raise RuntimeError(
                "Source validation failed: "
                f"integrity={integrity}, foreign_keys={len(foreign_keys)}, "
                f"revision={revision!r}"
            )
        if missing:
            raise RuntimeError("Source is missing tables: " + ", ".join(missing))
        return {
            "revision": revision,
            "normalized_revision": normalized_revision,
            "integrity": integrity,
            "foreign_key_errors": len(foreign_keys),
        }
    finally:
        connection.close()


def _target_identity(engine) -> tuple[str, list[str]]:
    with engine.connect() as connection:
        database = connection.execute(text("SELECT current_database()")).scalar_one()
        tables = inspect(connection).get_table_names()
    return database, sorted(tables)


def _reset_target(engine, *, confirmation: str) -> None:
    database, _ = _target_identity(engine)
    if database != confirmation:
        raise RuntimeError(
            f"Target database is {database!r}, not explicit confirmation "
            f"{confirmation!r}"
        )
    with engine.begin() as connection:
        connection.execute(text("DROP VIEW IF EXISTS sequence_status"))
        METADATA.drop_all(bind=connection)
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))


def _source_cursor(connection, table_name: str):
    table = METADATA.tables[table_name]
    columns = [column.name for column in table.columns]
    selected = ", ".join(columns)
    order = ", ".join(column.name for column in table.primary_key.columns)
    statement = f"SELECT {selected} FROM {table_name}"
    if order:
        statement += f" ORDER BY {order}"
    return connection.execute(statement)


def _copy_data(source: Path, target_engine, *, batch_size: int) -> dict[str, int]:
    source_connection = connect_read_only(str(source))
    counts: dict[str, int] = {}
    deferred: dict[str, list[dict]] = {}
    try:
        with target_engine.begin() as target:
            for table_name in TABLE_ORDER:
                counts[table_name] = 0
                deferred_column = DEFERRED_REFERENCES.get(table_name)
                if deferred_column is not None:
                    deferred[table_name] = []
                cursor = _source_cursor(source_connection, table_name)
                while True:
                    batch = [dict(row) for row in cursor.fetchmany(batch_size)]
                    if not batch:
                        break
                    counts[table_name] += len(batch)
                    if table_name == "schema_metadata":
                        for row in batch:
                            target.execute(
                                text(
                                    "INSERT INTO schema_metadata(key,value) "
                                    "VALUES (:key,:value) "
                                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
                                ),
                                row,
                            )
                        continue
                    if deferred_column is not None:
                        deferred[table_name].extend(
                            {
                                "_row_id": row["id"],
                                "_reference": row[deferred_column],
                            }
                            for row in batch
                            if row[deferred_column] is not None
                        )
                        for row in batch:
                            row[deferred_column] = None
                    target.execute(
                        METADATA.tables[table_name].insert(), batch,
                    )

            for table_name, references in deferred.items():
                if not references:
                    continue
                column_name = DEFERRED_REFERENCES[table_name]
                table = METADATA.tables[table_name]
                statement = (
                    update(table)
                    .where(table.c.id == bindparam("_row_id"))
                    .values({column_name: bindparam("_reference")})
                )
                for start in range(0, len(references), batch_size):
                    target.execute(
                        statement, references[start:start + batch_size],
                    )

            for table_name in TABLE_ORDER:
                table = METADATA.tables[table_name]
                if "id" not in table.c:
                    continue
                target.execute(text(
                    f"""SELECT setval(
                        pg_get_serial_sequence('{table_name}', 'id'),
                        GREATEST(COALESCE((SELECT MAX(id) FROM {table_name}), 1), 1),
                        EXISTS(SELECT 1 FROM {table_name})
                    )"""
                ))
    finally:
        source_connection.close()
    return counts


def _hash_value(digest, value) -> None:
    if value is None:
        payload = b"N"
    elif isinstance(value, bytes):
        payload = b"B" + value
    elif isinstance(value, float):
        payload = b"F" + struct.pack("!d", value)
    elif isinstance(value, bool):
        payload = b"T" if value else b"F"
    else:
        payload = b"S" + str(value).encode("utf-8")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _hash_rows(rows: Iterable, columns: list[str]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        count += 1
        for column in columns:
            _hash_value(digest, row[column])
    return count, digest.hexdigest()


def _sqlite_relation_hash(connection, relation: str, columns: list[str], order: str):
    return _hash_rows(
        connection.execute(
            f"SELECT {', '.join(columns)} FROM {relation} ORDER BY {order}"
        ),
        columns,
    )


def _postgres_relation_hash(engine, relation: str, columns: list[str], order: str):
    statement = text(
        f"SELECT {', '.join(columns)} FROM {relation} ORDER BY {order}"
    )
    with engine.connect() as connection:
        result = connection.execution_options(
            stream_results=True, yield_per=1000,
        ).execute(statement).mappings()
        return _hash_rows(result, columns)


def _validate_parity(source: Path, target_engine) -> dict:
    source_connection = connect_read_only(str(source))
    relations: dict[str, dict] = {}
    try:
        for table_name in TABLE_ORDER:
            table = METADATA.tables[table_name]
            columns = [column.name for column in table.columns]
            order = ", ".join(column.name for column in table.primary_key.columns)
            source_count, source_hash = _sqlite_relation_hash(
                source_connection, table_name, columns, order,
            )
            target_count, target_hash = _postgres_relation_hash(
                target_engine, table_name, columns, order,
            )
            relations[table_name] = {
                "source_count": source_count,
                "target_count": target_count,
                "source_sha256": source_hash,
                "target_sha256": target_hash,
                "match": (
                    source_count == target_count and source_hash == target_hash
                ),
            }

        source_cursor = source_connection.execute(
            "SELECT * FROM sequence_status ORDER BY sequence_id"
        )
        view_columns = [item[0] for item in source_cursor.description]
        source_view = _hash_rows(source_cursor, view_columns)
        target_view = _postgres_relation_hash(
            target_engine, "sequence_status", view_columns, "sequence_id",
        )
        relations["sequence_status"] = {
            "source_count": source_view[0],
            "target_count": target_view[0],
            "source_sha256": source_view[1],
            "target_sha256": target_view[1],
            "match": source_view == target_view,
        }
    finally:
        source_connection.close()

    with target_engine.connect() as connection:
        invalid_constraints = connection.execute(text(
            """SELECT conname FROM pg_constraint
               WHERE connamespace=current_schema()::regnamespace
                 AND NOT convalidated"""
        )).scalars().all()
    mismatches = [
        relation for relation, result in relations.items()
        if not result["match"]
    ]
    return {
        "relations": relations,
        "mismatches": mismatches,
        "invalid_constraints": invalid_constraints,
        "ok": not mismatches and not invalid_constraints,
    }


def migrate(
    *,
    source: Path,
    target_url: str,
    apply: bool,
    validate_existing: bool,
    replace_target: bool,
    confirm_target_database: str | None,
    batch_size: int,
) -> dict:
    source = source.expanduser().resolve()
    source_validation = _validate_source(source)
    normalized_target = normalize_database_url(target_url)
    url = make_url(normalized_target)
    if url.get_backend_name() != "postgresql":
        raise ValueError("Migration target must be PostgreSQL")
    engine = create_database_engine(normalized_target)
    try:
        database, tables = _target_identity(engine)
        report = {
            "schema": "v2d.mv_hoi.sqlite_to_postgresql.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "source_validation": source_validation,
            "target": url.render_as_string(hide_password=True),
            "target_database": database,
            "target_initial_tables": tables,
            "apply": apply,
            "validate_existing": validate_existing,
        }
        orchestration_tables = sorted(set(tables) & (
            set(TABLE_ORDER) | {"alembic_version"}
        ))
        if validate_existing:
            if not orchestration_tables:
                raise RuntimeError("Target has no orchestration schema to validate")
            parity = _validate_parity(source, engine)
            report.update({
                "target_revision": current_database_revision(normalized_target),
                "parity": parity,
            })
            if report["target_revision"] != ALEMBIC_HEAD or not parity["ok"]:
                raise RuntimeError(
                    "Existing PostgreSQL target failed parity validation: "
                    + ", ".join(parity["mismatches"])
                )
            return report
        if not apply:
            report["ready"] = not orchestration_tables or replace_target
            return report
        if orchestration_tables:
            if not replace_target or not confirm_target_database:
                raise RuntimeError(
                    "Target already contains orchestration tables; pass "
                    "--replace-target and --confirm-target-database"
                )
            _reset_target(
                engine, confirmation=confirm_target_database,
            )
        upgrade_database(normalized_target)
        counts = _copy_data(source, engine, batch_size=batch_size)
        parity = _validate_parity(source, engine)
        report.update({
            "copied_counts": counts,
            "target_revision": current_database_revision(normalized_target),
            "parity": parity,
        })
        if report["target_revision"] != ALEMBIC_HEAD or not parity["ok"]:
            raise RuntimeError(
                "PostgreSQL copy completed but parity validation failed: "
                + ", ".join(parity["mismatches"])
            )
        return report
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--target-env", default="MV_HOI_POSTGRES_TEST_URL",
        help="Environment variable containing the password-free SQLAlchemy URL",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--validate-existing", action="store_true",
        help="Hash-compare an existing PostgreSQL copy without writing",
    )
    parser.add_argument("--replace-target", action="store_true")
    parser.add_argument("--confirm-target-database")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    target_url = os.environ.get(args.target_env)
    if not target_url:
        raise RuntimeError(f"{args.target_env} is not set")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if args.apply and args.validate_existing:
        raise ValueError("--apply and --validate-existing are mutually exclusive")
    report = migrate(
        source=args.source,
        target_url=target_url,
        apply=args.apply,
        validate_existing=args.validate_existing,
        replace_target=args.replace_target,
        confirm_target_database=args.confirm_target_database,
        batch_size=args.batch_size,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(args.report)
    print(payload, end="")


if __name__ == "__main__":
    main()
