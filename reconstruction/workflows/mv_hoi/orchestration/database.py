"""Database backend and schema-upgrade helpers for MV-HOI orchestration.

SQLite remains useful for local development and as migration rollback
evidence. PostgreSQL is the production-capable backend. SQLAlchemy owns URL,
engine, schema, and migration handling; the compatibility connection returned
by :func:`connect` preserves the established qmark-SQL helper API while the
remaining helpers are migrated to SQLAlchemy Core incrementally.
"""

from __future__ import annotations

from collections.abc import Iterator as IteratorABC, Mapping
from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.pool import NullPool


MV_HOI_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = MV_HOI_DIR / "processing_v2.db"
DEFAULT_TEST_DB_PATH = MV_HOI_DIR / "processing_test_v2.db"
ALEMBIC_HEAD = "0011_request_queue_priority"
REVISION_ALIASES = {
    "0007_successful_export_effective_status": "0007_export_effective_status",
}
SUPPORTED_BACKENDS = frozenset(("sqlite", "postgresql"))

_SQLITE_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"
_POSTGRES_NOW = (
    "to_char(clock_timestamp() AT TIME ZONE 'UTC', "
    """'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')"""
)
_AUTO_ID_TABLES = frozenset((
    "sequences",
    "workflow_executions",
    "qc_reviews",
    "processing_campaigns",
    "stage_requests",
    "intermediate_cleanup_jobs",
))


class DatabaseRow(Mapping[str, Any]):
    """Mapping row that also preserves sqlite3.Row integer indexing."""

    __slots__ = ("_keys", "_values", "_mapping")

    def __init__(self, keys: tuple[str, ...], values: tuple[Any, ...]):
        self._keys = keys
        self._values = values
        self._mapping = dict(zip(keys, values))

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._mapping[key]

    def __iter__(self) -> IteratorABC[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def keys(self):
        return self._keys


def _postgres_row_factory(cursor):
    if cursor.description is None:
        return tuple
    columns = tuple(column.name for column in cursor.description)
    return lambda values: DatabaseRow(columns, tuple(values))


def _qmark_to_pyformat(statement: str) -> str:
    """Translate qmark binds without touching quoted SQL literals."""

    output: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(statement):
        char = statement[index]
        if quote:
            output.append(char)
            if char == quote:
                if index + 1 < len(statement) and statement[index + 1] == quote:
                    output.append(statement[index + 1])
                    index += 1
                else:
                    quote = None
        elif char in ("'", '"'):
            quote = char
            output.append(char)
        elif char == "?":
            output.append("%s")
        else:
            output.append(char)
        index += 1
    return "".join(output)


def _postgres_statement(statement: str) -> tuple[str, bool]:
    """Translate the small SQLite SQL surface retained by ``db.py``."""

    translated = statement.replace(_SQLITE_NOW, _POSTGRES_NOW)
    translated = re.sub(
        r"\bINSERT\s+OR\s+IGNORE\s+INTO\b",
        "INSERT INTO",
        translated,
        flags=re.IGNORECASE,
    )
    ignored_insert = bool(re.search(
        r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", statement, flags=re.IGNORECASE,
    ))
    if ignored_insert and " ON CONFLICT " not in translated.upper():
        translated = translated.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"

    returning_id = False
    insert = re.match(
        r"\s*INSERT\s+(?:OR\s+IGNORE\s+)?INTO\s+([A-Za-z_][A-Za-z0-9_]*)",
        statement,
        flags=re.IGNORECASE,
    )
    if (
        insert
        and insert.group(1).lower() in _AUTO_ID_TABLES
        and " RETURNING " not in translated.upper()
    ):
        translated = translated.rstrip().rstrip(";") + " RETURNING id"
        returning_id = True
    return _qmark_to_pyformat(translated), returning_id


class PostgresCursor:
    """Cursor facade exposing the sqlite cursor attributes used by callers."""

    def __init__(self, cursor, *, returning_id: bool = False):
        self._cursor = cursor
        self.lastrowid = None
        if returning_id:
            row = cursor.fetchone()
            self.lastrowid = row[0] if row is not None else None

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def description(self):
        return self._cursor.description

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __iter__(self):
        return iter(self._cursor)


class PostgresConnection:
    """DB-API facade translating the legacy qmark SQL at one boundary."""

    backend_name = "postgresql"

    def __init__(self, connection):
        self._connection = connection
        self._driver = connection.driver_connection
        self._driver.row_factory = _postgres_row_factory

    def execute(self, statement: str, parameters=()) -> PostgresCursor:
        stripped = statement.strip()
        if stripped.upper() == "BEGIN IMMEDIATE":
            statement = "BEGIN"
        translated, returning_id = _postgres_statement(statement)
        cursor = self._driver.execute(translated, parameters or ())
        return PostgresCursor(cursor, returning_id=returning_id)

    def executemany(self, statement: str, parameters) -> PostgresCursor:
        translated, returning_id = _postgres_statement(statement)
        if returning_id:
            # Bulk callers deliberately do not consume per-row generated IDs.
            # The compatibility translator adds RETURNING id for single-row
            # inserts, so remove that synthetic clause for executemany.
            translated = re.sub(
                r"\s+RETURNING\s+id\s*$", "", translated,
                flags=re.IGNORECASE,
            )
        cursor = self._driver.cursor()
        cursor.executemany(translated, parameters)
        return PostgresCursor(cursor)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def backend_name(target: str | os.PathLike[str] | None = None) -> str:
    return make_url(normalize_database_url(target)).get_backend_name()


def is_postgresql_connection(connection) -> bool:
    return getattr(connection, "backend_name", None) == "postgresql"


def database_target(*, test: bool = False) -> str:
    """Return the configured URL/path, resolving environment at call time."""
    if test:
        if url := os.environ.get("MV_HOI_TEST_DATABASE_URL"):
            return url
        return os.environ.get("MV_HOI_TEST_DB_PATH", str(DEFAULT_TEST_DB_PATH))
    if url := os.environ.get("MV_HOI_DATABASE_URL"):
        return url
    variable = "MV_HOI_TEST_DB_PATH" if test else "MV_HOI_DB_PATH"
    fallback = DEFAULT_TEST_DB_PATH if test else DEFAULT_DB_PATH
    return os.environ.get(variable, str(fallback))


def normalize_database_url(target: str | os.PathLike[str] | None = None) -> str:
    """Convert a path or supported URL into a normalized SQLAlchemy URL."""
    value = str(target or database_target())
    if "://" not in value:
        path = Path(value).expanduser().resolve()
        return f"sqlite+pysqlite:///{path}"
    url = make_url(value)
    if url.get_backend_name() not in SUPPORTED_BACKENDS:
        raise ValueError(
            "Unsupported orchestration database URL "
            f"{url.render_as_string(hide_password=True)!r}"
        )
    if url.get_backend_name() == "postgresql" and url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    if (
        url.get_backend_name() == "postgresql"
        and url.get_driver_name() != "psycopg"
    ):
        raise ValueError("PostgreSQL orchestration requires the psycopg driver")
    return url.render_as_string(hide_password=False)


def sqlite_path(target: str | os.PathLike[str] | None = None) -> str:
    """Return the SQLite filename represented by a path or database URL."""
    url = make_url(normalize_database_url(target))
    if url.get_backend_name() != "sqlite":
        raise ValueError("The configured orchestration database is not SQLite")
    if not url.database:
        raise ValueError("The orchestration SQLite database must have a filename")
    return url.database


def create_database_engine(
    target: str | os.PathLike[str] | None = None,
) -> Engine:
    """Create a short-lived SQLAlchemy engine with backend safety settings."""
    normalized = normalize_database_url(target)
    backend = make_url(normalized).get_backend_name()
    connect_args = (
        {"timeout": 30}
        if backend == "sqlite"
        else {"connect_timeout": 15, "application_name": "v2d-mv-hoi"}
    )
    engine = create_engine(
        normalized, future=True, poolclass=NullPool, connect_args=connect_args,
    )

    if backend == "sqlite":
        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.execute("PRAGMA busy_timeout = 30000")
                cursor.execute("PRAGMA journal_mode = WAL")
            finally:
                cursor.close()
    else:
        @event.listens_for(engine, "connect")
        def _configure_postgresql(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("SET TIME ZONE 'UTC'")
            finally:
                cursor.close()

    return engine


@lru_cache(maxsize=32)
def _runtime_engine(normalized_url: str) -> Engine:
    """Reuse SQLAlchemy engine configuration while NullPool keeps connections short-lived."""
    return create_database_engine(normalized_url)


def connect(
    target: str | os.PathLike[str] | None = None,
) -> sqlite3.Connection | PostgresConnection:
    """Return a SQLAlchemy-managed DB-API connection for compatibility helpers.

    The public DB module retains its established qmark-SQL API and dictionary
    row shapes, while connection creation, URL handling, pooling policy, and
    SQLite configuration all pass through the SQLAlchemy backend boundary.
    """
    normalized = normalize_database_url(target)
    conn = _runtime_engine(normalized).raw_connection()
    if make_url(normalized).get_backend_name() == "sqlite":
        conn.driver_connection.row_factory = sqlite3.Row
        return conn
    return PostgresConnection(conn)


def connect_read_only(
    target: str | os.PathLike[str] | None = None,
) -> sqlite3.Connection | PostgresConnection:
    """Open a query-only connection without modifying schema state."""
    if backend_name(target) == "sqlite":
        path = Path(sqlite_path(target)).expanduser().resolve()
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn
    conn = connect(target)
    conn.execute("SET default_transaction_read_only = on")
    conn.commit()
    return conn


@contextmanager
def reservation(
    target: str | os.PathLike[str] | None = None,
) -> Iterator[sqlite3.Connection | PostgresConnection]:
    """Open a transaction for scheduling reservations."""
    conn = connect(target)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def alembic_config(target: str | os.PathLike[str] | None = None):
    """Build an Alembic Config without relying on the caller's cwd."""
    from alembic.config import Config

    migrations_dir = MV_HOI_DIR / "alembic"
    config = Config(str(MV_HOI_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(migrations_dir))
    # Alembic stores options in a ConfigParser, where percent signs introduce
    # interpolation. URL-encoded PostgreSQL options therefore need escaping.
    config.set_main_option(
        "sqlalchemy.url", normalize_database_url(target).replace("%", "%%")
    )
    return config


def _current_schema_tables(connection) -> set[str]:
    """Return tables owned by the active schema, excluding search-path fallbacks."""

    schema = None
    if connection.dialect.name == "postgresql":
        schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    return set(inspect(connection).get_table_names(schema=schema))


def upgrade_database(target: str | os.PathLike[str] | None = None) -> None:
    """Validate/bootstrap a v2 database and upgrade it to Alembic head."""
    from alembic import command
    engine = create_database_engine(target)
    try:
        with engine.connect() as conn:
            tables = _current_schema_tables(conn)
            if "pipelines" in tables:
                raise RuntimeError(
                    "Legacy pipeline tables detected. Use migrate_processing_db.py; "
                    "the source database is read-only."
                )
            if "stage_runs" in tables:
                raise RuntimeError(
                    "Obsolete generic schema detected. Recreate or migrate this "
                    "database into the explicit v2 schema."
                )
            has_alembic = "alembic_version" in tables
            if has_alembic:
                current = conn.execute(text(
                    "SELECT version_num FROM alembic_version"
                )).scalar_one_or_none()
                if current in REVISION_ALIASES:
                    conn.execute(
                        text(
                            "UPDATE alembic_version SET version_num=:replacement "
                            "WHERE version_num=:current"
                        ),
                        {
                            "replacement": REVISION_ALIASES[current],
                            "current": current,
                        },
                    )
                    conn.commit()
            if tables and not has_alembic:
                required = {
                    "schema_metadata", "pipeline_versions", "sequences",
                    "blacklisted_sequences", "workflow_executions",
                    "calibration_runs", "preprocess_runs", "reconstruction_runs",
                    "qc_reviews", "export_runs",
                }
                missing = sorted(required - tables)
                if missing:
                    raise RuntimeError(
                        "Unknown orchestration schema; missing v2 tables: "
                        + ", ".join(missing)
                    )
                version = conn.execute(text(
                    "SELECT value FROM schema_metadata WHERE key='schema_version'"
                )).scalar_one_or_none()
                if version != "2":
                    raise RuntimeError(
                        f"Unsupported orchestration schema version: {version!r}"
                    )
        config = alembic_config(target)
        if tables and not has_alembic:
            command.stamp(config, "0001_v2_baseline")
        command.upgrade(config, "head")
    finally:
        engine.dispose()


def current_database_revision(
    target: str | os.PathLike[str] | None = None,
) -> str | None:
    """Read the current Alembic revision without creating or upgrading anything."""
    engine = create_database_engine(target)
    try:
        with engine.connect() as conn:
            if "alembic_version" not in _current_schema_tables(conn):
                return None
            return conn.execute(text(
                "SELECT version_num FROM alembic_version"
            )).scalar_one_or_none()
    finally:
        engine.dispose()
