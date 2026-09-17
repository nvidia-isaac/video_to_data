"""Alembic environment for the MV-HOI orchestration database."""

from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url


MV_HOI_DIR = Path(__file__).resolve().parent.parent
if str(MV_HOI_DIR) not in sys.path:
    sys.path.insert(0, str(MV_HOI_DIR))

from orchestration.schema import METADATA  # noqa: E402


config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
target_metadata = METADATA


def run_migrations_offline() -> None:
    dialect = make_url(config.get_main_option("sqlalchemy.url")).get_backend_name()
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=dialect == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        if is_sqlite:
            # SQLite cannot rebuild a referenced parent table while foreign-key
            # enforcement is enabled. Batch migrations rebuild tables to
            # change CHECK constraints, so suspend enforcement temporarily.
            connection.exec_driver_sql("PRAGMA foreign_keys = OFF")
            connection.exec_driver_sql("PRAGMA busy_timeout = 30000")
            connection.commit()
        try:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=is_sqlite,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
            connection.commit()
            if is_sqlite:
                violations = connection.exec_driver_sql(
                    "PRAGMA foreign_key_check"
                ).fetchall()
                if violations:
                    raise RuntimeError(
                        "Database migration produced "
                        f"{len(violations)} foreign-key violation(s)"
                    )
        finally:
            if is_sqlite:
                connection.exec_driver_sql("PRAGMA foreign_keys = ON")
                connection.commit()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
