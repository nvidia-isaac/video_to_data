"""Bootstrap and diagnose a portable MV-HOI orchestration host."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

try:
    from .runtime import MV_HOI_DIR, config_path, env_file, state_dir, state_path
except ImportError:
    from runtime import MV_HOI_DIR, config_path, env_file, state_dir, state_path


CRON_MARKER_START = "# BEGIN v2d-mv-hoi managed"
CRON_MARKER_END = "# END v2d-mv-hoi managed"


def _default_env_text() -> str:
    state = state_dir()
    production_db = Path(
        os.environ.get("MV_HOI_DB_PATH", state / "db" / "processing_v2.db")
    ).expanduser().resolve()
    test_db = Path(
        os.environ.get("MV_HOI_TEST_DB_PATH", state / "db" / "processing_test_v2.db")
    ).expanduser().resolve()
    venv = Path(os.environ.get("MV_HOI_VENV", state / "venv")).expanduser().resolve()
    return "\n".join([
        "# Private MV-HOI submit-host configuration. Do not commit this file.",
        f"export MV_HOI_STATE_DIR={state}",
        f"export MV_HOI_DB_PATH={production_db}",
        f"export MV_HOI_TEST_DB_PATH={test_db}",
        f"export MV_HOI_CONFIG_PATH={config_path()}",
        f"export MV_HOI_VENV={venv}",
        "# PostgreSQL cutover/test URLs contain no password; use ~/.pgpass or PGPASSFILE.",
        "# export PGPASSFILE=$HOME/.pgpass",
        "# export MV_HOI_POSTGRES_TEST_URL=postgresql+psycopg://user@host:5432/database?sslmode=require",
        "# Set this only after PostgreSQL migration and parity validation:",
        "# export MV_HOI_DATABASE_URL=$MV_HOI_POSTGRES_TEST_URL",
        "# Enable only on the single authorized host:",
        "# export MV_HOI_ALLOW_SUBMIT=1",
        "# export S3_ENV=$HOME/secrets/setup_s3_env.sh",
        "# CSS_ENV remains a supported legacy alias for S3_ENV.",
        "# export V2D_IMAGE_REGISTRY=registry.example.com/your-team",
        "# export KRATOS_DRS_ENV=$HOME/secrets/setup_kratos_drs_env.sh",
        "# Set to 0 while the external QC query service is unavailable; campaign work continues.",
        "# export MV_HOI_QC_QUERY_ENABLED=0",
        "# export MV_HOI_QC_QUERY_POLL_INTERVAL_SECONDS=10",
        "# export MV_HOI_QC_QUERY_MAX_POLLS=60",
        "# export HITL_AWS_ENV=$HOME/secrets/setup_hitl_aws_env.sh",
        "# export STATUS_PUBLISH_ENV=$HOME/secrets/setup_mv_hoi_status_publish_env.sh",
        "# export MV_HOI_REVALIDATION_CAMPAIGN=<frozen-campaign-name>",
        "# export MV_HOI_BACKLOG_CAMPAIGN=<frozen-campaign-name>",
        "# export MV_HOI_VERIFY_PAYLOAD_HASHES=1",
        "",
    ])


def bootstrap(*, install_dependencies: bool = True) -> None:
    state = state_dir()
    for kind in ("db", "generated", "manifests", "logs", "locks"):
        state_path(kind).mkdir(parents=True, exist_ok=True)
    environment = env_file()
    environment.parent.mkdir(parents=True, exist_ok=True)
    if not environment.exists():
        environment.write_text(_default_env_text())
        environment.chmod(0o600)
        print(f"Created private environment template: {environment}")
    else:
        print(f"Preserved existing environment file: {environment}")

    venv = state / "venv"
    if install_dependencies:
        if not (venv / "bin" / "python").exists():
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([
            str(venv / "bin" / "python"), "-m", "pip", "install", "-r",
            str(MV_HOI_DIR / "requirements.txt"),
        ], check=True)
    try:
        from .database import upgrade_database
    except ImportError:
        from database import upgrade_database
    try:
        from .database import database_target
    except ImportError:
        from database import database_target
    upgrade_database(database_target())
    print(f"State directory ready: {state}")
    print("Review host.env, migrate/copy the authoritative DB, then run host.py doctor.")


def doctor(*, pipeline_version: str | None = None) -> int:
    failures: list[str] = []
    checks: list[tuple[str, bool, str]] = []
    checks.append(("config", config_path().is_file(), str(config_path())))
    checks.append(("environment", env_file().is_file(), str(env_file())))
    checks.append(("submit authority", os.environ.get("MV_HOI_ALLOW_SUBMIT") == "1",
                   "MV_HOI_ALLOW_SUBMIT=1"))
    for command in ("osmo", "aws"):
        found = shutil.which(command)
        checks.append((command, bool(found), found or "not found in PATH"))
    # Actual per-dataset reads below validate credentials, including AWS profiles
    # and instance roles; do not require static access keys here.

    loaded_config = None
    try:
        from .config_utils import load_config, validate_deployment_config
        from .database import (
            ALEMBIC_HEAD, backend_name, connect_read_only,
            current_database_revision, database_target,
        )
    except ImportError:
        from config_utils import load_config, validate_deployment_config
        from database import (
            ALEMBIC_HEAD, backend_name, connect_read_only,
            current_database_revision, database_target,
        )
    try:
        loaded_config = load_config(MV_HOI_DIR)
        for dataset, dataset_cfg in loaded_config.get("datasets", {}).items():
            for pipeline in dataset_cfg.get("pipelines", {}):
                validate_deployment_config(dataset_cfg, pipeline)
        checks.append(("dataset config", bool(loaded_config.get("datasets")), "loaded"))
    except Exception as exc:
        checks.append(("dataset config", False, str(exc)))
        loaded_config = None
    try:
        revision = current_database_revision(database_target())
        checks.append((
            "database", revision == ALEMBIC_HEAD,
            f"{database_target()} revision={revision or 'unversioned'} expected={ALEMBIC_HEAD}",
        ))
    except Exception as exc:
        checks.append(("database", False, str(exc)))
    try:
        connection = connect_read_only(database_target())
        try:
            if backend_name(database_target()) == "sqlite":
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
            else:
                invalid = connection.execute(
                    """SELECT conname FROM pg_constraint
                       WHERE connamespace=current_schema()::regnamespace
                         AND NOT convalidated"""
                ).fetchall()
                integrity = "ok" if not invalid else "invalid_constraints"
                foreign_keys = invalid
        finally:
            connection.close()
        checks.append((
            "database integrity", integrity == "ok" and not foreign_keys,
            f"integrity={integrity}, foreign_key_errors={len(foreign_keys)}",
        ))
    except Exception as exc:
        checks.append(("database integrity", False, str(exc)))
    if loaded_config:
        try:
            from .campaign_inventory import _client
        except ImportError:
            from campaign_inventory import _client
        for dataset, dataset_cfg in loaded_config.get("datasets", {}).items():
            try:
                client, bucket, prefix = _client(dataset_cfg["swift_base"])
                client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
                checks.append((f"storage {dataset}", True, dataset_cfg["swift_base"]))
            except Exception as exc:
                checks.append((f"storage {dataset}", False, str(exc)))
    if pipeline_version:
        try:
            from .registry_versions import validate_release
        except ImportError:
            from registry_versions import validate_release
        try:
            if loaded_config:
                for dataset_cfg in loaded_config.get("datasets", {}).values():
                    validate_release(
                        pipeline_version, registry=dataset_cfg.get("image_registry"),
                    )
            else:
                validate_release(pipeline_version)
            checks.append(("immutable images", True, pipeline_version))
        except Exception as exc:
            checks.append(("immutable images", False, str(exc)))

    for name, ok, detail in checks:
        print(f"{'OK' if ok else 'FAIL':4} {name}: {detail}")
        if not ok:
            failures.append(name)
    return 1 if failures else 0


def cron_block() -> str:
    root = MV_HOI_DIR / "orchestration"
    entries = [
        "CRON_TZ=America/Los_Angeles",
        f"0,30 * * * * {root / 'campaign_cron.sh'}",
        f"5,15,25,35,45,55 * * * * {root / 'cleanup_cron.sh'}",
        f"50 * * * * {root / 'mark_ready_cron.sh'}",
        f"10,40 * * * * {root / 'publish_status_cron.sh'}",
    ]
    return "\n".join([CRON_MARKER_START, *entries, CRON_MARKER_END])


def install_cron(*, apply: bool) -> None:
    block = cron_block()
    if not apply:
        print(block)
        print("\nDry-run only; pass --apply to install this managed block.")
        return
    current = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True,
    )
    text = current.stdout if current.returncode == 0 else ""
    if CRON_MARKER_START in text:
        before, tail = text.split(CRON_MARKER_START, 1)
        _, after = tail.split(CRON_MARKER_END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    updated = text.rstrip() + "\n\n" + block + "\n"
    subprocess.run(["crontab", "-"], input=updated, text=True, check=True)
    print("Installed MV-HOI cron block")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--skip-install", action="store_true")
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--version", help="Validate this immutable image release")
    cron_parser = subparsers.add_parser("install-cron")
    cron_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap(install_dependencies=not args.skip_install)
    elif args.command == "doctor":
        raise SystemExit(doctor(pipeline_version=args.version))
    else:
        install_cron(apply=args.apply)


if __name__ == "__main__":
    main()
