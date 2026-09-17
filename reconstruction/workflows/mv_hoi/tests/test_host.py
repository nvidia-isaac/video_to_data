from pathlib import Path
import sys


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import host


def test_default_environment_preserves_explicit_database_and_venv(monkeypatch, tmp_path):
    state = tmp_path / "state"
    production = tmp_path / "authoritative.db"
    test = tmp_path / "test.db"
    venv = tmp_path / "existing-venv"
    monkeypatch.setenv("MV_HOI_STATE_DIR", str(state))
    monkeypatch.setenv("MV_HOI_DB_PATH", str(production))
    monkeypatch.setenv("MV_HOI_TEST_DB_PATH", str(test))
    monkeypatch.setenv("MV_HOI_VENV", str(venv))

    text = host._default_env_text()

    assert f"export MV_HOI_DB_PATH={production}" in text
    assert f"export MV_HOI_TEST_DB_PATH={test}" in text
    assert f"export MV_HOI_VENV={venv}" in text
    assert str(state / "db" / "processing_v2.db") not in text
    assert "MV_HOI_VERIFY_PAYLOAD_HASHES=1" in text
    assert "MV_HOI_QC_QUERY_ENABLED=0" in text
    assert "KRATOS_DRS_ENV=$HOME/secrets/setup_kratos_drs_env.sh" in text
    assert "MV_HOI_QC_QUERY_POLL_INTERVAL_SECONDS=10" in text
    assert "MV_HOI_QC_QUERY_MAX_POLLS=60" in text


def test_cron_block_runs_two_campaign_cycles_hitl_readiness_and_hourly_status():
    block = host.cron_block()

    assert "0,30 * * * *" in block and "campaign_cron.sh" in block
    assert "5,15,25,35,45,55 * * * *" in block and "cleanup_cron.sh" in block
    assert "50 * * * *" in block and "mark_ready_cron.sh" in block
    assert "10,40 * * * *" in block and "publish_status_cron.sh" in block
    assert "submit_cron.sh" not in block
    assert "export_cron.sh" not in block
    assert block.count("campaign_cron.sh") == 1
    assert block.count("cleanup_cron.sh") == 1


def test_cleanup_cron_is_separately_locked_and_uses_configured_bounds():
    text = (WORKFLOW_DIR / "orchestration" / "cleanup_cron.sh").read_text()

    assert "mv_hoi_cleanup.lock" in text
    assert "mv_hoi_campaign.lock" not in text
    assert "--configured-asynchronous --summary --apply" in text
    assert "MV_HOI_BACKLOG_CAMPAIGN" in text


def test_campaign_and_export_crons_defer_missing_qc_credentials():
    orchestration_dir = WORKFLOW_DIR / "orchestration"
    for name in ("campaign_cron.sh", "export_cron.sh"):
        text = (orchestration_dir / name).read_text()
        assert 'MV_HOI_QC_QUERY_ENABLED:-1' in text
        assert 'if ! source "$KRATOS_DRS_ENV"' in text
        assert "exports will remain WAITING_QC" in text
        assert "unset PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID" in text
        assert "unset KRATOS_PROFILE KRATOS_AUTH_TYPE KRATOS_NAMESPACE" in text
