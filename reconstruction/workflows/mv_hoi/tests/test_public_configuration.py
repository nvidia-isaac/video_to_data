"""Portable deployment contracts; no registry, cloud or QC service is contacted."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))
from orchestration import campaign_inventory, config_utils, export, export_commit, registry_versions, submit
from orchestration.storage import parse_storage_url, s3_client_kwargs


@pytest.fixture(autouse=True)
def clean_storage_environment(monkeypatch):
    for prefix in ("S3", "CSS"):
        for key in ("ACCESS_KEY", "SECRET_KEY", "ENDPOINT_URL", "REGION"):
            monkeypatch.delenv(f"{prefix}_{key}", raising=False)


@pytest.mark.parametrize("url, expected", [
    ("s3://example-bucket/data/seq", (None, "example-bucket", "data/seq")),
    ("example-bucket/data/seq", (None, "example-bucket", "data/seq")),
    ("s3://example-bucket/", (None, "example-bucket", "")),
    ("swift://storage.example.com/AUTH_example/example-bucket/data/seq",
     ("https://storage.example.com", "example-bucket", "data/seq")),
])
def test_storage_parser_contract_across_host_entrypoints(url, expected):
    for parse in (parse_storage_url, submit._parse_swift_url,
                  export._parse_swift_url, campaign_inventory._parse_swift_url):
        assert parse(url) == expected


@pytest.mark.parametrize("url", ["", "s3://", "s3:///data", "swift://host/account", "https://host/bucket"])
def test_invalid_remote_paths_are_rejected(url):
    with pytest.raises(ValueError):
        parse_storage_url(url)


def test_s3_settings_override_legacy_aliases(monkeypatch):
    for prefix, value in (("CSS", "legacy"), ("S3", "configured")):
        for key in ("ACCESS_KEY", "SECRET_KEY", "ENDPOINT_URL", "REGION"):
            monkeypatch.setenv(f"{prefix}_{key}", f"{value}-{key}")
    assert s3_client_kwargs("https://url-host") == {
        "endpoint_url": "configured-ENDPOINT_URL",
        "aws_access_key_id": "configured-ACCESS_KEY",
        "aws_secret_access_key": "configured-SECRET_KEY",
        "region_name": "configured-REGION",
    }
    assert parse_storage_url("s3://bucket/key")[0] == "configured-ENDPOINT_URL"


def test_standard_aws_chain_and_explicit_legacy_settings(monkeypatch):
    assert s3_client_kwargs() == {"endpoint_url": None}
    monkeypatch.setenv("CSS_ACCESS_KEY", "legacy-access")
    monkeypatch.setenv("CSS_SECRET_KEY", "legacy-secret")
    assert s3_client_kwargs()["aws_access_key_id"] == "legacy-access"
    monkeypatch.delenv("CSS_SECRET_KEY")
    with pytest.raises(ValueError, match="both S3_ACCESS_KEY"):
        s3_client_kwargs()


@pytest.mark.parametrize("factory", [submit.get_s3_client, export.get_s3_client,
                                      campaign_inventory._client, export_commit._parse])
def test_all_cloud_clients_use_standard_s3_and_configured_credentials(monkeypatch, factory):
    import boto3
    calls = []
    fake_client = object()
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: calls.append((args, kwargs)) or fake_client)
    monkeypatch.setenv("S3_ACCESS_KEY", "test-access")
    monkeypatch.setenv("S3_SECRET_KEY", "test-secret")
    monkeypatch.setenv("S3_REGION", "us-west-2")
    assert factory("s3://example-bucket/data/seq") == (fake_client, "example-bucket", "data/seq")
    assert calls[0][1]["endpoint_url"] is None
    assert calls[0][1]["region_name"] == "us-west-2"
    assert calls[0][1]["aws_access_key_id"] == "test-access"


def test_legacy_sync_script_accepts_s3_urls_and_sdk_credentials(monkeypatch):
    path = WORKFLOW_DIR.parents[1] / "scripts" / "sync_css.py"
    spec = importlib.util.spec_from_file_location("portable_sync_css", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._parse_swift_url("s3://example-bucket/data/seq") == ("example-bucket", "data/seq")
    calls = []
    monkeypatch.setattr(module.boto3, "client", lambda *args, **kwargs: calls.append(kwargs))
    module._get_s3_client("swift://storage.example.com/AUTH_example/bucket/key")
    assert calls[0]["endpoint_url"] == "https://storage.example.com"
    assert "aws_access_key_id" not in calls[0]


def test_public_template_reports_missing_deployment_settings(monkeypatch):
    monkeypatch.delenv("V2D_IMAGE_REGISTRY", raising=False)
    cfg = config_utils.load_config(WORKFLOW_DIR)["datasets"]["sc_office_4exo_1"]
    with pytest.raises(ValueError, match="Missing deployment settings") as exc:
        config_utils.validate_deployment_config(cfg, config_utils.RECON_PIPELINE)
    for name in ("swift_base", "mesh_base", "weights_base_url", "image_registry", "osmo_pools", "hitl_s3_base"):
        assert name in str(exc.value)
    assert config_utils.get_cleanup_settings({"datasets": {"example": cfg}}, "example", "any-campaign") is None


def test_registry_requires_explicit_configuration(monkeypatch):
    monkeypatch.delenv("V2D_IMAGE_REGISTRY", raising=False)
    with pytest.raises(registry_versions.RegistryVersionError, match="V2D_IMAGE_REGISTRY"):
        registry_versions.docker_repository("image")
    assert registry_versions.docker_repository("image", "registry.example.com/team/") == "registry.example.com/team/image"


def test_generic_registry_release_discovery_preserves_semver_validation(monkeypatch):
    monkeypatch.setenv("V2D_IMAGE_REGISTRY", "registry.example.com/team")
    calls = []
    monkeypatch.setattr(registry_versions.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(
        returncode=0, stdout='{"Tags": ["latest", "1.10.0", "1.9.0"]}', stderr=""))
    assert registry_versions.list_repository_versions("image") == ["1.9.0", "1.10.0"]
    assert calls == [["skopeo", "list-tags", "docker://registry.example.com/team/image"]]


def test_kratos_requires_explicit_tables_and_project_before_query(monkeypatch):
    monkeypatch.setattr(export, "_execute_kratos_drs_json_query", lambda *a: pytest.fail("must not query"))
    with pytest.raises(export.QCQueryUnavailableError, match="kratos_status_table.*kratos_project_id"):
        export.query_completed_kratos_annotations("catalog.schema.annotations", ["workflow.json"])


def test_missing_optional_kratos_client_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "kratos", None)
    monkeypatch.delitem(sys.modules, "kratos.drs_jobs", raising=False)
    with pytest.raises(export.QCQueryUnavailableError, match="provider.*WAITING_QC|provider.*waiting for QC"):
        export._execute_kratos_drs_json_query("SELECT 1")


def test_sync_cli_reaches_listing_with_supplied_s3_url(monkeypatch, capsys):
    path = WORKFLOW_DIR.parents[1] / "scripts" / "sync_css.py"
    spec = importlib.util.spec_from_file_location("sync_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def paginate(**kwargs):
        calls.append(kwargs)
        return [{"Contents": [{"Key": "data/seq/sample.bin", "Size": 4}]}]

    client = SimpleNamespace(get_paginator=lambda _: SimpleNamespace(paginate=paginate))
    monkeypatch.setattr(module.boto3, "client", lambda *args, **kwargs: client)
    monkeypatch.setattr(sys, "argv", ["sync_css.py", "ls", "s3://example-bucket/data/seq"])
    module.main()
    assert calls == [{"Bucket": "example-bucket", "Prefix": "data/seq/", "Delimiter": "/"}]
    output = capsys.readouterr().out
    assert "Listing s3://example-bucket/data/seq" in output
    assert "sample.bin" in output
    assert "0 directories, 1 files" in output


@pytest.mark.parametrize("dataset_registry,cli_registry,expected", [
    (None, None, "registry.example.com/environment"),
    ("registry.example.com/dataset", None, "registry.example.com/dataset"),
    ("registry.example.com/dataset", "registry.example.com/cli", "registry.example.com/cli"),
])
def test_submit_cli_uses_selected_registry_without_changing_environment(
    monkeypatch, dataset_registry, cli_registry, expected,
):
    import os

    monkeypatch.setenv("V2D_IMAGE_REGISTRY", "registry.example.com/environment")
    cfg = {
        "swift_base": "s3://example-bucket/dataset", "osmo_pool": "example-pool",
        "pipelines": {config_utils.CALIBRATION_PIPELINE: {}},
    }
    if dataset_registry:
        cfg["image_registry"] = dataset_registry
    monkeypatch.setattr(submit, "load_config", lambda: {"datasets": {"example": cfg}})
    monkeypatch.setattr(submit, "init_db", lambda *_: None)
    monkeypatch.setattr(submit, "refresh_workflow_states", lambda *a, **kw: None)
    submitted = []
    monkeypatch.setattr(submit, "submit_sequence", lambda *a, **kw: submitted.append((a, kw)))
    registry_queries = []
    monkeypatch.setattr(registry_versions.subprocess, "run", lambda cmd, **kw:
        registry_queries.append(cmd) or SimpleNamespace(
            returncode=0, stdout='{"Tags": ["latest", "1.2.3"]}', stderr=""))
    argv = ["submit.py", "--dataset", "example", "--pipeline", "mv_calibration",
            "--sequence", "seq", "--dry_run"]
    if cli_registry:
        argv.extend(["--image-registry", cli_registry])
    monkeypatch.setattr(sys, "argv", argv)
    submit.main()
    assert len(registry_queries) == len(registry_versions.managed_repositories())
    assert all(cmd[-1].startswith(f"docker://{expected}/") for cmd in registry_queries)
    assert submitted[0][1]["pipeline_version"] == "1.2.3"
    assert os.environ["V2D_IMAGE_REGISTRY"] == "registry.example.com/environment"


@pytest.mark.parametrize("name,value", [
    ("swift_base", "   "), ("image_registry", "???"),
    ("image_registry", "https://registry.example.com/team"),
    ("osmo_pools", [""]), ("osmo_pools", ["???"]),
])
def test_invalid_deployment_settings_fail_before_submission(name, value):
    cfg = {
        "swift_base": "s3://example-bucket/dataset",
        "image_registry": "registry.example.com/team", "osmo_pools": ["example-pool"],
    }
    cfg[name] = value
    with pytest.raises(ValueError):
        config_utils.validate_deployment_config(cfg, config_utils.CALIBRATION_PIPELINE)
