from pathlib import Path
import sys

import pytest
import yaml

MV_HOI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MV_HOI))

from orchestration.pool_selection import (
    PoolSelector,
    PoolSelectionError,
    PoolState,
    WorkflowResourceError,
    active_counts_from_executions,
    validate_workflow_resources,
    workload_profile_from_text,
)


GIB = 1024**3


def _node(*, cpu: int, gpu: int, memory: int = 512, storage: int = 1000):
    return {
        "platform": "test",
        "cpu": float(cpu),
        "gpu": float(gpu),
        "memory": float(memory * GIB),
        "storage": float(storage * GIB),
    }


def _state(
    name: str, *, cpu: int, gpu: int, quota_free: int,
    quota_used: int = 0, quota_limit: int = 100,
):
    node = _node(cpu=cpu, gpu=gpu)
    return PoolState(
        name=name, status="ONLINE", quota_used=quota_used,
        quota_free=quota_free, quota_limit=quota_limit, total_free=gpu,
        nodes=[dict(node)], capability_nodes=[dict(node)],
    )


CPU_WORKFLOW = """
workflow:
  resources:
    cpu_large: {cpu: 32, gpu: 0, memory: 64Gi, storage: 100Gi}
  tasks:
  - name: preprocess
    resource: cpu_large
"""

GPU_WORKFLOW = """
workflow:
  resources:
    gpu: {cpu: 8, gpu: 1, memory: 64Gi, storage: 100Gi}
    cpu_large: {cpu: 32, gpu: 0, memory: 64Gi, storage: 100Gi}
  tasks:
  - name: reconstruct
    resource: gpu
  - name: finish
    resource: cpu_large
"""


def test_production_workflows_have_exact_explicit_resource_profiles():
    expected = {
        "mv_calibration.yaml": ("cpu", "cpu_large", {"cpu_large"}),
        "mv_preprocess.yaml": ("cpu", "cpu_large", {"cpu_large"}),
        "mv_preprocess_oneoff.yaml": ("cpu", "cpu_large", {"cpu_large"}),
        "mv_hoi_reconstruction.yaml": (
            "gpu", "gpu", {"gpu", "cpu_large", "cpu_small"},
        ),
        "mv_hoi_revalidation.yaml": (
            "gpu", "gpu", {"gpu", "cpu_large", "cpu_small", "cpu_export"},
        ),
        "mv_hoi_revalidation_export_retry.yaml": (
            "cpu", "cpu_export", {"cpu_small", "cpu_export"},
        ),
        "mv_hoi_build_engines.yaml": ("gpu", "gpu", {"gpu"}),
    }
    for filename, (mode, primary, resources) in expected.items():
        profile = validate_workflow_resources(MV_HOI / "osmo" / filename)
        assert profile.mode == mode
        assert profile.primary_resource == primary
        assert {item.name for item in profile.resource_classes} == resources

    export_template = (MV_HOI / "osmo" / "mv_hoi_export.yaml").read_text()
    rendered = export_template.replace(
        "__TASKS__", "  - name: export_test\n    resource: cpu_export",
    )
    profile = workload_profile_from_text(rendered, name="mv_hoi_export")
    assert profile.mode == "cpu"
    assert profile.primary.cpu == 16

    engine_builder = (MV_HOI / "osmo" / "mv_hoi_build_engines.yaml").read_text()
    assert "find /tmp/foundation_stereo_weights" in engine_builder
    assert "-name '*_sm_8_9.engine' -size +0c -exec cp" in engine_builder
    assert "read -r ENGINE" not in engine_builder


@pytest.mark.parametrize(
    "workflow, message",
    [
        (
            """workflow:\n  resources:\n    default: {cpu: 1}\n  tasks:\n  - name: a\n""",
            "no explicit resource profile",
        ),
        (
            """workflow:\n  resources:\n    cpu: {cpu: 1}\n  tasks:\n  - name: a\n    resource: missing\n""",
            "undefined resource profile",
        ),
        (
            """workflow:\n  resources:\n    cpu: {cpu: 1}\n    unused: {cpu: 2}\n  tasks:\n  - name: a\n    resource: cpu\n""",
            "unused resource profile",
        ),
    ],
)
def test_resource_validation_rejects_drift(workflow, message):
    with pytest.raises(WorkflowResourceError, match=message):
        workload_profile_from_text(workflow, name="bad")


def test_cpu_and_gpu_workloads_choose_the_pool_with_more_fit_slots():
    selector = PoolSelector(
        ("h100", "l40s"), "h100",
        {
            "h100": _state("h100", cpu=64, gpu=8, quota_free=1),
            "l40s": _state("l40s", cpu=256, gpu=8, quota_free=6),
        },
        log=lambda _message: None,
    )
    assert selector.choose(
        "preprocess", workflow_text=CPU_WORKFLOW,
    ).pool == "l40s"
    assert selector.choose(
        "reconstruction", workflow_text=GPU_WORKFLOW,
    ).pool == "l40s"


def test_provisional_reservations_distribute_a_dispatch_burst():
    selector = PoolSelector(
        ("h100", "l40s"), "h100",
        {
            "h100": _state("h100", cpu=64, gpu=2, quota_free=2),
            "l40s": _state("l40s", cpu=96, gpu=3, quota_free=3),
        },
        log=lambda _message: None,
    )
    selected = [
        selector.choose("reconstruction", workflow_text=GPU_WORKFLOW).pool
        for _ in range(5)
    ]
    assert selected.count("h100") == 2
    assert selected.count("l40s") == 3


def test_gpu_ties_use_total_free_then_configured_order():
    h100 = _state("h100", cpu=64, gpu=2, quota_free=1)
    l40s = _state("l40s", cpu=64, gpu=4, quota_free=1)
    selector = PoolSelector(
        ("h100", "l40s"), "h100", {"h100": h100, "l40s": l40s},
        active_counts={"reconstruction": {"h100": 0, "l40s": 99}},
        log=lambda _message: None,
    )
    assert selector.choose(
        "reconstruction", workflow_text=GPU_WORKFLOW,
    ).pool == "l40s"

    h100 = _state("h100", cpu=64, gpu=2, quota_free=1)
    l40s = _state("l40s", cpu=64, gpu=2, quota_free=1)
    selector = PoolSelector(
        ("h100", "l40s"), "h100", {"h100": h100, "l40s": l40s},
        log=lambda _message: None,
    )
    assert selector.choose(
        "reconstruction", workflow_text=GPU_WORKFLOW,
    ).pool == "h100"


def test_cpu_ties_use_aggregate_free_cpu_then_configured_order():
    selector = PoolSelector(
        ("h100", "l40s"), "h100",
        {
            "h100": _state("h100", cpu=64, gpu=1, quota_free=1),
            "l40s": _state("l40s", cpu=80, gpu=1, quota_free=1),
        },
        active_counts={"preprocess": {"h100": 0, "l40s": 99}},
        log=lambda _message: None,
    )
    assert selector.choose(
        "preprocess", workflow_text=CPU_WORKFLOW,
    ).pool == "l40s"


def test_zero_immediate_capacity_routes_overflow_without_blocking():
    selector = PoolSelector(
        ("h100", "l40s"), "h100",
        {
            "h100": _state(
                "h100", cpu=64, gpu=2, quota_free=0,
                quota_used=90, quota_limit=100,
            ),
            "l40s": _state(
                "l40s", cpu=64, gpu=2, quota_free=0,
                quota_used=50, quota_limit=100,
            ),
        },
        log=lambda _message: None,
    )
    decision = selector.choose("reconstruction", workflow_text=GPU_WORKFLOW)
    assert decision.pool == "h100"
    assert decision.reason == "queued_overflow_best_initial_capacity"


def test_query_failure_falls_back_and_explicit_override_wins():
    fallback = PoolSelector(
        ("h100", "l40s"), "h100", {}, query_error="offline",
        log=lambda _message: None,
    )
    assert fallback.choose(
        "preprocess", workflow_text=CPU_WORKFLOW,
    ).pool == "h100"

    selector = PoolSelector(
        ("h100", "l40s"), "h100",
        {
            "h100": _state("h100", cpu=32, gpu=1, quota_free=1),
            "l40s": _state("l40s", cpu=256, gpu=8, quota_free=8),
        },
        log=lambda _message: None,
    )
    assert selector.choose(
        "preprocess", workflow_text=CPU_WORKFLOW, override="h100",
    ).pool == "h100"


def test_offline_and_incapable_pools_are_excluded():
    offline = _state("h100", cpu=256, gpu=8, quota_free=8)
    offline.status = "OFFLINE"
    capable = _state("l40s", cpu=64, gpu=1, quota_free=1)
    selector = PoolSelector(
        ("h100", "l40s"), "h100", {"h100": offline, "l40s": capable},
        log=lambda _message: None,
    )
    assert selector.choose(
        "reconstruction", workflow_text=GPU_WORKFLOW,
    ).pool == "l40s"

    incapable = _state("l40s", cpu=1, gpu=0, quota_free=0)
    selector = PoolSelector(
        ("h100", "l40s"), "h100", {"h100": offline, "l40s": incapable},
        log=lambda _message: None,
    )
    with pytest.raises(PoolSelectionError, match="No online configured pool"):
        selector.choose("reconstruction", workflow_text=GPU_WORKFLOW)


def test_active_execution_counts_distinguish_revalidation_retry():
    counts = active_counts_from_executions([
        {"pipeline_stage": "preprocess", "pool": "h100"},
        {"pipeline_stage": "revalidation", "pool": "l40s", "workflow_spec_path": "revalidation.yaml"},
        {"pipeline_stage": "revalidation", "pool": "h100", "workflow_spec_path": "export_retry.yaml"},
    ])
    assert counts == {
        "mv_preprocess": {"h100": 1},
        "mv_hoi_revalidation": {"l40s": 1},
        "mv_hoi_revalidation_export_retry": {"h100": 1},
    }


def test_partial_resource_query_failure_keeps_the_healthy_pool():
    def runner(command):
        if command[1:3] == ["pool", "list"]:
            return {
                "node_sets": [{"pools": [
                    {
                        "name": pool, "status": "ONLINE",
                        "resource_usage": {
                            "quota_used": "0", "quota_free": "8",
                            "quota_limit": "8", "total_free": "8",
                        },
                    }
                    for pool in ("h100", "l40s")
                ]}],
            }
        pool = command[command.index("--pool") + 1]
        if pool == "h100":
            raise RuntimeError("h100 resource endpoint unavailable")
        node = _node(cpu=256, gpu=8)
        return {"resources": [{
            "platform_available_fields": {pool: {"test": node}},
            "platform_workflow_allocatable_fields": {pool: {"test": node}},
        }]}

    selector = PoolSelector.collect(
        {
            "osmo_pools": ["h100", "l40s"],
            "osmo_pool_fallback": "h100",
        },
        runner=runner,
        log=lambda _message: None,
    )

    assert selector.choose(
        "preprocess", workflow_text=CPU_WORKFLOW,
    ).pool == "l40s"
