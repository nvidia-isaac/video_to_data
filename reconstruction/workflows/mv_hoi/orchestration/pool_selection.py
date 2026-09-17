"""Resource-aware OSMO pool selection for MV-HOI workflow submissions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import subprocess
from threading import Lock
from typing import Callable, Iterable

import yaml


DEFAULT_POOL = "isaac-apps-l40-05"
DEFAULT_POOLS = (
    DEFAULT_POOL,
    "isaac-dev-l40s-04",
    "isaac-dev-h100-01",
    "isaac-lab-l40s-03",
)
HIGH_PRIORITY_POOL = "isaac-apps-l40-05"
_TEMPLATE_VALUE = re.compile(r"\{\{[^{}\n]+\}\}")


@dataclass(frozen=True)
class ResourceClass:
    name: str
    cpu: float
    gpu: int
    memory_gib: float
    storage_gib: float


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    mode: str
    resource_classes: tuple[ResourceClass, ...]
    primary_resource: str

    @property
    def primary(self) -> ResourceClass:
        return next(
            item for item in self.resource_classes
            if item.name == self.primary_resource
        )


class WorkflowResourceError(ValueError):
    """Raised when workflow task/resource declarations do not reconcile."""


class PoolSelectionError(RuntimeError):
    """Raised when no online candidate supports every workflow resource class."""


@dataclass
class PoolState:
    name: str
    status: str
    quota_used: int
    quota_free: int
    quota_limit: int
    total_free: int
    nodes: list[dict[str, float]] = field(default_factory=list)
    capability_nodes: list[dict[str, float]] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        if self.quota_limit <= 0:
            return 1.0
        return self.quota_used / self.quota_limit


@dataclass(frozen=True)
class PoolDecision:
    pool: str
    workload: str
    mode: str
    reason: str
    scores: dict[str, dict[str, object]]

    def detail(self, event: str) -> str:
        payload = {
            "selected": self.pool,
            "workload": self.workload,
            "mode": self.mode,
            "reason": self.reason,
            "scores": self.scores,
        }
        return (
            f"{event}; pool_selection="
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )


def configured_pools(dataset_cfg: dict) -> tuple[str, ...]:
    pools = dataset_cfg.get("osmo_pools")
    if pools:
        normalized = tuple(dict.fromkeys(str(pool) for pool in pools if str(pool)))
        if normalized:
            return normalized
    return (str(dataset_cfg.get("osmo_pool") or DEFAULT_POOL),)


def fallback_pool(dataset_cfg: dict) -> str:
    return str(
        dataset_cfg.get("osmo_pool_fallback")
        or dataset_cfg.get("osmo_pool")
        or configured_pools(dataset_cfg)[0]
    )


def osmo_submission_priority(pool: str | None) -> str | None:
    """Return the requested OSMO priority for the selected pool."""
    if str(pool or "") == HIGH_PRIORITY_POOL:
        return "HIGH"
    return None


def _run_json(command: list[str]) -> dict:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "OSMO capacity query failed").strip()
        raise RuntimeError(message)
    return json.loads(result.stdout)


def _bytes(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    units = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "K": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
    }
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * multiplier
    return float(text)


def _number(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if text.endswith("m"):
        return float(text[:-1]) / 1000.0
    return float(text)


def _resource_class(name: str, value: dict) -> ResourceClass:
    if not isinstance(value, dict):
        raise WorkflowResourceError(f"resource {name!r} must be a mapping")
    return ResourceClass(
        name=name,
        cpu=_number(value.get("cpu")),
        gpu=int(_number(value.get("gpu"))),
        memory_gib=_bytes(value.get("memory")) / 1024**3,
        storage_gib=_bytes(value.get("storage")) / 1024**3,
    )


def workload_profile_from_text(text: str, *, name: str) -> WorkloadProfile:
    """Resolve the effective resource demand from a rendered or templated YAML."""
    normalized = _TEMPLATE_VALUE.sub("template_value", text)
    try:
        document = yaml.safe_load(normalized)
    except yaml.YAMLError as exc:
        raise WorkflowResourceError(f"{name}: invalid workflow YAML: {exc}") from exc
    workflow = (document or {}).get("workflow")
    if not isinstance(workflow, dict):
        raise WorkflowResourceError(f"{name}: missing workflow mapping")
    resources = workflow.get("resources")
    tasks = workflow.get("tasks")
    if not isinstance(resources, dict) or not resources:
        raise WorkflowResourceError(f"{name}: no resource profiles declared")
    if not isinstance(tasks, list) or not tasks:
        raise WorkflowResourceError(f"{name}: no rendered workflow tasks")

    task_resources: list[str] = []
    for task in tasks:
        if not isinstance(task, dict) or not task.get("name"):
            raise WorkflowResourceError(f"{name}: invalid workflow task")
        resource = task.get("resource")
        if not resource:
            raise WorkflowResourceError(
                f"{name}: task {task['name']!r} has no explicit resource profile"
            )
        task_resources.append(str(resource))
    used = set(task_resources)
    declared = set(resources)
    undefined = sorted(used - declared)
    unused = sorted(declared - used)
    if undefined:
        raise WorkflowResourceError(
            f"{name}: undefined resource profile(s): {', '.join(undefined)}"
        )
    if unused:
        raise WorkflowResourceError(
            f"{name}: unused resource profile(s): {', '.join(unused)}"
        )

    classes = tuple(
        _resource_class(resource_name, resources[resource_name])
        for resource_name in resources
    )
    gpu_classes = [resource for resource in classes if resource.gpu > 0]
    if gpu_classes:
        mode = "gpu"
        primary = max(
            gpu_classes,
            key=lambda item: (
                item.gpu, item.cpu, item.memory_gib, item.storage_gib,
            ),
        )
    else:
        mode = "cpu"
        primary = max(
            classes,
            key=lambda item: (item.cpu, item.memory_gib, item.storage_gib),
        )
    return WorkloadProfile(name, mode, classes, primary.name)


def workload_profile_from_path(path: str | Path, *, name: str) -> WorkloadProfile:
    return workload_profile_from_text(Path(path).read_text(), name=name)


def validate_workflow_resources(path: str | Path) -> WorkloadProfile:
    path = Path(path)
    return workload_profile_from_path(path, name=path.name)


def active_counts_from_executions(rows: Iterable[dict]) -> dict[str, dict[str, int]]:
    """Count active execution types per pool for deterministic tie-breaking."""
    stage_workloads = {
        "calibration": "mv_calibration",
        "preprocess": "mv_preprocess",
        "reconstruction": "mv_hoi_reconstruction",
        "export": "mv_hoi_export",
        "revalidation": "mv_hoi_revalidation",
    }
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        pool = row.get("pool")
        stage = row.get("pipeline_stage")
        if not pool or stage not in stage_workloads:
            continue
        workload = stage_workloads[str(stage)]
        if stage == "revalidation" and "export_retry" in str(
            row.get("workflow_spec_path") or ""
        ):
            workload = "mv_hoi_revalidation_export_retry"
        by_pool = counts.setdefault(workload, {})
        by_pool[str(pool)] = by_pool.get(str(pool), 0) + 1
    return counts


def load_active_counts(
    dataset_cfg: dict, loader: Callable[[], Iterable[dict]],
) -> dict[str, dict[str, int]]:
    """Avoid a database read when legacy configuration has only one pool."""
    if len(configured_pools(dataset_cfg)) <= 1:
        return {}
    return active_counts_from_executions(loader())


def _parse_pool_rows(payload: dict, candidates: Iterable[str]) -> dict[str, PoolState]:
    wanted = set(candidates)
    states: dict[str, PoolState] = {}
    for node_set in payload.get("node_sets", []):
        for row in node_set.get("pools", []):
            name = row.get("name")
            if name not in wanted:
                continue
            usage = row.get("resource_usage") or {}
            states[name] = PoolState(
                name=name,
                status=str(row.get("status") or "UNKNOWN").upper(),
                quota_used=max(0, int(usage.get("quota_used") or 0)),
                quota_free=max(0, int(usage.get("quota_free") or 0)),
                quota_limit=max(0, int(usage.get("quota_limit") or 0)),
                total_free=max(0, int(usage.get("total_free") or 0)),
            )
    return states


def _node_resources(platform: str, values: dict | None) -> dict[str, float]:
    values = values or {}
    return {
        "platform": str(platform),
        "cpu": _number(values.get("cpu")),
        "gpu": _number(values.get("gpu")),
        "memory": _bytes(values.get("memory")),
        "storage": _bytes(values.get("storage")),
    }


def _parse_resource_rows(payload: dict, states: dict[str, PoolState]) -> None:
    for row in payload.get("resources", []):
        available_by_pool = row.get("platform_available_fields") or {}
        capacity_by_pool = (
            row.get("platform_workflow_allocatable_fields")
            or row.get("platform_allocatable_fields")
            or {}
        )
        for pool, state in states.items():
            for platform, available in (available_by_pool.get(pool) or {}).items():
                state.nodes.append(_node_resources(platform, available))
            for platform, capacity in (capacity_by_pool.get(pool) or {}).items():
                state.capability_nodes.append(_node_resources(platform, capacity))


def _slots_for_class(nodes: list[dict[str, float]], demand: ResourceClass) -> int:
    slots = 0
    for node in nodes:
        limits: list[int] = []
        if demand.cpu:
            limits.append(math.floor(node["cpu"] / demand.cpu))
        if demand.gpu:
            limits.append(math.floor(node["gpu"] / demand.gpu))
        if demand.memory_gib:
            limits.append(math.floor(node["memory"] / (demand.memory_gib * 1024**3)))
        if demand.storage_gib:
            limits.append(math.floor(node["storage"] / (demand.storage_gib * 1024**3)))
        slots += max(0, min(limits, default=0))
    return slots


def _node_fits(node: dict[str, float], demand: ResourceClass) -> bool:
    return (
        node["cpu"] >= demand.cpu
        and node["gpu"] >= demand.gpu
        and node["memory"] >= demand.memory_gib * 1024**3
        and node["storage"] >= demand.storage_gib * 1024**3
    )


class PoolSelector:
    """One OSMO capacity snapshot with provisional per-submission reservations."""

    def __init__(
        self,
        candidates: tuple[str, ...],
        fallback: str,
        states: dict[str, PoolState],
        *,
        query_error: str | None = None,
        active_counts: dict[str, dict[str, int]] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.candidates = candidates
        self.fallback = fallback
        self.states = states
        self.query_error = query_error
        self.active_counts = active_counts or {}
        self.initial_capacities: dict[tuple[str, str], int] = {}
        self.log = log
        self._lock = Lock()

    @classmethod
    def collect(
        cls,
        dataset_cfg: dict,
        *,
        runner: Callable[[list[str]], dict] = _run_json,
        active_counts: dict[str, dict[str, int]] | None = None,
        log: Callable[[str], None] = print,
    ) -> "PoolSelector":
        candidates = configured_pools(dataset_cfg)
        fallback = fallback_pool(dataset_cfg)
        if len(candidates) == 1:
            state = PoolState(candidates[0], "ONLINE", 0, 0, 0, 0)
            return cls(
                candidates, fallback, {candidates[0]: state},
                active_counts=active_counts, log=log,
            )
        errors: list[str] = []
        try:
            pool_payload = runner([
                "osmo", "pool", "list", "--pool", *candidates,
                "--mode", "free", "--format-type", "json",
            ])
            states = _parse_pool_rows(pool_payload, candidates)
        except Exception as exc:
            states = {}
            errors.append(f"combined pool query: {exc}")
            for pool in candidates:
                try:
                    payload = runner([
                        "osmo", "pool", "list", "--pool", pool,
                        "--mode", "free", "--format-type", "json",
                    ])
                    states.update(_parse_pool_rows(payload, (pool,)))
                except Exception as pool_exc:
                    errors.append(f"{pool} pool query: {pool_exc}")

        # Unlike ``pool list``, current OSMO ``resource list`` accepts
        # multiple pool values but returns nodes for only the first one.
        # Query each candidate explicitly, and retain a healthy pool when its
        # peer's resource endpoint is unavailable.
        for pool in tuple(states):
            try:
                resource_payload = runner([
                    "osmo", "resource", "list", "--pool", pool,
                    "--mode", "free", "--format-type", "json",
                ])
                _parse_resource_rows(resource_payload, states)
            except Exception as exc:
                errors.append(f"{pool} resource query: {exc}")
                states.pop(pool, None)
        if states:
            for error in errors:
                log(f"WARNING: partial OSMO pool capacity query failure: {error}")
            return cls(
                candidates, fallback, states,
                active_counts=active_counts, log=log,
            )
        message = " ".join("; ".join(errors).split())[:500]
        if not message:
            message = "capacity query returned no configured pools"
        log(
            "WARNING: OSMO pool capacity query failed; "
            f"falling back to {fallback}: {message}"
        )
        return cls(
            candidates, fallback, {}, query_error=message,
            active_counts=active_counts, log=log,
        )

    def _scores(
        self, workload: str, profile: WorkloadProfile,
    ) -> dict[str, dict[str, object]]:
        scores: dict[str, dict[str, object]] = {}
        for order, pool in enumerate(self.candidates):
            state = self.states.get(pool)
            if state is None:
                scores[pool] = {
                    "eligible": False, "capable": False,
                    "reason": "missing_pool", "order": order,
                }
                continue
            free_slots = {
                item.name: _slots_for_class(state.nodes, item)
                for item in profile.resource_classes
            }
            capable_slots = {
                item.name: _slots_for_class(state.capability_nodes, item)
                for item in profile.resource_classes
            }
            capable = all(value > 0 for value in capable_slots.values())
            capacity = free_slots[profile.primary_resource]
            if profile.mode == "gpu":
                capacity = min(
                    capacity,
                    state.quota_free // max(1, profile.primary.gpu),
                )
            active = int(self.active_counts.get(workload, {}).get(pool, 0))
            self.initial_capacities.setdefault((workload, pool), capacity)
            scores[pool] = {
                "eligible": state.status == "ONLINE" and capable and capacity > 0,
                "capable": capable,
                "status": state.status,
                "capacity": capacity,
                "initial_capacity": self.initial_capacities[(workload, pool)],
                "primary_resource": profile.primary_resource,
                "resource_slots": free_slots,
                "capability_slots": capable_slots,
                "quota_free": state.quota_free,
                "total_free_gpu": state.total_free,
                "free_cpu": int(sum(node["cpu"] for node in state.nodes)),
                "utilization_milli": round(state.utilization * 1000),
                "active_workflows": active,
                "order": order,
            }
        return scores

    def choose(
        self,
        workload: str,
        *,
        workflow_path: str | Path | None = None,
        workflow_text: str | None = None,
        override: str | None = None,
    ) -> PoolDecision:
        """Select and locally debit capacity as one thread-safe operation."""
        with self._lock:
            return self._choose_unlocked(
                workload,
                workflow_path=workflow_path,
                workflow_text=workflow_text,
                override=override,
            )

    def _choose_unlocked(
        self,
        workload: str,
        *,
        workflow_path: str | Path | None = None,
        workflow_text: str | None = None,
        override: str | None = None,
    ) -> PoolDecision:
        # Fixed and explicitly overridden routing does not depend on resource
        # discovery. This also keeps legacy single-pool callers compatible with
        # generated or test workflow paths that do not exist on the submit host.
        if override:
            if override not in self.candidates:
                raise ValueError(
                    f"Pool override {override!r} is not configured; "
                    f"choose one of {', '.join(self.candidates)}"
                )
            decision = PoolDecision(
                override, workload, "override", "explicit_override", {},
            )
            self._reserve_without_profile(decision)
            return decision
        if self.query_error:
            decision = PoolDecision(
                self.fallback, workload, "fallback",
                f"capacity_query_error_fallback: {self.query_error}", {},
            )
            self._reserve_without_profile(decision)
            return decision
        if len(self.candidates) == 1:
            decision = PoolDecision(
                self.candidates[0], workload, "fixed",
                "single_configured_pool", {},
            )
            self._reserve_without_profile(decision)
            return decision
        if (workflow_path is None) == (workflow_text is None):
            raise ValueError("provide exactly one of workflow_path or workflow_text")
        profile = (
            workload_profile_from_path(workflow_path, name=workload)
            if workflow_path is not None
            else workload_profile_from_text(str(workflow_text), name=workload)
        )
        scores = self._scores(workload, profile)
        eligible = [pool for pool in self.candidates if scores[pool]["eligible"]]
        if eligible:
            if profile.mode == "gpu":
                key = lambda pool: (
                    int(scores[pool]["capacity"]),
                    int(scores[pool]["total_free_gpu"]),
                    -int(scores[pool]["order"]),
                )
                reason = "most_schedulable_gpu_slots"
            else:
                key = lambda pool: (
                    int(scores[pool]["capacity"]),
                    int(scores[pool]["free_cpu"]),
                    -int(scores[pool]["order"]),
                )
                reason = "most_schedulable_cpu_slots"
            selected = max(eligible, key=key)
        else:
            capable = [
                pool for pool in self.candidates
                if scores[pool].get("status") == "ONLINE"
                and scores[pool].get("capable")
            ]
            if not capable:
                raise PoolSelectionError(
                    f"No online configured pool can fit every resource class for "
                    f"{workload}; scores={json.dumps(scores, sort_keys=True)}"
                )
            else:
                tie_metric = (
                    "total_free_gpu" if profile.mode == "gpu" else "free_cpu"
                )
                selected = max(
                    capable,
                    key=lambda pool: (
                        int(scores[pool]["initial_capacity"]),
                        int(scores[pool][tie_metric]),
                        -int(scores[pool]["order"]),
                    ),
                )
                reason = "queued_overflow_best_initial_capacity"
        decision = PoolDecision(selected, workload, profile.mode, reason, scores)
        self._reserve(decision, profile)
        self.log(
            f"  Pool selection: {workload} -> {selected} ({reason}); "
            f"scores={json.dumps(scores, sort_keys=True)}"
        )
        return decision

    def _reserve(self, decision: PoolDecision, profile: WorkloadProfile) -> None:
        state = self.states.get(decision.pool)
        if state is not None:
            demand = profile.primary
            fitting = [node for node in state.nodes if _node_fits(node, demand)]
            if fitting:
                node = min(
                    fitting,
                    key=lambda item: (
                        item["gpu"] - demand.gpu,
                        item["cpu"] - demand.cpu,
                        item["memory"] - demand.memory_gib * 1024**3,
                    ),
                )
                node["cpu"] -= demand.cpu
                node["gpu"] -= demand.gpu
                node["memory"] -= demand.memory_gib * 1024**3
                node["storage"] -= demand.storage_gib * 1024**3
                if demand.gpu:
                    state.quota_free = max(0, state.quota_free - demand.gpu)
        by_pool = self.active_counts.setdefault(decision.workload, {})
        by_pool[decision.pool] = int(by_pool.get(decision.pool, 0)) + 1

    def _reserve_without_profile(self, decision: PoolDecision) -> None:
        by_pool = self.active_counts.setdefault(decision.workload, {})
        by_pool[decision.pool] = int(by_pool.get(decision.pool, 0)) + 1
