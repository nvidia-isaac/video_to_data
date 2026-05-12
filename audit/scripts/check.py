"""Discover what licenses, dependencies, and taint exposure exist in the repo.

This script DESCRIBES the repo's current licensing state. It does not compare
against declarations — declarations belong elsewhere or nowhere. The output
is structured JSON consumed by the license-audit skill to synthesize a report.

Pre-flight: invokes validate_structure first. If validation fails, check
refuses to run.

Usage:
    python check.py [--output FILE] [--repo-root PATH] [--structure PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import validate_structure as v


# -----------------------------------------------------------------------------

PYPROJECT = "pyproject.toml"
REQ_GLOBS = ("requirements*.txt",)
PY_GLOB = "**/*.py"
DEP_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")  # extracts package name from a spec like "torch>=2.0,<3"
IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z0-9_.]+)|import\s+([A-Za-z0-9_.,\s]+))", re.MULTILINE)


@dataclass
class TaintHit:
    vector_id: str
    upstream_package: str
    file: str
    line: int
    kind: str  # "import" | "dependency"


@dataclass
class ModuleFinding:
    id: str
    path: str
    license_files: list[dict] = field(default_factory=list)        # [{path, classification}]
    pyproject_license: str | None = None
    pyproject_path: str | None = None
    has_module_root_license: bool = False
    runtime_deps: list[str] = field(default_factory=list)          # raw dep spec strings (top-level only)
    nested_pyprojects: list[str] = field(default_factory=list)     # other pyproject.toml under this module
    taint_hits: list[TaintHit] = field(default_factory=list)
    status: str = "unknown"                                        # "green" | "yellow" | "red"
    status_reasons: list[str] = field(default_factory=list)        # human-readable reasons
    output_inheritance: list[dict] = field(default_factory=list)   # [{license_id, commercial, source: taint_vector_id|module_root}]
    notes: list[str] = field(default_factory=list)


@dataclass
class AuditDiscovery:
    timestamp: str
    repo_root: str
    structure_path: str
    structure_version: int
    schema_version: int
    modules: list[ModuleFinding] = field(default_factory=list)
    licenses_seen: list[str] = field(default_factory=list)
    unclassified_license_files: list[str] = field(default_factory=list)
    taint_vectors_active: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Per-module discovery

def find_license_files_in(root: Path, repo_root: Path) -> list[Path]:
    results: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.name in v.LICENSE_FILENAMES:
            if any(part.startswith(".") for part in path.relative_to(repo_root).parts):
                continue
            results.append(path)
    return results


def parse_pyproject(path: Path) -> tuple[str | None, list[str]]:
    """Return (license-declaration-string, top-level-dependencies)."""
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None, []
    project = data.get("project") or {}
    lic = project.get("license")
    if isinstance(lic, dict):
        lic_str = lic.get("text") or lic.get("file")
    elif isinstance(lic, str):
        lic_str = lic
    else:
        lic_str = None
    deps = list(project.get("dependencies") or [])
    return lic_str, deps


def dep_name(spec: str) -> str:
    m = DEP_SPEC_RE.match(spec)
    return m.group(1).lower().replace("-", "_") if m else spec.lower()


def scan_imports(py_file: Path, packages: set[str]) -> list[tuple[str, int]]:
    """Return list of (matched_package, line_number) for any package in `packages`."""
    try:
        text = py_file.read_text(errors="replace")
    except OSError:
        return []
    hits: list[tuple[str, int]] = []
    for m in IMPORT_RE.finditer(text):
        names_field = m.group(1) or m.group(2) or ""
        for raw in re.split(r"[,\s]+", names_field):
            top = raw.split(".")[0].strip()
            top_norm = top.lower().replace("-", "_")
            if top_norm in packages:
                line_no = text[:m.start()].count("\n") + 1
                hits.append((top_norm, line_no))
    return hits


def build_taint_index(taint_vectors: list[dict]) -> dict[str, list[str]]:
    """Map normalized upstream package name -> list of taint vector IDs that cover it."""
    idx: dict[str, list[str]] = {}
    for tv in taint_vectors:
        for pkg in tv.get("upstream_packages") or []:
            key = pkg.lower().replace("-", "_")
            idx.setdefault(key, []).append(tv["id"])
    return idx


# -----------------------------------------------------------------------------
# Status computation
#
# Rule (documented; not declared in YAML for MVP):
#
#   RED    — any active taint vector references a license whose `commercial`
#            status is `not_allowed`.
#   YELLOW — any active taint vector references a license whose `commercial`
#            status is one of {restrictive, needs_legal_review, mixed, gated,
#            research_only_outside_nvidia}. Also: module has no LICENSE file
#            at module root.
#   GREEN  — no taint exposure AND module has a LICENSE file at its root
#            classified as permissive (commercial: ok).
# -----------------------------------------------------------------------------

RED_COMMERCIAL = {"not_allowed"}
YELLOW_COMMERCIAL = {
    "restrictive", "needs_legal_review", "mixed", "gated",
    "research_only_outside_nvidia", "ok_with_restrictions",
}


def compute_module_status(
    finding: ModuleFinding,
    catalog: dict,
    taint_vectors: list[dict],
) -> tuple[str, list[str], list[dict]]:
    """Return (status, reasons, output_inheritance)."""
    reasons: list[str] = []
    output_inheritance: list[dict] = []
    tv_by_id = {tv["id"]: tv for tv in taint_vectors}

    # Module-root license
    module_root_lic_id: str | None = None
    for lf in finding.license_files:
        # module-root LICENSE means LICENSE directly under the module path
        # (no extra path segments between module and the LICENSE file)
        rel = lf["path"]
        # finding.path is e.g. "video_ingestion_agent/"
        if rel.startswith(finding.path) and "/" not in rel[len(finding.path):].rstrip("/"):
            # Wait — finding.path ends with /, and rel[len(finding.path):] should be just the filename
            tail = rel[len(finding.path):]
            if tail in {"LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.LESSER"}:
                module_root_lic_id = lf["classification"]
                break

    finding.has_module_root_license = module_root_lic_id is not None

    # If module-root license exists, it contributes to output inheritance
    if module_root_lic_id:
        lic = catalog.get(module_root_lic_id, {})
        output_inheritance.append({
            "license_id": module_root_lic_id,
            "commercial": lic.get("commercial"),
            "source": "module_root_license",
        })

    # Each active taint vector contributes
    active_vector_ids = {h.vector_id for h in finding.taint_hits}
    has_red = False
    has_yellow = False
    for tv_id in sorted(active_vector_ids):
        tv = tv_by_id.get(tv_id, {})
        lic_id = tv.get("license")
        lic = catalog.get(lic_id, {}) if lic_id else {}
        commercial = lic.get("commercial")
        output_inheritance.append({
            "license_id": lic_id,
            "commercial": commercial,
            "source": f"taint_vector:{tv_id}",
        })
        if commercial in RED_COMMERCIAL:
            has_red = True
            reasons.append(f"taint vector {tv_id} carries {lic_id} (commercial: {commercial})")
        elif commercial in YELLOW_COMMERCIAL:
            has_yellow = True
            reasons.append(f"taint vector {tv_id} carries {lic_id} (commercial: {commercial})")

    # Module-level structural concerns
    if not finding.has_module_root_license:
        has_yellow = True
        reasons.append("no LICENSE file at module root (nested LICENSE files may exist)")

    if has_red:
        status = "red"
    elif has_yellow:
        status = "yellow"
    else:
        status = "green"
        reasons.append("no taint exposure; module-root LICENSE present and permissive")

    return status, reasons, output_inheritance


def discover_module(
    module_id: str,
    module_path: Path,
    repo_root: Path,
    catalog: dict,
    taint_index: dict[str, list[str]],
) -> ModuleFinding:
    finding = ModuleFinding(id=module_id, path=str(module_path.relative_to(repo_root)) + "/")

    # LICENSE files
    for lic_path in find_license_files_in(module_path, repo_root):
        try:
            text = lic_path.read_text(errors="replace")
        except OSError as e:
            finding.notes.append(f"could not read LICENSE at {lic_path.relative_to(repo_root)}: {e}")
            continue
        cls = v.classify_license(text, catalog)
        finding.license_files.append({
            "path": str(lic_path.relative_to(repo_root)),
            "classification": cls,
        })

    # pyproject.toml at module root + any nested
    root_pp = module_path / PYPROJECT
    if root_pp.exists():
        lic_str, deps = parse_pyproject(root_pp)
        finding.pyproject_license = lic_str
        finding.pyproject_path = str(root_pp.relative_to(repo_root))
        finding.runtime_deps = deps
    for nested_pp in module_path.rglob(PYPROJECT):
        if nested_pp == root_pp:
            continue
        finding.nested_pyprojects.append(str(nested_pp.relative_to(repo_root)))

    # Taint hits — dependencies
    tainted_pkgs = set(taint_index.keys())
    seen_dep_hits: set[tuple[str, str]] = set()
    for spec in finding.runtime_deps:
        name = dep_name(spec)
        if name in tainted_pkgs:
            for tv_id in taint_index[name]:
                key = (tv_id, name)
                if key in seen_dep_hits:
                    continue
                seen_dep_hits.add(key)
                finding.taint_hits.append(TaintHit(
                    vector_id=tv_id, upstream_package=name,
                    file=finding.pyproject_path or "", line=0, kind="dependency",
                ))

    # Taint hits — imports
    for py_file in module_path.rglob("*.py"):
        if any(part.startswith(".") for part in py_file.relative_to(repo_root).parts):
            continue
        for pkg, line in scan_imports(py_file, tainted_pkgs):
            for tv_id in taint_index[pkg]:
                finding.taint_hits.append(TaintHit(
                    vector_id=tv_id, upstream_package=pkg,
                    file=str(py_file.relative_to(repo_root)), line=line, kind="import",
                ))

    return finding


# -----------------------------------------------------------------------------
# Main

def run(structure_path: Path, repo_root: Path) -> AuditDiscovery:
    # Pre-flight validate
    validation = v.run(structure_path, repo_root)
    if not validation.can_proceed:
        sys.stderr.write(v.render_human(validation) + "\n")
        sys.stderr.write("\nrefusing to run discovery — fix structure errors first.\n")
        sys.exit(1)

    structure, _ = v.load_structure(structure_path)
    assert structure is not None
    catalog = structure.get("licenses") or {}
    taint_vectors = structure.get("taint_vectors") or []
    ignore_paths = structure.get("ignore_paths") or []
    taint_index = build_taint_index(taint_vectors)

    discovery = AuditDiscovery(
        timestamp=datetime.now(timezone.utc).isoformat(),
        repo_root=str(repo_root),
        structure_path=str(structure_path),
        structure_version=structure.get("structure_version", 0),
        schema_version=structure.get("schema_version", 0),
    )

    module_names = v.discover_modules(repo_root, ignore_paths)
    licenses_seen: set[str] = set()
    unclassified: list[str] = []
    taint_active: set[str] = set()

    for name in module_names:
        finding = discover_module(name, repo_root / name, repo_root, catalog, taint_index)
        # Compute status and output inheritance now that taint hits are known
        status, reasons, inheritance = compute_module_status(finding, catalog, taint_vectors)
        finding.status = status
        finding.status_reasons = reasons
        finding.output_inheritance = inheritance
        discovery.modules.append(finding)
        for lf in finding.license_files:
            if lf["classification"]:
                licenses_seen.add(lf["classification"])
            else:
                unclassified.append(lf["path"])
        for hit in finding.taint_hits:
            taint_active.add(hit.vector_id)

    discovery.licenses_seen = sorted(licenses_seen)
    discovery.unclassified_license_files = unclassified
    discovery.taint_vectors_active = sorted(taint_active)
    return discovery


def to_jsonable(obj):
    if isinstance(obj, AuditDiscovery):
        d = asdict(obj)
        return d
    return asdict(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--structure", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON to file instead of stdout")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    structure_path = (args.structure or (script_dir.parent / "audit_structure.yaml")).resolve()
    repo_root = (args.repo_root or script_dir.parent.parent).resolve()

    discovery = run(structure_path, repo_root)
    payload = json.dumps(to_jsonable(discovery), indent=2)
    if args.output:
        args.output.write_text(payload)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
