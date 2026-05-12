"""Validate audit_structure.yaml is in shape for the audit to run.

Checks the audit's knowledge base is internally consistent and that the
catalog covers what's actually in the repo. Refuses to greenlight an audit
if the catalog can't classify a LICENSE file it finds — that means the
catalog needs an update before the report would be meaningful.

Usage:
    python validate_structure.py [--json] [--repo-root PATH]

Exit codes:
    0 — structure is valid, audit can proceed
    1 — structure has errors that must be fixed before auditing
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: uv pip install pyyaml", file=sys.stderr)
    sys.exit(2)


# -----------------------------------------------------------------------------

REQUIRED_TOP_KEYS = {"schema_version", "structure_version", "licenses", "taint_vectors", "ignore_paths"}
SUPPORTED_SCHEMA = {1}
LICENSE_FILENAMES = {"LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.LESSER"}

# Canonical set of values for the `commercial:` field on each license entry.
# This list is the source of truth for what values check.py knows how to map
# to status badges. Adding a new value here requires updating check.py's
# RED_COMMERCIAL / YELLOW_COMMERCIAL sets accordingly.
KNOWN_COMMERCIAL_VALUES = {
    "ok",                            # 🟢 — fully commercial-friendly
    "restrictive",                   # 🟡 — e.g., copyleft / share-alike
    "mixed",                         # 🟡 — multi-license datasets (HOT3D etc.)
    "gated",                         # 🟡 — access-gated, commercial status separate
    "ok_with_restrictions",          # 🟡 — commercial OK but with AUP / use-policy
    "research_only_outside_nvidia",  # 🟡 — NVIDIA Sec 3.3
    "needs_legal_review",            # 🟡 — uncertain, must escalate
    "not_allowed",                   # 🔴 — hard non-commercial
}


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    check: str
    message: str
    path: str | None = None
    suggestion: str | None = None


@dataclass
class Result:
    structure_path: str
    issues: list[Issue] = field(default_factory=list)
    discovered_modules: list[str] = field(default_factory=list)
    discovered_licenses: list[dict] = field(default_factory=list)  # [{path, classification}]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def can_proceed(self) -> bool:
        return not self.errors


# -----------------------------------------------------------------------------
# Loading and schema

def load_structure(path: Path) -> tuple[dict | None, list[Issue]]:
    if not path.exists():
        return None, [Issue("error", "load", f"audit_structure.yaml not found at {path}")]
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return None, [Issue("error", "yaml_parse", f"YAML parse error: {e}", path=str(path))]
    if not isinstance(data, dict):
        return None, [Issue("error", "schema", "top-level YAML must be a mapping", path=str(path))]
    return data, []


def validate_schema(structure: dict) -> list[Issue]:
    issues: list[Issue] = []
    missing = REQUIRED_TOP_KEYS - structure.keys()
    if missing:
        issues.append(Issue("error", "schema", f"missing required keys: {sorted(missing)}"))

    sv = structure.get("schema_version")
    if sv not in SUPPORTED_SCHEMA:
        issues.append(Issue(
            "error", "schema_version",
            f"schema_version={sv!r} unsupported (this script handles {sorted(SUPPORTED_SCHEMA)})",
        ))

    for lic_id, lic in (structure.get("licenses") or {}).items():
        if not isinstance(lic, dict):
            issues.append(Issue("error", "license_entry", f"license {lic_id!r} is not a mapping"))
            continue
        for required in ("name", "category", "commercial", "fingerprints"):
            if required not in lic:
                issues.append(Issue(
                    "error", "license_entry",
                    f"license {lic_id!r} missing required field {required!r}",
                ))
        if not (lic.get("fingerprints") or []):
            issues.append(Issue(
                "error", "license_entry",
                f"license {lic_id!r} has no fingerprints — audit cannot classify LICENSE files against it",
            ))
        # Warn (not error) on unknown commercial values — catches typos like
        # `not_alllowed` that would otherwise fall through to 🟢 silently in check.py.
        commercial = lic.get("commercial")
        if commercial is not None and commercial not in KNOWN_COMMERCIAL_VALUES:
            issues.append(Issue(
                "warning", "commercial_value",
                f"license {lic_id!r} has unrecognized commercial value {commercial!r}",
                suggestion=(
                    f"known values are {sorted(KNOWN_COMMERCIAL_VALUES)}. "
                    "If this is intentional, add the value to KNOWN_COMMERCIAL_VALUES "
                    "in validate_structure.py and update check.py's RED/YELLOW sets."
                ),
            ))

    for tv in structure.get("taint_vectors") or []:
        if not isinstance(tv, dict):
            issues.append(Issue("error", "taint_vector", "taint vector is not a mapping"))
            continue
        for required in ("id", "license", "description", "upstream_packages"):
            if required not in tv:
                issues.append(Issue(
                    "error", "taint_vector",
                    f"taint vector {tv.get('id', '?')!r} missing required field {required!r}",
                ))

    return issues


def validate_internal_refs(structure: dict) -> list[Issue]:
    issues: list[Issue] = []
    license_ids = set((structure.get("licenses") or {}).keys())
    seen_tv_ids: set[str] = set()
    for tv in structure.get("taint_vectors") or []:
        tv_id = tv.get("id")
        if tv_id in seen_tv_ids:
            issues.append(Issue("error", "taint_vector", f"duplicate taint_vector id {tv_id!r}"))
        seen_tv_ids.add(tv_id)
        ref = tv.get("license")
        if ref and ref not in license_ids:
            issues.append(Issue(
                "error", "internal_ref",
                f"taint_vector {tv_id!r} references unknown license {ref!r}",
                suggestion=f"add {ref!r} to licenses: catalog, or fix the reference",
            ))
    return issues


# -----------------------------------------------------------------------------
# Discovery

def discover_modules(repo_root: Path, ignore_paths: list[str]) -> list[str]:
    ignore_names = {p.rstrip("/").strip("/") for p in ignore_paths}
    modules: list[str] = []
    for child in sorted(repo_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in ignore_names or child.name.startswith("."):
            continue
        modules.append(child.name)
    return modules


def discover_license_files(repo_root: Path, ignore_paths: list[str]) -> list[Path]:
    ignore_names = {p.rstrip("/").strip("/") for p in ignore_paths}
    found: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name not in LICENSE_FILENAMES:
            continue
        if any(part in ignore_names or part.startswith(".") for part in path.relative_to(repo_root).parts):
            continue
        found.append(path)
    return found


def classify_license(text: str, catalog: dict) -> str | None:
    for lic_id, lic in catalog.items():
        for fp in lic.get("fingerprints") or []:
            if fp in text:
                return lic_id
    return None


def validate_license_coverage(structure: dict, repo_root: Path) -> tuple[list[Issue], list[dict]]:
    issues: list[Issue] = []
    classifications: list[dict] = []
    catalog = structure.get("licenses") or {}
    ignore = structure.get("ignore_paths") or []
    for lic_path in discover_license_files(repo_root, ignore):
        try:
            text = lic_path.read_text(errors="replace")
        except OSError as e:
            issues.append(Issue("warning", "license_read", f"could not read {lic_path}: {e}"))
            continue
        cls = classify_license(text, catalog)
        rel = str(lic_path.relative_to(repo_root))
        classifications.append({"path": rel, "classification": cls})
        if cls is None:
            head = "\n".join(text.splitlines()[:5])
            issues.append(Issue(
                "error", "license_coverage",
                f"LICENSE file at {rel} does not match any fingerprint in catalog",
                path=rel,
                suggestion=(
                    "add a new entry to licenses: in audit_structure.yaml with a "
                    f"fingerprint that matches this file. First lines:\n{head}"
                ),
            ))
    return issues, classifications


# -----------------------------------------------------------------------------
# Main

def run(structure_path: Path, repo_root: Path) -> Result:
    result = Result(structure_path=str(structure_path))
    structure, load_issues = load_structure(structure_path)
    result.issues.extend(load_issues)
    if structure is None:
        return result

    result.issues.extend(validate_schema(structure))
    if result.errors:
        return result

    result.issues.extend(validate_internal_refs(structure))
    cov_issues, classifications = validate_license_coverage(structure, repo_root)
    result.issues.extend(cov_issues)
    result.discovered_licenses = classifications
    result.discovered_modules = discover_modules(repo_root, structure.get("ignore_paths") or [])
    return result


def render_human(result: Result) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Audit structure validation")
    lines.append("=" * 70)
    lines.append(f"structure: {result.structure_path}")
    lines.append("")

    if result.discovered_modules:
        lines.append(f"Modules discovered ({len(result.discovered_modules)}):")
        for m in result.discovered_modules:
            lines.append(f"  - {m}")
        lines.append("")

    if result.discovered_licenses:
        lines.append(f"LICENSE files discovered ({len(result.discovered_licenses)}):")
        for entry in result.discovered_licenses:
            cls = entry["classification"] or "UNCLASSIFIED"
            lines.append(f"  - {entry['path']:60s}  → {cls}")
        lines.append("")

    if result.errors:
        lines.append(f"ERRORS ({len(result.errors)}) — audit cannot proceed:")
        for i in result.errors:
            lines.append(f"  [{i.check}] {i.message}")
            if i.suggestion:
                lines.append(f"    → {i.suggestion}")
        lines.append("")

    if result.warnings:
        lines.append(f"Warnings ({len(result.warnings)}):")
        for i in result.warnings:
            lines.append(f"  [{i.check}] {i.message}")
        lines.append("")

    lines.append("RESULT: " + ("OK — audit can proceed" if result.can_proceed else "FAIL — fix errors before auditing"))
    return "\n".join(lines)


def render_json(result: Result) -> str:
    payload = {
        "structure_path": result.structure_path,
        "can_proceed": result.can_proceed,
        "discovered_modules": result.discovered_modules,
        "discovered_licenses": result.discovered_licenses,
        "issues": [asdict(i) for i in result.issues],
    }
    return json.dumps(payload, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="Repo root (default: parent of audit/ dir containing this script)")
    parser.add_argument("--structure", type=Path, default=None,
                        help="Path to audit_structure.yaml (default: ../audit_structure.yaml)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    structure_path = args.structure or (script_dir.parent / "audit_structure.yaml")
    repo_root = args.repo_root or script_dir.parent.parent

    result = run(structure_path.resolve(), repo_root.resolve())
    print(render_json(result) if args.json else render_human(result))
    return 0 if result.can_proceed else 1


if __name__ == "__main__":
    raise SystemExit(main())
