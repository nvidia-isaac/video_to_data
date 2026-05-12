---
name: validate-audit-structure
description: Pre-flight validator for the license audit's knowledge base (audit/audit_structure.yaml). Confirms the catalog can classify every LICENSE file in the repo, the YAML is well-formed, and internal references are consistent. Runs before any license audit, or standalone when you suspect structure drift. Use when the audit refuses to start, when adding/renaming top-level dirs, when a new LICENSE file enters the tree, or when a new restrictive upstream dependency is added.
---

# Validate Audit Structure

The license audit consults `audit/audit_structure.yaml` as its knowledge base
(license catalog, taint vectors, ignored paths). If that file is internally
inconsistent or doesn't cover what's actually in the repo, the audit's report
will be wrong. This skill catches that before any audit runs.

## When to invoke

- **Before running the license audit** — the audit will call this automatically; you can also run it directly to debug.
- **After adding a new top-level directory** — gets surfaced as a new "discovered module."
- **After adding a new restrictive upstream dependency** — confirms a taint vector covers it (currently informational only — full taint coverage check lives in `check.py`).
- **When the audit refuses to run** — its error message will point here.
- **As a standalone sanity check** — quick way to confirm the audit's worldview matches the repo's current state.

## How to run

```bash
python3 audit/scripts/validate_structure.py
```

Flags:
- `--json` — emit machine-readable JSON instead of human output
- `--repo-root PATH` — point at a different repo root (default: parent of `audit/`)
- `--structure PATH` — point at a different structure file (default: `audit/audit_structure.yaml`)

Exit codes:
- `0` — structure is valid; audit can proceed
- `1` — errors found; fix `audit_structure.yaml` before running audit

## What it checks

1. **YAML parses cleanly** and contains the required top-level keys: `schema_version`, `structure_version`, `licenses`, `taint_vectors`, `ignore_paths`.
2. **Each license entry** has `name`, `category`, `commercial`, and at least one `fingerprint`.
3. **Each taint vector** has `id`, `license`, `description`, `upstream_packages`, and references a license that exists in the catalog.
4. **LICENSE-file coverage** — every `LICENSE`, `LICENSE.txt`, `LICENSE.md`, `COPYING`, or `COPYING.LESSER` file in the repo matches at least one fingerprint in the catalog. An unmatched file is an error.
5. **Reports discovered modules** — every top-level directory not in `ignore_paths`. Informational; no failure on new modules.

## Resolution playbook

| Error | What to do |
|---|---|
| `LICENSE file at PATH does not match any fingerprint` | Read the file. Identify the license. If it's a license already in the catalog with a fingerprint that should have matched, add a more specific fingerprint to the existing entry. If it's a new license, add a new entry to `licenses:` with at least one distinctive fingerprint. |
| `taint_vector ... references unknown license ...` | Either add the license to the catalog or fix the reference in `taint_vectors:`. |
| `license ... has no fingerprints` | Add at least one fingerprint string to that license entry. |
| `missing required keys: [...]` | Add the missing top-level keys to `audit_structure.yaml`. |
| `schema_version=X unsupported` | The structure file was written for a newer schema than this script understands. Update the script or downgrade the file. |

## Picking a fingerprint

A good fingerprint is a substring that appears in the target license's text but is unlikely to appear in any other license. Avoid generic phrases ("Copyright (c)", "All rights reserved", "Redistribution and use") — they match too much. Prefer license-name verbatim or a distinctive sentence from the license body.

Example (good):
- Apache 2.0 → `"SPDX-License-Identifier: Apache-2.0"` (deterministic), `"Apache License, Version 2.0"` (verbatim title)
- NVIDIA Sec 3.3 NC → `"used or intended for use non-commercially"` (distinctive NC clause)

Example (bad):
- `"NVIDIA"` — matches Apache LICENSE files with NVIDIA copyright
- `"GPL"` — matches AGPL, LGPL, GPL-2, etc.

## Output interpretation

The human-readable output has three sections:

- **Modules discovered** — top-level dirs not in `ignore_paths`. New modules appear here automatically; no action unless you want to add one to `ignore_paths`.
- **LICENSE files discovered** — every LICENSE-like file the audit will see, with its classification. `UNCLASSIFIED` is an error.
- **ERRORS / Warnings** — issues that block or note the audit.

Final line is `RESULT: OK` or `RESULT: FAIL`.

## Relationship to the license-audit skill

The `license-audit` skill calls this skill (or the underlying script) as its first step. If validation fails, license-audit refuses to start. This skill is also useful standalone for quick sanity checks.
