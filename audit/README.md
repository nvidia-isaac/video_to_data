# License Audit

This directory contains the infrastructure for auditing the repo's licensing
posture. The audit's job is **visibility, not enforcement** — it describes what
licenses and restrictions actually exist in the tree, so developers know what
they're working with and users know which paths are commercial vs research.

The repo is allowed to contain academic-licensed code. The audit just makes
the posture visible.

## What lives here

```
audit/
  audit_structure.yaml   The audit's knowledge base — license catalog,
                         taint vector definitions, ignore paths. Stable
                         knowledge only; anything that drifts is discovered
                         at audit time, not declared here.
  scripts/
    validate_structure.py  Pre-flight: confirms the structure file is
                           internally consistent and that the catalog
                           covers every LICENSE in the repo.
    check.py               Discovery: walks the repo, classifies LICENSE
                           files, scans imports and deps for tainted
                           upstream packages, emits structured JSON.
  reports/
    latest.md              Most recent audit report (committed)
    YYYY-MM-DD-<sha>.md    Archived reports — history is the change log
  README.md              You are here.
```

Two Claude skills (in `.claude/skills/`) orchestrate these scripts:

- `validate-audit-structure` — runs the pre-flight; useful standalone when you suspect catalog drift.
- `license-audit` — runs validate, then check, then synthesizes the markdown report.

## How to run an audit

From inside Claude Code, invoke the skill:

```
/license-audit
```

Or run the scripts directly:

```bash
python3 audit/scripts/validate_structure.py     # pre-flight
python3 audit/scripts/check.py                  # discovery, JSON to stdout
```

The skill is the recommended path — it handles the validate → discover →
synthesize → write-report sequence and produces the markdown deliverable.

## Reproduce this audit (for reviewers)

Follow these steps to confirm the system works against the current tree.
Expected output snippets are shown after each command; if your output differs
materially, something has drifted.

### 1. Validate the audit structure passes

```bash
python3 audit/scripts/validate_structure.py
```

Expected tail:
```
Modules discovered (3):
  - reconstruction
  - robotic_grounding
  - video_ingestion_agent

LICENSE files discovered (4):
  - video_ingestion_agent/LICENSE                                 → apache2
  - reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/LICENSE  → nvidia_sec_3_3
  - reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda/torch_ngp_grid_encoder/LICENSE  → mit
  - reconstruction/modules/v2d_nlf/lib/lib_smpl/smplpytorch/LICENSE  → gpl3

RESULT: OK — audit can proceed
```

If you see `RESULT: FAIL`, follow the resolution in the error message — most
likely a LICENSE file has no matching fingerprint in the catalog.

### 2. Run discovery

```bash
python3 audit/scripts/check.py --output /tmp/audit_discovery.json
```

This should print `wrote /tmp/audit_discovery.json` and exit 0. Inspect:

```bash
python3 -c "
import json
d = json.load(open('/tmp/audit_discovery.json'))
print('modules:', [m['id'] for m in d['modules']])
print('licenses_seen:', d['licenses_seen'])
print('taint_vectors_active:', d['taint_vectors_active'])
for m in d['modules']:
    print(m['id'], m['status'])
"
```

Expected:
```
modules: ['reconstruction', 'robotic_grounding', 'video_ingestion_agent']
licenses_seen: ['apache2', 'gpl3', 'mit', 'nvidia_sec_3_3']
taint_vectors_active: ['meta_sam_3d', 'nc_depth_models', 'nvidia_sec_3_3', 'smpl_family']
reconstruction red
robotic_grounding red
video_ingestion_agent green
```

### 3. Stress-test validation catches a broken catalog

Confirm the maintenance loop is functional. Edit `audit/audit_structure.yaml`,
break Apache 2.0's fingerprints temporarily:

```yaml
  apache2:
    ...
    fingerprints:
      - "XXX_DOES_NOT_MATCH_ANYTHING_XXX"
```

Re-run validate:

```bash
python3 audit/scripts/validate_structure.py
```

Expected: `RESULT: FAIL` with an error pointing at
`video_ingestion_agent/LICENSE` as unclassified, plus a `suggestion:` block
quoting the first lines of that file. Revert the change after.

### 4. Confirm the latest report exists and is current

```bash
ls -la audit/reports/
head -20 audit/reports/latest.md
```

`latest.md` should exist as a copy of the most recent dated report.
It should start with `# License Audit Report — <date>` and contain the
status legend (🟢 🟡 🔴 ⚪) within the first 30 lines.

## Requirements

- Python 3.11+ (uses `tomllib` from stdlib)
- PyYAML

## How to read a report

Reports describe **what is**, not what should be:

- **Per module:** which LICENSE files exist, what they classify as, what the
  module's `pyproject.toml` declares, which restrictive upstream packages it
  uses (with file:line citations).
- **Cross-cutting:** licenses found across the repo, taint vectors with any
  exposure, unclassified LICENSE files.

The report does not recommend actions. Decisions about restructuring, license
updates, or partner clarifications happen elsewhere (PRs, separate review
docs) — not in this audit.

## Maintaining `audit_structure.yaml`

The structure file holds only stable knowledge:

- **License catalog** — every license the audit might encounter, with text
  fingerprints used for classification.
- **Taint vector definitions** — which restrictive licenses propagate, and
  which upstream package names trigger them.
- **Ignore paths** — top-level dirs that are not modules.

Module lists, per-module postures, and "which modules touch which taint" are
**discovered at audit time** — never declared here.

When does the file get updated?

| Trigger | Action |
|---|---|
| `validate_structure.py` reports a LICENSE file with no matching fingerprint | Read the file, add the appropriate license entry (or new fingerprint to an existing entry) |
| A new restrictive upstream package is being introduced | Add it to an existing taint vector's `upstream_packages`, or create a new taint vector |
| A new top-level dir should be ignored by the audit | Add it to `ignore_paths` |
| Schema needs change | Bump `schema_version`, update the validate/check scripts accordingly |

Validation refuses to greenlight an audit if the catalog can't classify a
LICENSE file in the repo. Fix the catalog first; the audit's report depends
on accurate classification.

## Picking license fingerprints

A good fingerprint is a substring that appears in the target license's text
but is unlikely to appear in any other license. Avoid generic phrases
("Copyright (c)", "All rights reserved"). Prefer:

- The license's verbatim title (`"Apache License, Version 2.0"`)
- An SPDX identifier (`"SPDX-License-Identifier: Apache-2.0"`)
- A distinctive clause unique to that license (`"used or intended for use non-commercially"`)

## Philosophy

- The audit is **descriptive**, not prescriptive.
- The audit's knowledge base is **stable**; everything else is **discovered**.
- Findings are **facts with citations**, not inferences.
- The catalog is **extensible**: when the audit can't classify something, it
  asks for the catalog to be updated rather than guessing.
- Reports accumulate over time; **diffing reports is how drift is detected**.

## Out of scope (for now)

- CI integration (PR-time checks, post-merge triggers, scheduled runs)
- Private storage for sensitive/strategic content
- Notifications (GitHub issues, Slack)

These are deliberate omissions for the local-only MVP. Once the core audit
loop is validated, the CI layer can be wired on top of these scripts.
