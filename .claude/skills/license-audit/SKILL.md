---
name: license-audit
description: Run a license audit of this repo. Validates the audit's knowledge base, walks every module to discover what LICENSE files exist, what dependencies and imports are present, and which restrictive upstream packages each module uses. Synthesizes findings into a markdown report at audit/reports/. Use whenever asked to audit license posture, check commercial vs research paths, generate a license report, audit license boundaries, or surface license drift. The audit describes what IS — it does not enforce, gate, or recommend.
---

# License Audit

The audit's job is **visibility, not enforcement**. It describes what licenses
and license restrictions actually exist in the repo, organized by module, so
developers know what they're working with and users know which path
(commercial vs academic) a given output sits on. The repo is allowed to contain
academic code — the audit just makes the posture visible.

The audit's knowledge base lives in `audit/audit_structure.yaml`. The audit
does not declare what licenses each module *should* have — it discovers what
they *do* have, classifies via the catalog, and reports.

## When to invoke

- The user asks for a "license audit," "license check," "compliance audit," or "license report."
- The user asks "what license restrictions touch X module?"
- The user asks for a snapshot of commercial vs research paths.
- The user asks to update the audit after a merge or new dependency.

## Pipeline

```
audit_structure.yaml  ──┐
                        ├─→ validate_structure.py (pre-flight)
repo files ─────────────┘            │
                                     ↓ (passes)
                        check.py (discovery) ──→ JSON findings ──→ markdown report
```

## How to run

Execute these steps in order. Stop on any failure.

### Step 1 — Validate the audit's knowledge base

```bash
python3 audit/scripts/validate_structure.py
```

If this fails: the catalog or schema needs updating before the audit can run.
Follow the resolution playbook in the validate-audit-structure skill. **Do not
attempt to work around validation failures** — they signal that the audit's
worldview is out of sync with the repo, and the report would be wrong.

### Step 2 — Run discovery

```bash
python3 audit/scripts/check.py --output /tmp/audit_discovery.json
```

This writes a JSON document describing every LICENSE file, dependency
declaration, and tainted import the audit found, organized by module.
Discovery automatically runs validation first; if validation fails, check
refuses to proceed.

### Step 3 — Synthesize the report

Read `/tmp/audit_discovery.json` and produce a markdown report at:

```
audit/reports/YYYY-MM-DD-<short-sha>.md
```

Also update `audit/reports/latest.md` to be a copy of the new report.

The report MUST:

- **Describe what is, not what should be.** No "this module needs a LICENSE." Instead: "this module has no LICENSE file at module root; sub-paths have these classifications: ..."
- **Cite paths and line numbers for every claim.** If you say "module X imports smplpytorch," cite the file:line from the `taint_hits` array.
- **Group findings by module.** Use the module IDs from the JSON.
- **Distinguish discovered facts from inference.** If the JSON shows a fact, state it. If you are inferring something the JSON didn't surface (rare — avoid this), mark it explicitly as inferred.
- **Never invent licenses or restrictions not in the catalog.** If a module imports something not covered by any taint vector, the audit doesn't know about its license — say so explicitly.

## Report structure

Use this structure. Adapt content based on JSON findings. The canonical reference is `audit/reports/latest.md`.

### Header

```
# License Audit Report — <date> (<git short-sha>)

**Repo:** <repo>
**Branch:** <branch>
**Structure version:** <n> (schema <m>)
**Generated:** <UTC date>
**Validation:** ✓ passed (or describe failure)
```

### Status legend (always include)

Four colors with precedence 🔴 > 🟡 > ⚪ > 🟢. ⚪ is absence of knowledge, not a finding — a layer or module can be 🟢 alongside ⚪ items.

| Badge | Meaning | Commercial-field values that map here |
|:---:|---|---|
| 🟢 | commercial OK, no conditions | `ok` |
| 🟡 | conditional restrictions (known) | `restrictive`, `mixed`, `gated`, `ok_with_restrictions`, `research_only_outside_nvidia`, `needs_legal_review` |
| 🔴 | hard non-commercial | `not_allowed` |
| ⚪ | unknown — out of scope this run | (no value; used when audit didn't classify) |

The canonical commercial-value → status mapping is in `audit/scripts/check.py`
(`RED_COMMERCIAL`, `YELLOW_COMMERCIAL`). Do not restate the rule elsewhere; reference that.

Always describe the **5-layer framework** in the legend section: (1) Code, (2) Pretrained model weights, (3) Example datasets, (4) Runtime dependencies, (5) Produced datasets / artifacts.

### Executive summary — heatmap table

| Module | Module | L1 Code | L2 Weights | L3 Datasets | L4 Runtime deps | L5 Outputs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|

One row per module. Cells are badges. Layer-level badge = worst of the layer's rows (excluding ⚪).

Below the table, totals: 🟢/🟡/🔴 module counts; layers with ⚪ unknown content.

### Per-module section

Heading: `# <emoji> <STATUS> — <module_id>` (h1 within the report — visually prominent).
One-line reason for the module status.

Then **all 5 layers** for every module, in order, each with its own status badge:

```
## Layer 1: Code <badge>
## Layer 2: Pretrained model weights <badge>
## Layer 3: Example datasets <badge>
## Layer 4: Runtime dependencies <badge>
## Layer 5: Produced datasets / artifacts <badge>
```

Even if a layer is empty or out-of-scope, include the heading. Out-of-scope → ⚪.

Within each layer, include:

1. **Layer status line** — one sentence explaining the badge.
2. **Detail tables** populated from JSON findings. Each row carries a badge.
3. **Path summary table** at the bottom of the layer:

```
| Path | Status | Notes |
|---|:---:|---|
| Commercial path | <badge> | what's available / why not |
| Academic path | <badge> | what's available / why not |
| Tainted / inherited per upstream | <badge> | which upstreams propagate restrictions IN |
```

The "Tainted/inherited" row may be ⚪ or 🟢 if no relevant taint applies in that layer.

Per-layer content guidance:

- **L1 (Code):** LICENSE files table; in-tree upstream copies table; module-root metadata (pyproject path, declared license).
- **L2 (Weights):** typically ⚪ (runtime-downloaded, audit can't see).
- **L3 (Datasets):** 🟢 if no in-tree datasets; 🟡 if dataset files in tree but unclassified.
- **L4 (Runtime deps):** module-root pyproject deps table; taint exposure table with citations.
- **L5 (Outputs):** "Implied output license posture" — what license outputs inherit, from where. Logical entailment from L1 + L4.

### Trailing sections

- **Cross-cutting observations** — table of factual cross-module patterns.
- **Coverage gaps** — table of known limitations (typically ⚪).
- **Validation status** — table linking validate-run, unclassified counts, catalog verification status.

### Mandatory elements

- Status badges (🟢 🟡 🔴 ⚪) on **every fact row** in every table.
- Module-level badge in heading.
- Per-layer badge in layer heading.
- Per-layer Path summary table.
- Path/line citations for every taint claim.

## Constraints

- **No recommendations.** No "should add a LICENSE." No "consider restructuring." Findings are facts; decisions live elsewhere.
- **No partner-specific or legal-interpretive content.** This report goes in the public NVIDIA-visible repo. Keep it factual and non-strategic.
- **No claims beyond the JSON.** If you can't cite a path/line from the discovery output, don't make the claim.
- **The audit_structure.yaml is the only knowledge base.** Don't invent licenses or restrictions outside the catalog. If the audit discovers something the catalog doesn't cover, that's a validate-time failure — you shouldn't be running synthesis yet.

## After writing the report

Tell the user:
1. The report path
2. The headline numbers (modules, licenses found, taint vectors active)
3. Any unclassified LICENSE files (these indicate catalog drift — direct user to update audit_structure.yaml and re-run)

Do not push, commit, or notify anyone — that's a separate workflow concern, not the audit's job.
