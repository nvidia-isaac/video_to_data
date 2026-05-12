# License Audit Report — 2026-05-12 (65338f22)

**Repo:** nvidia-isaac/video_to_data
**Branch:** feature/agentic-audit-setup
**Structure version:** 1 (schema 1)
**Generated:** 2026-05-12 UTC
**Validation:** ✓ passed (all LICENSE files in tree classify against the catalog)

---

## Status legend

| Badge | Meaning | Examples of underlying license |
|:---:|---|---|
| 🟢 | `ok` — commercial use fine, no conditions | Apache 2.0, MIT, BSD, CC BY-SA 4.0 |
| 🟡 | conditional — usable but with restrictions | GPL-3, NVIDIA Sec 3.3, SAM License (AUP), HOT3D (multi-license) |
| 🔴 | `not_allowed` — hard non-commercial | CC BY-NC, CC BY-NC-SA, SMPL/MANO Max Planck NC |
| ⚪ | unknown — audit did not classify; out of scope this run | runtime-downloaded weights, unclassified dataset files |

Precedence for aggregating: 🔴 > 🟡 > ⚪ > 🟢. ⚪ marks absence of knowledge,
not a finding; layers and modules can be 🟢 alongside ⚪ items.

The 5-layer framework (v9):

1. **Code** — authored source, in-tree upstream copies, module LICENSE
2. **Pretrained model weights** — checkpoints used at inference time
3. **Example datasets** — sample data shipped in the repo
4. **Runtime dependencies** — declared deps + tainted upstream imports
5. **Produced datasets / artifacts** — what the module emits (inherited posture)

---

## Executive summary

| Module | Module | L1 Code | L2 Weights | L3 Datasets | L4 Runtime deps | L5 Outputs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `video_ingestion_agent` | 🟢 | 🟢 | ⚪ | 🟢 | 🟢 | 🟢 |
| `reconstruction` | 🔴 | 🟡 | ⚪ | 🟢 | 🔴 | 🔴 |
| `robotic_grounding` | 🔴 | 🟡 | ⚪ | 🟡 | 🔴 | 🔴 |

**Totals:** 🟢 1 module · 🟡 0 · 🔴 2.
**Layers with ⚪ unknown content:** L2 (all modules) — weights are runtime-downloaded, not in tree.

---
---

# 🔴 RED — `reconstruction`

**Path:** `reconstruction/`
**Reason for RED:** Layer 4 has two hard-NC taints (CC BY-NC depth, SMPL family); Layer 5 outputs inherit them.

---

## Layer 1: Code 🟡

**Layer status:** 🟡 No module-root LICENSE; two in-tree upstream copies under research-only / copyleft terms.

#### LICENSE files in this module

| Status | Path | Classification | Commercial |
|:---:|---|---|---|
| 🟡 | *(module root — none)* | — | **no LICENSE at module root** |
| 🟡 | `reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/LICENSE` | `nvidia_sec_3_3` | research_only_outside_nvidia |
| 🟢 | `reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda/torch_ngp_grid_encoder/LICENSE` | `mit` | ok |
| 🟡 | `reconstruction/modules/v2d_nlf/lib/lib_smpl/smplpytorch/LICENSE` | `gpl3` | restrictive |

#### In-tree upstream code copies

| Status | Path | License | Commercial |
|:---:|---|---|---|
| 🟡 | `reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/` | `nvidia_sec_3_3` | research_only_outside_nvidia |
| 🟡 | `reconstruction/modules/v2d_nlf/lib/lib_smpl/smplpytorch/` | `gpl3` | restrictive (linking taints wrappers) |

#### Module-root metadata

| Status | Item | Value |
|:---:|---|---|
| 🟡 | pyproject.toml at module root | not present |
| 🟡 | Declared license in pyproject | n/a |

#### Path summary — Layer 1

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🟡 | No module-root LICENSE makes commercial intent ambiguous at top level; in-tree NVIDIA Sec 3.3 + GPL-3 sub-paths cannot be used in commercial output paths without legal review. |
| Academic path | 🟢 | All sub-paths available for research use. |
| Tainted / inherited per upstream | 🟡 | `nvidia_sec_3_3` (in-tree FoundationPose); `gpl3` (in-tree smplpytorch) |

---

## Layer 2: Pretrained model weights ⚪

**Layer status:** ⚪ Weights are runtime-downloaded, not in tree — audit cannot enumerate.

| Status | Item | Note |
|:---:|---|---|
| ⚪ | Per-weight enumeration | Out of scope this run — weights downloaded at runtime |
| ⚪ | Per-weight license verification | Out of scope this run — would require parsing per-sub-module download scripts |

#### Path summary — Layer 2

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | ⚪ | Cannot determine — per-weight verification required (UniDepth/DA3 are known CC BY-NC; FoundationPose/BundleSDF are NVIDIA Sec 3.3; SAM 3D models are SAM License; others permissive — but not verified by this run) |
| Academic path | ⚪ | Same — depends on per-weight terms |
| Tainted / inherited per upstream | ⚪ | Same |

---

## Layer 3: Example datasets 🟢

**Layer status:** 🟢 No example datasets shipped in tree.

| Status | Item | Note |
|:---:|---|---|
| 🟢 | In-tree example datasets | None shipped |

#### Path summary — Layer 3

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🟢 | No in-tree datasets to consider |
| Academic path | 🟢 | Same |
| Tainted / inherited per upstream | 🟢 | None |

---

## Layer 4: Runtime dependencies 🔴

**Layer status:** 🔴 Hard-NC imports detected (CC BY-NC depth + SMPL family).

#### Declared dependencies at module root

| Status | Item | Note |
|:---:|---|---|
| 🟡 | Module-root pyproject.toml | not present — deps declared across 49 nested `reconstruction/modules/v2d_*/pyproject.toml` files |
| ⚪ | Per-sub-module dep extraction | Out of scope this run |

#### Taint exposure (imports of restricted upstream packages)

| Status | Taint vector | License | Commercial | Via package | Citations |
|:---:|---|---|---|---|---|
| 🟡 | `nvidia_sec_3_3` | `nvidia_sec_3_3` | research_only_outside_nvidia | `foundationpose` | `reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/estimater.py:15` |
| 🟡 | `nvidia_sec_3_3` | `nvidia_sec_3_3` | research_only_outside_nvidia | `bundlesdf` | `reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/Utils.py:50`; `reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/bundlesdf/run_nerf.py:14` |
| 🔴 | `nc_depth_models` | `ccbync4` | **not_allowed** | `unidepth` | `reconstruction/modules/v2d_unidepth/lib/image_to_depth.py:5`; `reconstruction/modules/v2d_unidepth/lib/video_to_depth.py:5` |
| 🟡 | `meta_sam_3d` | `sam_license` | ok_with_restrictions | `sam_3d_body` | 21 hits in `reconstruction/modules/v2d_sam3d_body/lib/` (e.g., `mv_optimize_mhr_params.py:31`, `:32`, `:33`) |
| 🔴 | `smpl_family` | `smpl_nc` | **not_allowed** | `smplpytorch` | `reconstruction/modules/v2d_nlf/lib/lib_smpl/smplpytorch/demo.py:1` |

#### Path summary — Layer 4

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🔴 | `v2d_unidepth` and `v2d_nlf` (smplpytorch path) cannot be used in commercial output paths. `v2d_foundation_pose` (FoundationPose + BundleSDF) is research-only outside NVIDIA. `v2d_sam3d_body` is commercial-OK under SAM License with AUP restrictions. |
| Academic path | 🟢 | All sub-modules available for research use. |
| Tainted / inherited per upstream | 🔴 | `unidepth` → `ccbync4` (NC); `smplpytorch` → `smpl_nc` (NC); `foundationpose`, `bundlesdf` → `nvidia_sec_3_3` (NC outside NVIDIA); `sam_3d_body` → `sam_license` (ok with AUP) |

---

## Layer 5: Produced datasets / artifacts 🔴

**Layer status:** 🔴 Two hard-NC inheritances active.

#### Implied output license posture

| Status | License inherited | Commercial | From |
|:---:|---|---|---|
| 🔴 | `ccbync4` | **not_allowed** | depth pipelines via UniDepth |
| 🔴 | `smpl_nc` | **not_allowed** | SMPL fitting via smplpytorch |
| 🟡 | `nvidia_sec_3_3` | research_only_outside_nvidia | FoundationPose poses + BundleSDF meshes |
| 🟡 | `sam_license` | ok_with_restrictions | outputs through SAM 3D Body |

#### Path summary — Layer 5

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🔴 | No commercial output is possible from any pipeline that passes through depth (UniDepth) or SMPL fitting. Pipelines that exclude those two stages produce outputs governed by NVIDIA Sec 3.3 (NC outside NVIDIA) or SAM License (ok with AUP). |
| Academic path | 🟢 | All output paths available for research use. |
| Tainted / inherited per upstream | 🔴 | As listed in Layer 4 taint table above — propagates to every output that flows through the corresponding stage. |

---
---

# 🔴 RED — `robotic_grounding`

**Path:** `robotic_grounding/`
**Reason for RED:** Layer 4 imports MANO directly (Max Planck NC); Layer 5 outputs inherit it.

---

## Layer 1: Code 🟡

**Layer status:** 🟡 No LICENSE files anywhere; module-root `pyproject.toml` has no `license` field.

#### LICENSE files in this module

| Status | Path | Classification | Commercial |
|:---:|---|---|---|
| 🟡 | *(module root — none)* | — | **no LICENSE at module root** |
| 🟡 | *(anywhere in module)* | — | **no LICENSE files anywhere walked** |

#### In-tree upstream code copies

| Status | Item | Note |
|:---:|---|---|
| 🟢 | In-tree upstream copies | None detected |

#### Module-root metadata

| Status | Item | Value |
|:---:|---|---|
| 🟢 | pyproject.toml at module root | `robotic_grounding/pyproject.toml` ✓ |
| 🟡 | Declared license in pyproject | **no `license` field** |

#### Path summary — Layer 1

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🟡 | No LICENSE in tree and no `license` field in pyproject → commercial intent of authored code is unstated. Cannot rely on Apache 2.0 / MIT defaults. |
| Academic path | 🟢 | Code is openly readable for research; no in-tree NC source. |
| Tainted / inherited per upstream | 🟢 | No in-tree upstream copies |

---

## Layer 2: Pretrained model weights ⚪

**Layer status:** ⚪ Out of scope.

| Status | Item | Note |
|:---:|---|---|
| ⚪ | Per-weight enumeration | Out of scope this run |

#### Path summary — Layer 2

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | ⚪ | Cannot determine |
| Academic path | ⚪ | Same |
| Tainted / inherited per upstream | ⚪ | Same |

---

## Layer 3: Example datasets 🟡

**Layer status:** 🟡 Dataset assets are in tree under `source/.../assets/human_motion_data/`; audit does not classify dataset files at rest in this run.

| Status | Item | Note |
|:---:|---|---|
| 🟡 | In-tree dataset assets | Present (USDA / OBJ / parquet under `source/robotic_grounding/.../assets/human_motion_data/`); provenance + license not classified by this audit run |
| ⚪ | Per-asset license metadata | Out of scope this run — would require per-file provenance sidecar |

#### Path summary — Layer 3

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🟡 | Dataset assets unclassified; cannot confirm commercial-OK at file level. Per v9 / module docs, sources include Arctic (CC BY-NC-SA), HOT3D (multi-license), SOMA — but per-file provenance is not in the audit's data. |
| Academic path | 🟡 | Available for research subject to each source dataset's terms. |
| Tainted / inherited per upstream | ⚪ | Per-file provenance not recorded |

---

## Layer 4: Runtime dependencies 🔴

**Layer status:** 🔴 Hard-NC import of MANO.

#### Declared dependencies at module root

| Status | Item | Note |
|:---:|---|---|
| 🟢 | Module-root pyproject.toml | Present |
| ⚪ | Top-level deps extracted | None at module root; deps live in nested `robotic_grounding/source/robotic_grounding/pyproject.toml` (out of scope this run) |

#### Taint exposure (imports of restricted upstream packages)

| Status | Taint vector | License | Commercial | Via package | Citations |
|:---:|---|---|---|---|---|
| 🔴 | `smpl_family` | `smpl_nc` | **not_allowed** | `mano` | `robotic_grounding/source/robotic_grounding/robotic_grounding/retarget/distance_utils.py:12`; `robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2p/mdp/commands/commands.py:243` |

#### Path summary — Layer 4

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🔴 | MANO import in retarget + v2p task code means these code paths cannot be used in commercial output pipelines. |
| Academic path | 🟢 | Available for research. |
| Tainted / inherited per upstream | 🔴 | `mano` → `smpl_nc` (NC) |

---

## Layer 5: Produced datasets / artifacts 🔴

**Layer status:** 🔴 Inherits hard-NC SMPL terms.

#### Implied output license posture

| Status | License inherited | Commercial | From |
|:---:|---|---|---|
| 🔴 | `smpl_nc` | **not_allowed** | MANO usage in retargeting + v2p task code |

#### Path summary — Layer 5

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🔴 | No commercial output is possible from pipelines that consume MANO-fitted data. Max Planck offers a commercial licensing pathway: `ps-license@tue.mpg.de`. |
| Academic path | 🟢 | All outputs available for research. |
| Tainted / inherited per upstream | 🔴 | `mano` → `smpl_nc` propagates |

---
---

# 🟢 GREEN — `video_ingestion_agent`

**Path:** `video_ingestion_agent/`
**Reason for GREEN:** No taint exposure across detectable layers; module-root LICENSE present and permissive.

---

## Layer 1: Code 🟢

#### LICENSE files in this module

| Status | Path | Classification | Commercial |
|:---:|---|---|---|
| 🟢 | `video_ingestion_agent/LICENSE` | `apache2` | ok |

#### In-tree upstream code copies

| Status | Item | Note |
|:---:|---|---|
| 🟢 | In-tree upstream copies | None |

#### Module-root metadata

| Status | Item | Value |
|:---:|---|---|
| 🟢 | pyproject.toml at module root | `video_ingestion_agent/pyproject.toml` ✓ |
| 🟢 | Declared license in pyproject | `Apache-2.0` |

#### Path summary — Layer 1

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🟢 | All authored code is Apache 2.0 |
| Academic path | 🟢 | Same |
| Tainted / inherited per upstream | 🟢 | None |

---

## Layer 2: Pretrained model weights ⚪

**Layer status:** ⚪ Out of scope for discovery; module docs declare permissive weights.

| Status | Item | Note |
|:---:|---|---|
| ⚪ | Per-weight enumeration | Out of scope this run |
| ⚪ | Per-weight license verification | Out of scope this run — module docs claim Qwen3-VL (Apache-2.0) and SigLIP-2 (Apache-2.0); not verified by audit |

#### Path summary — Layer 2

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | ⚪ | Unverified by audit; module docs claim permissive weights only |
| Academic path | ⚪ | Same |
| Tainted / inherited per upstream | ⚪ | Same |

---

## Layer 3: Example datasets 🟢

| Status | Item | Note |
|:---:|---|---|
| 🟢 | In-tree example datasets | None shipped |

#### Path summary — Layer 3

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🟢 | No in-tree datasets |
| Academic path | 🟢 | Same |
| Tainted / inherited per upstream | 🟢 | None |

---

## Layer 4: Runtime dependencies 🟢

#### Declared dependencies at module root (12)

| Status | Dep spec | Expected license (per common knowledge — not auto-verified) |
|:---:|---|---|
| 🟢 | `numpy>=1.24.0` | BSD-3 |
| 🟢 | `opencv-python>=4.8.0` | Apache 2.0 |
| 🟢 | `pillow>=10.0.0` | HPND (MIT-compatible) |
| 🟢 | `langgraph>=0.2.0` | MIT |
| 🟢 | `langchain-core>=0.3.0` | MIT |
| 🟢 | `requests>=2.31.0` | Apache 2.0 |
| 🟢 | `openai>=1.0.0` | Apache 2.0 |
| 🟢 | `sqlalchemy>=2.0.0` | MIT |
| 🟢 | `pydantic>=2.0.0` | MIT |
| 🟢 | `tqdm>=4.66.0` | MIT |
| 🟢 | `pyyaml>=6.0.0` | MIT |
| 🟢 | `rich>=13.0.0` | MIT |

Dep-license auto-resolution (PyPI metadata querying) is out of scope this run.

#### Taint exposure

| Status | Item | Note |
|:---:|---|---|
| 🟢 | Taint exposure | None detected |

#### Path summary — Layer 4

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🟢 | All 12 declared deps are permissive |
| Academic path | 🟢 | Same |
| Tainted / inherited per upstream | 🟢 | None |

---

## Layer 5: Produced datasets / artifacts 🟢

#### Implied output license posture

| Status | License inherited | Commercial | From |
|:---:|---|---|---|
| 🟢 | `apache2` | ok | This module's own LICENSE; no upstream taint |

#### Path summary — Layer 5

| Path | Status | Notes |
|---|:---:|---|
| Commercial path | 🟢 | Outputs (frame embeddings, retrieval indices, graph/vector DBs, structured metadata) are commercially usable under Apache 2.0 |
| Academic path | 🟢 | Same |
| Tainted / inherited per upstream | 🟢 | None |

---
---

## Cross-cutting observations

- The `smpl_family` taint touches **two** modules — `reconstruction` (via `smplpytorch` in `v2d_nlf/lib/lib_smpl/`) and `robotic_grounding` (via `mano` in retarget + v2p code). Same family of restriction, different upstream packages, different modules.
- 2 of 3 modules have no LICENSE file at module root. `reconstruction` has correctly-placed sub-LICENSEs at three nested paths covering in-tree upstream copies. `robotic_grounding` has no LICENSE files anywhere walked.
- `video_ingestion_agent` is the only module with **both** a module-root LICENSE **and** a `license` field in its `pyproject.toml`.
- The audit's deterministic taint detection lives in Layer 4 (imports vs. catalog). Layer 5 is logical entailment from Layer 4 + Layer 1.

---

## Coverage gaps (what this audit does NOT cover)

| Status | Gap | Why |
|:---:|---|---|
| ⚪ | Pretrained weights at runtime | Downloaded files, not in tree |
| ⚪ | Dataset files at rest | Not classified per-file in this run |
| ⚪ | Git-lfs pointer files | Audit reads file text; pointer files aren't license text |
| ⚪ | Per-sub-module breakdown of `reconstruction` | 49 nested pyprojects listed by count, not parsed |
| ⚪ | Runtime dep license resolution | Deps listed but upstream licenses not auto-resolved from PyPI |
| ⚪ | Output datasets at rest | Report describes *implied inherited posture* only, not emitted files |

---

## Validation status

| Status | Item | Result |
|:---:|---|---|
| 🟢 | `audit/scripts/validate_structure.py` | passed (exit 0) |
| 🟢 | Unclassified LICENSE files in tree | none |
| 🟢 | `audit_structure.yaml` schema | schema_version=1, structure_version=1 |
| 🟢 | Catalog entries verified against upstream sources | `nvidia_sec_3_3` (local LICENSE), `smpl_nc` (MANO scope only), `smplx_nc`, `sam_license` (both sam-3d-objects and sam-3d-body GitHub LICENSE files), `ccbysa4`, `ccbync4`, `ccbyncsa4`, `hot3d_meta` |
| ⚪ | Catalog entries NOT verified this run | `apache2`, `mit`, `bsd2`, `bsd3`, `gpl3` (well-defined SPDX standards). SMPL and SMPL-H license pages were not separately verified — `smpl_nc` is verified for MANO only |
