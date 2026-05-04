# Release-Readiness Assessment

> Pre-publication audit of `video_ingestion_agent` for external open-source release.
> Recorded so it can be picked up later — nothing in this document changes code.

## Verdict — **NOT YET READY**

The codebase itself is in good shape:

- 284/285 unit tests pass; 14/14 component smoke tests pass.
- End-to-end pipeline runs cleanly on a real video (segmentation → verify →
  refine → entity graph → embeddings → DB write → HTML report).
- `ruff check` + `ruff format` clean; pre-commit passes.
- Apache-2.0 LICENSE present; SPDX headers on every source file.
- Dependencies are all OSS-license-compatible.
- `.gitignore` already covers logs, runs, outputs, caches, IDE files.

What blocks publication is the **public-facing surface**: URLs, docs,
defaults, and missing OSS conventions. About 1–2 days of cleanup,
excluding any external-infrastructure decisions (W&B project rename,
NGC re-publish, OSMO removal).

---

## Blockers — must fix before flipping to public

### 1. URL / identifier mismatch from the rename

The rename pass intentionally left URLs alone because the new external
slug hadn't been decided. Today the repo is internally inconsistent.

- `pyproject.toml:111-115` — `name = "video_ingestion_agent"` but every
  `[project.urls]` entry still points at `github.com/NVIDIA/v2p_video_agent`.
- `pyproject.toml:8-10` — `authors = [{name = "V2P Team", email =
  "v2p@nvidia.com"}]` still uses the old team identity.
- `docs/conf.py:79` — `github_url` points at internal GitLab
  (`gitlab-master.nvidia.com/liuw/v2p_video_agent`).
- `docs/conf.py:101-107` — `video_ingestion_agent_docs_config` has 4 URL
  fields (internal git, external git, internal/external code-link bases)
  all still referencing `v2p_video_agent`.
- `docs/conf.py:29` — `released = False`. Flips docs to "external"
  rendering when set True; needs flipping (and the URLs above need to be
  correct for external mode).
- `README.md:11` — the docs link points at internal GitLab Pages
  (`v2p-video-agent-eec0b3.gitlab-master-pages.nvidia.com`) which won't
  resolve for external readers.

**Decision needed:** what is the public GitHub slug?
`NVIDIA/video_ingestion_agent`? Same question for the public docs site
(`nvidia.github.io/video_ingestion_agent`?).

### 2. README is far too thin for a public landing page

`README.md` is 17 lines: title, one-line description, broken docs URL,
license. For a public NVIDIA repo this needs:

- Brief feature overview (segmentation, entity-graph, retrieval).
- Hardware / OS requirements.
- Install instructions (`uv pip install -e .` + the `[server]` extra for
  the vLLM workflow).
- Quickstart example a new reader can copy-paste — at minimum the
  `serve.py` + `run_ingestion.py` commands we know work end-to-end.
- Link to the published Sphinx docs.
- Citation / license / acknowledgements.

### 3. `docs/pages/deployment.rst` is mostly internal infrastructure

The whole page is OSMO-deployment-focused with internal-only references:

- Line 4: "OSMO cluster" — internal NVIDIA orchestration system.
- Lines 18, 28, 47: `nvcr.io/nvstaging/isaac-amr/video_ingestion_agent:latest`
  — internal NGC staging registry; external users can't pull.
- Line 84: `/mnt/nfs/outputs` — internal NFS layout.
- Lines 50–end: full OSMO workflow walkthrough.

**Decision needed:** keep this page (with a clear "internal NVIDIA
deployment, illustrative only" disclaimer at the top) or remove the OSMO
section entirely from the public docs?

### 4. Internal infrastructure baked into runtime defaults

These defaults will break first-run UX for external users:

- `src/video_ingestion_agent/webapp/config.py:34` —
  `default_db_dir = "/mnt/amlfs-02/shared/liuw/v2p/database"`
- `src/video_ingestion_agent/webapp/config.py:391` —
  `db_scan_dirs = ("/mnt/amlfs-02/shared/liuw/v2p/database",)`
- `src/video_ingestion_agent/benchmark/wandb_logger.py:34` —
  `entity: str = "nvidia-isaac"` (default W&B entity).

Replace defaults with portable ones (`outputs/`, `~/.video_ingestion_agent/db`,
or env-var driven). The W&B entity should default to `None` (use the
caller's account) and become an explicit opt-in.

### 5. Dockerfile base image is internal-only

`Dockerfile:27` — `ARG BASE_IMAGE=nvcr.io/nvstaging/isaac-amr/cosmos_reason_2`.
External users can't pull this. Either:

- Switch the default `BASE_IMAGE` to a publicly pullable PyTorch+CUDA
  image (e.g. `nvcr.io/nvidia/pytorch:24.10-py3`), or
- Document explicitly that the image needs a base override and provide
  an override example.

### 6. OSMO workflows are internal-only

`osmo_workflows/*.yaml` reference internal cluster pools, NFS shares,
and NGC tags. `scripts/run_osmo.py` defaults to
`nvcr.io/nvstaging/isaac-amr/v2p_*:latest`. **Decision needed:** ship as
"NVIDIA-internal example, advanced users only" or move to a separate
internal-only repo?

---

## Outstanding deliverables — research / writing in flight

These are release-gating but different in kind from the repo-hygiene
punch list above: they need execution and writing time, not edits.

### A. Benchmark segmentation + retrieval on HoT3D and EPIC-KITCHENS

Numbers in the paper and on the public docs landing page should come
from real benchmark runs, not anecdote.

**Segmentation (EPIC-KITCHENS-100).** Infrastructure is already wired
up: `configs/benchmark_epic_kitchens.yaml`,
`src/video_ingestion_agent/benchmark/load_epic_kitchens.py`,
`scripts/run_benchmark.py`, and the EPIC-KITCHENS adapter
(`src/video_ingestion_agent/benchmark/adapter.py`). Run end-to-end
across the standard validation split, log to W&B (after the entity-
default fix in Blocker §4), and capture: segment-level precision /
recall / F1 against the gold verb+noun annotations, refinement-loop
convergence rate, and wall-clock per video.

**Segmentation + retrieval (HoT3D).** No HoT3D-specific benchmark
harness exists yet — `src/video_ingestion_agent/benchmark/` has only
the EPIC-KITCHENS loader. Need:

- A HoT3D loader analogous to `load_epic_kitchens.py` that maps HoT3D
  ground-truth annotations into `ClipContext`-compatible records.
- A retrieval-evaluation harness on top of the existing
  `RetrievalAgent` — query → recovered clips → IoU/recall against gold
  intervals. Nothing in the current `tests/` exercises retrieval against
  ground truth.
- A run script (`scripts/run_hot3d_benchmark.py`) and a config
  (`configs/benchmark_hot3d.yaml`).

Existing HoT3D references in the repo are NFS paths only
(`/mnt/amlfs-02/shared/liuw/v2p/hot3d/...`), which need to either move
to a portable location or be replaced with a download script before
the benchmark is reproducible externally.

**Output:** a results table in `docs/pages/benchmark.rst` (currently
covers the design, not numbers) and a section in the paper.

### B. Finish the paper draft

Public release should ideally land alongside the paper, or at least
with an arXiv link in the README. Open items:

- Locate / surface the current draft (not in this repo today; presumably
  in a separate Overleaf or paper repo). Decide whether to mirror a
  preprint into this repo under `paper/` or just link out.
- Lock benchmark numbers from item A above into the results section.
- Figures: the architecture diagram (`docs/images/video_ingestion_agent_overview.jpg`)
  is reusable; pipeline / retrieval-loop figures should be regenerated
  to match the rename.
- Citation: once a preprint is up, add `CITATION.cff` to the repo
  (also listed under Nits) and update the README "Citation" section.
- License + author list review for paper-vs-repo consistency.

These two items are the long pole. Repo-hygiene blockers can be
finished in 1–2 days; benchmarks + paper are weeks.

---

## Cleanup — should fix, low cost

- **No `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `CHANGELOG.md`.** All four are NVIDIA-OSS standard. Crib from an
  existing public NVIDIA repo's templates.
- **No GitHub Actions.** `.gitlab-ci.yml` runs pre-commit + Sphinx
  build — port the equivalent two jobs to `.github/workflows/ci.yml`
  so PRs against the public repo get checked.
- **No test markers for GPU / integration tests.** Today all 285 tests
  run as unit tests because nothing real-time ever exercises a GPU path
  in CI. Once GitHub Actions is live, mark the GPU-touching tests with
  `@pytest.mark.gpu` and skip them in the no-GPU CI lane.
- **`entity_graph_build.log` (413 KB) sits in the working tree.** Not
  tracked by git (verified: `git ls-files entity_graph_build.log` → empty;
  `.gitignore` line 88 covers `*.log`). Just `rm` it before publishing.
- **pre-commit pin is old.** `.pre-commit-config.yaml:43` pins ruff
  v0.9.6; current ruff (0.15.x) prints a "removed rule UP038" warning
  locally because pyproject.toml's `ignore` list still contains UP038.
  Either bump the pin or drop UP038 from the ignore list (and verify
  no new violations surface).
- **`pyproject.toml [project.scripts]` was deleted in the rename pass.**
  No README / docs claim those CLI commands exist (verified during
  rename), but worth a single re-grep before publishing.
- **Author block.** `{name = "V2P Team", email = "v2p@nvidia.com"}` —
  rename to a real maintainer alias for the public repo.

---

## Nits — nice to have, not gating

- Add badges to README (license, Python version, docs build).
- `CITATION.cff` if the project is intended to be cited academically.
- mypy is configured permissively (`disallow_untyped_defs = false`).
  Fine for alpha; tightening can wait.
- Coverage isn't enforced — `pyproject.toml` runs `--cov` but no
  threshold. Optional.

---

## Decisions needed before action

These determine the shape of the follow-up implementation plan:

1. **Public GitHub slug** — `NVIDIA/video_ingestion_agent`? Different?
2. **Public docs URL** — GitHub Pages under that repo? `nvidia.github.io/...`?
3. **Maintainer identity** — what name + email goes in `pyproject.toml`?
4. **OSMO + NGC-internal content** — keep with a disclaimer, or strip?
   This determines whether `osmo_workflows/`, `scripts/run_osmo.py`,
   and the OSMO half of `docs/pages/deployment.rst` stay or go.
5. **Dockerfile base image** — replace default with a publicly pullable
   image, or require users to override `--build-arg BASE_IMAGE`?
6. **W&B defaults** — drop the `nvidia-isaac` entity / `v2p-benchmark`
   project entirely, or keep them as documented examples in optional
   benchmark code?

---

## Verification — what "ready" looks like

After the blocker pass:

1. `rg -i 'v2p|gitlab-master|nvstaging|nvidia-isaac|/mnt/amlfs|liuw'`
   returns zero hits outside test fixtures and historical commit
   messages.
2. `git ls-files` is clean — no logs, no caches, no `__pycache__`.
3. README quickstart commands work on a fresh clone in a fresh venv on
   a machine with one GPU, a `HF_TOKEN`, and ffmpeg.
4. `cd docs && make html` succeeds with `released = True` and the
   rendered HTML's external links resolve to public URLs.
5. `pre-commit run --all-files` clean.
6. `python scripts/run_ingestion.py <video> -c configs/ingestion.yaml`
   end-to-end on a sample video (already verified to work today).
7. A second-pair-of-eyes pass over the rendered docs landing page,
   architecture page, and README catches anything internal that text
   search missed.
