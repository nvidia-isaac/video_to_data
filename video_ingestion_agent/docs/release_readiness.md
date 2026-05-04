# Release-Readiness Plan

> Tracking the remaining work between `video_ingestion_agent` as it sits today
> in the `nvidia-isaac/video_to_data` monorepo and a public open-source release.
> Items completed this session are marked **DONE** with the relevant commits;
> remaining work is grouped by why it's still open.

## Verdict — **partial; safe to push the import branch, not yet safe to flip the repo public**

What's working:

- 288/289 unit tests pass (4 added this session for the retrieval path
  fallback); 14/14 component smoke tests pass.
- Full ingestion pipeline runs end-to-end on real video at the new
  location (segmentation → optional verify/refine → entity graph →
  embeddings → DB write → HTML report).
- Retrieval agent runs end-to-end and now actually extracts `.mp4`
  clips (the leading-slash path bug is patched defensively).
- Webapp boots, serves UI, and reads our local `outputs/graph.db`
  through the `DatabaseService` layer.
- vLLM 0.15.x is now pinned in `pyproject.toml` so a fresh
  `uv pip install -e ".[all]"` doesn't pull the broken 0.20+ Qwen3-VL.
- Apache-2.0 LICENSE present; SPDX headers on every source file.

What still blocks public flip: a handful of internal-infrastructure
references the maintainer chose to defer until the broader monorepo
integration is complete; an author-email placeholder; and the larger
research/writing deliverables (benchmarks + paper).

---

## Blockers — must fix before flipping to public

### 1. URL / identifier sweep — **DONE** (commit `ca490b3` / earlier)

- `pyproject.toml [project.urls]` — all 4 URLs now point at
  `github.com/nvidia-isaac/video_to_data` (Repository / Issues) and
  `.../tree/main/video_ingestion_agent` (Homepage / Documentation).
- `docs/conf.py` — `github_url`, `internal_git_url`, `external_git_url`,
  and the two `code_link_base_url` entries all point at the monorepo;
  `released = True`.
- `README.md` docs link points at the subfolder URL.
- Two leftover GitLab issue links in `docs/pages/development.rst` and
  `docs/pages/troubleshooting.rst` rewritten to GitHub Issues.
- **One remainder:** `pyproject.toml` `authors[0].email` is still
  `TODO@nvidia.com`. Maintainer to supply the real address.

### 2. README rewrite — **DONE** (commit `ca490b3` / `b0db57a`)

Replaced the original 17-line stub with a 158-line README covering:
feature overview, requirements, install (one-shot `[all]` plus
partial-extras list), quickstart matching the verified end-to-end
flow, deferred-verification gotcha note, docs links, license,
citation placeholder.

### 3. `docs/pages/deployment.rst` is mostly internal infrastructure — **DEFERRED**

Maintainer choice (this session): keep the OSMO + nvstaging
references in place until the system is integrated and verified end
to end inside the monorepo. Cleanup to follow.

References still in:
- "OSMO cluster" prose at the top of the page.
- `nvcr.io/nvstaging/isaac-amr/video_ingestion_agent:latest` in three
  Docker examples.
- `/mnt/nfs/outputs` example NFS path.
- Full OSMO workflow walkthrough (lines 50–end).

### 4. Internal infrastructure baked into runtime defaults — **DEFERRED**

Maintainer choice: address with #3 once the system is running in
prod. Functional impact: zero — `default_db_dir` is overridden at
runtime via `configs/retrieval.yaml`'s `database.directory`. Only a
cosmetic source-code embarrassment for public eyes.

References still in:
- `src/video_ingestion_agent/webapp/config.py:34` —
  `default_db_dir = "/mnt/amlfs-02/shared/liuw/v2p/database"`
- `src/video_ingestion_agent/webapp/config.py:391` —
  `db_scan_dirs = ("/mnt/amlfs-02/shared/liuw/v2p/database",)`
- `src/video_ingestion_agent/benchmark/wandb_logger.py:34` —
  `entity: str = "nvidia-isaac"` (default W&B entity).

### 5. Dockerfile base image is internal-only — **DEFERRED**

Maintainer choice: defer with #3 / #4. `Dockerfile:27` —
`ARG BASE_IMAGE=nvcr.io/nvstaging/isaac-amr/cosmos_reason_2`.

### 6. OSMO workflows are internal-only — **DEFERRED**

Maintainer choice: defer with #3 / #4. `osmo_workflows/*.yaml` and
`scripts/run_osmo.py` defaults at `nvcr.io/nvstaging/isaac-amr/...`.

---

## Recently fixed (this session, commit `b0db57a`)

These weren't on the original blocker list — they surfaced during the
integration test inside the monorepo and were fixed in line with the
"smooth onboarding" goal:

- **vLLM version pin.** `vllm>=0.8.0` resolved to `vllm==0.20.1` in a
  fresh venv, which has a Qwen3-VL deepstack regression
  (`num_tokens=336 > deepstack_input_embeds_num_tokens=329`) that
  crashes inference on first call. Tightened to
  `vllm>=0.15.1,<0.20`.
- **`[all]` extra** now includes `[server]` so a single
  `uv pip install -e ".[all]"` brings vLLM along with everything else.
- **Retrieval clip-extraction "Video not found" with leading slash.**
  The analyzer LLM occasionally rewrites `data/foo.mp4` →
  `/data/foo.mp4` when echoing search results back. Defensive
  recovery now lives in `extract_clip._get_video_path`: it strips a
  bogus leading `/` and falls back to the `video_id` registry. Four
  unit tests cover the cases.
- **Troubleshooting docs.** Two new entries in
  `docs/pages/troubleshooting.rst` covering the vLLM-deepstack and
  retrieval-leading-slash symptoms.

These fixes also imply: a deeper analyzer-LLM refactor (don't trust
LLM-echoed paths/timestamps/video_ids — re-resolve from search-result
identity) is still open as a future improvement, but the user-facing
wound is closed.

---

## Outstanding deliverables — research / writing in flight

Unchanged from the original assessment; both items are weeks of work,
not part of the blocker pass.

### A. Benchmark segmentation + retrieval on HoT3D and EPIC-KITCHENS

- **EPIC-KITCHENS-100** harness exists
  (`configs/benchmark_epic_kitchens.yaml`,
  `src/video_ingestion_agent/benchmark/load_epic_kitchens.py`,
  `scripts/run_benchmark.py`,
  `src/video_ingestion_agent/benchmark/adapter.py`). Run end-to-end
  on the standard val split, log to W&B.
- **HoT3D** scaffolding is now partially in place — the user added
  `src/video_ingestion_agent/benchmark/load_hot3d.py`,
  `src/video_ingestion_agent/benchmark/evaluate_hot3d.py`, and
  `scripts/run_benchmark_hot3d.py` (still has unused-import warnings
  flagged by ruff — deliberate WIP scaffolding). Need a config and a
  retrieval-evaluation harness on top.
- HoT3D references are still NFS paths
  (`/mnt/amlfs-02/shared/liuw/v2p/hot3d/...`); needs a portable
  fixture or a download script before the benchmark is reproducible
  externally.
- **Output:** results table in `docs/pages/benchmark.rst` (currently
  covers design, not numbers) and a section in the paper.

### B. Finish the paper draft

- Locate the current draft (not in this repo) and decide whether to
  mirror under `paper/` or link out.
- Lock benchmark numbers from item A.
- Refresh figures (architecture diagram already renamed to
  `docs/images/video_ingestion_agent_overview.jpg`); pipeline /
  retrieval-loop figures need regeneration.
- Add `CITATION.cff` and update the README citation block once a
  preprint is up.

---

## Cleanup — should fix, low cost

- **`CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` / `SECURITY.md` /
  `CHANGELOG.md`** — these typically live at the **monorepo root**.
  Confirm what's already at `nvidia-isaac/video_to_data` root and
  reuse rather than adding subfolder duplicates.
- **GitHub Actions workflow.** The monorepo already follows a
  `<package>_pipeline_ci.yml` convention
  (`reconstruction_pipeline_ci.yml`, `robotic_grounding_ci.yml`).
  Add `video_ingestion_agent_pipeline_ci.yml` running pre-commit +
  Sphinx build, scoped to `video_ingestion_agent/**` paths. The
  current `.gitlab-ci.yml` inside the subfolder can be removed once
  ported.
- **Test markers for GPU / integration tests.** All 288 tests run as
  CPU unit tests today. Mark the GPU-touching ones with
  `@pytest.mark.gpu` so a CPU-only CI lane can skip them cleanly.
- **`pyproject.toml [project.scripts]` orphans** — re-grepped this
  session, no remaining doc references to the deleted `v2p-*` CLI
  commands. **No action needed.**
- **`entity_graph_build.log`** (413 KB working-tree artifact, not
  git-tracked) — **deleted this session.**
- **pre-commit ruff pin is old** (`v0.9.6`) — attempted to drop the
  `UP038` ignore this session; surfaced 15 pre-existing
  `isinstance(x, (X, Y))` violations the team explicitly preferred to
  keep. **Reverted.** The local "removed rule UP038" warning persists
  cosmetically; bumping the pre-commit ruff pin to current would
  silence it but risks new lint flags.
- **Author block** — partially fixed (name = "NVIDIA Isaac Team"),
  email still `TODO@nvidia.com`. Real address pending.

---

## Nits — nice to have, not gating

- Add badges to README (license, Python version, docs build).
- `CITATION.cff` once a preprint is up.
- Tighten mypy (`disallow_untyped_defs = false` today).
- Add a coverage threshold to `pyproject.toml`'s pytest config.

---

## Decisions — status

| # | Decision | Status |
| --- | --- | --- |
| 1 | Public GitHub slug | **Decided:** `nvidia-isaac/video_to_data` (subfolder `video_ingestion_agent/`). |
| 2 | Public docs URL | **Decided:** subfolder URL on GitHub for now (`.../tree/main/video_ingestion_agent`); separate Sphinx hosting deferred. |
| 3 | Maintainer identity | **Partial:** name = "NVIDIA Isaac Team"; email TBD. |
| 4 | OSMO + NGC-internal content | **Deferred** — maintainer keeping until system integration is verified. |
| 5 | Dockerfile base image | **Deferred** — same group as #4. |
| 6 | W&B defaults | **Deferred** — same group as #4. |

---

## Verification — what's been confirmed; what still needs checking

**Confirmed this session:**

1. `pytest tests/ -o addopts=""` → **288 passed, 1 skipped** at the
   destination (`/home/liuw/Projects/video_to_data/video_ingestion_agent`).
2. `ruff check .` clean modulo pre-existing user-WIP HoT3D file
   warnings.
3. `pre-commit run --all-files` passes on the destination.
4. End-to-end ingestion: `python scripts/run_ingestion.py <video>
   -c configs/ingestion.yaml --no-verify` populates
   `outputs/graph.db` (1 video, 6 entities, 11 relationships, 3
   action_segments) and `outputs/vector.db` (14 frame embeddings).
5. Retrieval queries via `scripts/run_retrieval.py` return correct
   clips and now extract real `.mp4` files under `outputs/clips/`.
6. Webapp boots on `127.0.0.1:7861`, serves Gradio HTML (HTTP 200,
   138 KB), `DatabaseService` reads our local DBs correctly.
7. `rg -i 'gitlab-master|NVIDIA/v2p_video_agent|v2p-video-agent-eec0b3'`
   returns zero hits outside this tracking doc.

**Still to check (post-blocker-pass):**

1. `cd docs && make html` builds Sphinx with `released = True` and
   the rendered HTML's external links resolve to the monorepo URLs.
2. Once `[server]` extras and other internal-infra defaults are
   addressed, a fresh-clone fresh-venv install of `[all]` followed
   by the README quickstart works on a machine that doesn't have any
   of `liuw`'s NFS mounts available.
3. A second-pair-of-eyes pass over the rendered docs landing page,
   architecture page, and README to catch anything internal that
   text search missed.
