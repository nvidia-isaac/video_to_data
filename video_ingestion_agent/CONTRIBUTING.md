# Contributing to Video Ingestion Agent

Thanks for your interest. This file is a quick map; the authoritative
developer reference is **`docs/pages/development.rst`** (rendered in the
Sphinx docs).

## Reporting issues

File a GitHub issue with:

- Repro steps (a minimal video / config / command).
- Expected vs actual behavior.
- Relevant log lines from `pipeline.log` (single-video runs) or the
  per-shard logs (batch runs).
- Your `video_ingestion_agent`, `vllm`, `torch`, and CUDA versions.

## Pull-request workflow

1. Fork or branch from `main`.
2. Set up the dev environment:

   ```bash
   uv sync --all-extras
   pre-commit install
   ```

3. Make focused commits. Each commit should be independently reviewable;
   prefer many small commits over one mega-commit.
4. Run the full check matrix locally before pushing:

   ```bash
   pre-commit run --all-files     # ruff format + lint
   mypy src/video_ingestion_agent
   pytest                          # full suite
   ```

5. Open the PR; CI runs the same matrix on Linux/Python 3.11. Aim for a
   green CI before requesting review.

## Code style

- **Formatting + linting:** `ruff` (config in `pyproject.toml`,
  line length 100, target py310).
- **Types:** `mypy` strict on `src/video_ingestion_agent`.
- **Tests:** add or update `tests/` for any non-trivial change. The
  `--cov` configuration in `pyproject.toml` is informational, not gating.
- **Docs:** if your change is architectural or user-facing, update the
  relevant page under `docs/pages/`. The Sphinx build is part of CI.

## Commit messages

Follow the convention already in `git log`:

```
[<area>] <imperative summary>

Optional body explaining the *why*.
```

Example: `[video_ingestion] retrieval: cap relaxation depth at 4 levels`.

## Architecture conventions

Documented in **`docs/pages/development.rst`** and the repo-level
**`../AGENTS.md`**.
Highlights:

- LangGraph state graphs are the source of truth for ingestion +
  retrieval flows. Add new behavior by editing the graph builder, not
  by sidestepping it.
- Node functions are pure: read from the typed state, return a
  partial-state dict.
- New SQLite columns must be added to `database_writer.py` and any
  reader that touches the same table; both DBs assume WAL mode.
- The webapp's `reconstruction:` tab is an integration example, not a
  feature of the agent — keep its setup surface in
  `src/video_ingestion_agent/reconstruction_interface/ego_e2e/README.md`,
  not in the agent's architecture pages.
