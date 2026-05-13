# Repository Instructions for Coding Agents

Use this file as the shared operating manual for Codex, Claude Code,
Copilot, Cursor, and other coding agents working in this repository. It
applies to the whole monorepo unless a more specific instruction file exists in
a subdirectory.

## Project Map

- `video_ingestion_agent/`: LangGraph-based video ingestion and retrieval agent.
  It turns robot demonstration videos into `graph.db` and `vector.db`, then
  answers natural-language retrieval queries over the resulting action database.
- `reconstruction/`: Docker-first reconstruction modules for depth, masks,
  meshes, 6D pose, SMPL body estimates, calibration, and multi-view pipelines.
- `robotic_grounding/`: Isaac Lab and RSL-RL workflows for retargeting human
  motion and training/evaluating robot policies.
- `.claude/skills/`: Claude-specific walkthroughs for the ingestion agent. Use
  them as domain references, but keep new repo-wide guidance tool-neutral.

## Default Agent Workflow

1. Start by reading the relevant package README plus this file.
2. Keep changes scoped to one package unless the task explicitly crosses package
   boundaries.
3. Prefer documentation, tests, and small integration seams before touching
   heavyweight GPU or Docker paths.
4. Run the lightest validation that proves the change. Call out when full
   validation needs GPU, Docker, model weights, private registries, or secrets.
5. Do not download model weights, build large containers, start training jobs,
   or access private services unless the user asked for that work.
6. Do not commit generated artifacts, model weights, databases, logs, videos, or
   local credentials.

## Package-Specific Guidance

### `video_ingestion_agent/`

- Use `uv` for environment management.
- Core flows are LangGraph state graphs. Add behavior by updating the graph
  builder and typed node/state objects, not by bypassing the graph.
- Node functions should read typed state and return partial-state dictionaries.
- SQLite schema changes must update both writers and readers.
- Keep webapp reconstruction integration docs under
  `src/video_ingestion_agent/reconstruction_interface/ego_e2e/`.
- Useful checks:

  ```bash
  cd video_ingestion_agent

  # Documentation and static checks
  uv sync --extra dev --extra docs
  pre-commit run --all-files
  mypy src/video_ingestion_agent
  cd docs && make html SPHINXOPTS="--keep-going"

  # Unit tests, matching the CI extras more closely
  uv sync --extra local --extra dev --extra benchmark --extra webapp
  pytest tests/ -o addopts=""
  ```

  The full CI-style unit test environment also needs `ffmpeg`.

### `reconstruction/`

- Preserve the host/container split:
  - `modules/v2d_*/docker/` is lightweight host orchestration.
  - `modules/v2d_*/lib/` is container-only ML inference code.
- Package boundaries should use typed dataclasses from shared packages such as
  `v2d_common`; avoid leaking third-party types across package APIs.
- Every new inference parameter should be exposed through both lib and docker
  wrappers, including CLI and programmatic entry points.
- Multi-view CLIs should load YAML config, merge CLI overrides, and then call a
  `*_from_config(cfg)` function.
- Useful checks:

  ```bash
  cd reconstruction
  ./scripts/install_packages.sh
  ./scripts/build_containers.sh
  ```

  Docker image builds, model downloads, and most end-to-end checks require GPU
  access and may be slow. For small docs-only changes, do not run them.

### `robotic_grounding/`

- Development happens inside the Isaac Lab container; Git operations happen on
  the host.
- Avoid editing checked-in datasets, generated visualizations, and local
  credential files.
- Preserve CI assumptions around GPU/self-hosted runners and NGC access.
- Useful checks:

  ```bash
  cd robotic_grounding
  bash workflow/setup_deps.sh
  pre-commit run --show-diff-on-failure --files $(git ls-files .)
  ```

  Full build and end-to-end tests require Docker, GPU, Isaac Lab, and access to
  the configured NVIDIA container registry.

## Validation Matrix

| Change type | Recommended validation |
| --- | --- |
| Root docs or agent guidance | Markdown review plus link/path checks |
| `video_ingestion_agent` docs | `cd video_ingestion_agent/docs && make html SPHINXOPTS="--keep-going"` |
| `video_ingestion_agent` code | `pre-commit`, `mypy src/video_ingestion_agent`, `pytest tests/ -o addopts=""` |
| Reconstruction docs | Check referenced scripts and module paths exist |
| Reconstruction wrappers | Install the targeted docker package and run the smallest module command possible |
| Reconstruction lib/container code | Targeted Docker build plus module-specific smoke test |
| `robotic_grounding` docs | Check commands and workflow paths exist |
| `robotic_grounding` code | `pre-commit` plus targeted container or CI validation when available |

## Security and Data Handling

- Never print or commit HuggingFace tokens, NGC keys, CSS credentials, API keys,
  or `.envrc.local` contents.
- Treat sample videos, meshes, parquet files, generated databases, and model
  weights as large artifacts unless the repo already tracks them intentionally.
- Prefer adding small synthetic fixtures for tests.
- If a command needs network, private registry access, GPU, or a long-running
  job, explain the reason before running it.

## PR Checklist for Agents

- Summarize what changed and why.
- List validation performed, including commands that were skipped and why.
- Mention any GPU/Docker/private-access assumptions.
- Keep PRs focused. Avoid unrelated formatting churn in large generated or
  vendored areas.
