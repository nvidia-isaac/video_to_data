# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Video Ingestion Agent — entity-graph-based agentic workflow for collecting action clips from
robot demonstration videos. The package lives under `src/video_ingestion_agent/` and is installed
via `pyproject.toml` (uses a `src/` layout; install with `uv sync`, which honors the committed `uv.lock`).

Full Sphinx docs are built under `docs/`. When changing something architectural or
user-facing, check whether the relevant page in `docs/pages/` needs updating.

## Common commands

Tests, lint, formatting, and types (matches the pre-commit + CI config):

```bash
pytest                                         # full suite (coverage configured in pyproject)
pytest tests/test_ingestion.py                 # one file
pytest tests/test_ingestion.py::test_xxx -v    # one test

ruff check . --fix
ruff format .
mypy src/video_ingestion_agent
pre-commit run --all-files
```

Pipeline / agent entry points (all live under `scripts/`):

```bash
# vLLM server lifecycle (used by ingestion + retrieval when backend=vllm)
python scripts/serve.py -c configs/ingestion.yaml      # start
python scripts/serve.py --status | --logs | --stop

# Single-video ingestion: segmentation -> verify/refine -> entity graph -> report
python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml
#   --no-verify / --no-refine / --no-entity-graph / --no-report  toggle stages
#   -o runs/<name>                                                override run dir

# Multi-video ingestion (LPT sharded across GPUs, shared graph.db/vector.db via SQLite WAL)
python scripts/run_batch_ingestion.py --input-dir <dir> -c configs/ingestion.yaml \
    --output-dir runs/batch --num-shards 8 --resume

# Retrieval agent against a built database directory
python scripts/run_retrieval.py "<query>" -d outputs/ -c configs/retrieval.yaml

# EPIC-KITCHENS-100 benchmark
python scripts/run_benchmark.py -c configs/benchmark_epic_kitchens.yaml [--no-verify] [--resume] [--num-gpus N]

# Gradio webapp (extras: webapp)
python scripts/run_webapp.py [--port 7860] [--config FILE] [--share]
```

Docs:

```bash
cd docs && make html        # one-shot build into docs/_build/current/html/
cd docs && make livehtml    # auto-reload server on :8000
```

## Architecture

Two LangGraph state graphs sit at the center of the system. Anything you change in the
ingestion or retrieval flow should fit into the existing graph structure rather than running
as a side script — the graphs are the source of truth.

### Ingestion pipeline (`src/video_ingestion_agent/ingestion/`)

`ingestion_graph.create_pipeline_graph()` wires nodes into:

```
segment → extract_temp → verify → (refine loop) → cleanup_temp →
entity_extract → frame_embed → entity_link → db_write → report
```

- State: `PipelineState` (TypedDict) carrying `clips: list[ClipContext]`,
  `verifications`, `db_paths`, `iteration`, `status`, plus the `PipelineConfig`.
- Nodes are split into `segmentation_nodes.py` (segment / verify / refine / cleanup) and
  `entity_graph_nodes.py` (extract / embed / link / write / report).
- Sub-packages: `segmentation/` (chunked VLM segmenter, critic, refiner, dedup
  strategies — `heuristic` always-merge vs default `llm` ask-the-LLM); `entity_graph/`
  (entity extractors, cross-clip linker, SQLite writer).
- Stage toggles live on the config: `enable_verification`, `enable_refinement`,
  `enable_entity_graph`, `enable_reporting`. The CLI `--no-*` flags flip these.
- Temporary `.mp4` clips are produced only for the verify/refine loop and removed by
  `cleanup_temp` — persistent storage is `video_path + start_t/end_t`, never clip files.

### Retrieval agent (`src/video_ingestion_agent/retrieval/`)

`RetrievalAgent` (in `retrieval_graph.py`) implements an EGAgent-style loop with two
execution modes selected by `config.agent.parallel_tasks`:

- **Parallel** (default): `task_decomposer → Send-per-task → task_search subgraph → vqa_synthesizer`.
  Each sub-task runs its own compiled subgraph from `task_search_subgraph.py`.
- **Sequential**: `task_decomposer → search_planner ⟷ executor ⟷ analyzer → vqa_synthesizer`,
  driven by a `current_task_idx` counter.

Inside each sub-task search there is a hierarchical relaxation loop (4 levels, exact →
similar action/object). The agent has two tools (`tools/search_graph.py`,
`tools/search_frames.py`) plus `extract_clip.py`. Visual hits carry a `segment_id` so
`vector.db` matches resolve back to action segments in `graph.db`.

### Storage

Two SQLite files in `database.directory` (default `outputs/`), both in WAL mode for
concurrent shard writes:

- `graph.db` — `video_metadata`, `entities`, `relationships`, `action_segments`. Frame
  embeddings in `vector.db` are tagged with `segment_id` to bridge the two.
- `vector.db` — per-frame SigLIP-2 embeddings (default `google/siglip2-base-patch16-256`,
  768-dim). Schema and writer in `ingestion/entity_graph/database_writer.py`.

### Model backends (`src/video_ingestion_agent/models/`)

`ModelManager` is the single entry point; all backends implement `generate_text()` and
`generate_from_video()`. Pick via `models.vlm_backend` / `models.llm_backend` in YAML:

- `vllm` — production. Requires `scripts/serve.py` running; supports `vllm_tp_size` for
  tensor parallelism and `vllm_local_media: true` to pass `file://` URLs.
- `local` — in-process HuggingFace (needs `[local]` extra and a GPU).
- `api` — NVIDIA NIM / OpenAI-compatible (uses `NIM_API_KEY` or `models.api_key`).

The frame embedding model (SigLIP-2) is always loaded locally regardless of VLM backend.

### Webapp

`webapp/app.py` builds the Gradio app from `tabs/` (ingestion, query, database, settings)
on top of `services/` (database, ingestion, query) and `components/` (graph + pipeline
visualizers). `scripts/run_webapp.py` is just a launcher.

## Configuration

All config is Pydantic, defined in `ingestion/config.py` (`PipelineConfig`) and
`retrieval/config.py` (`RetrievalConfig`). YAML is loaded via `load_config()` /
`load_retrieval_config()`. Three shipped configs:

- `configs/ingestion.yaml` — single-video pipeline.
- `configs/batch_ingestion.yaml` — many videos sharing one TP=8 vLLM server.
- `configs/retrieval.yaml` — agent settings (steps, temperature, clips dir).
- `configs/benchmark_epic_kitchens.yaml` — EPIC-KITCHENS evaluation.

VLM/critic prompts live in `ingestion/segmentation/prompts.py` and
`ingestion/entity_graph/prompts.py`; YAML overrides take precedence when set. Retrieval
node prompts are in `retrieval/nodes/prompts.py`.

## Conventions worth respecting

- Add nodes by editing the relevant `*_graph.py` builder, not by sidestepping it.
- Keep node functions pure: read from the typed state, return a partial-state dict.
- Don't introduce a new persistent clip-file format — `ClipContext` is `video_path +
  timestamps` by design (the verify/refine loop is the only place temp clips exist).
- New SQLite columns must be added to `database_writer.py` and any reader that touches
  the same table; both DBs assume WAL mode.
- Ruff config lives in `pyproject.toml` (line length 100, target py310, selected rule
  sets `E/W/F/I/B/C4/UP`). Pre-commit also enforces yaml/json validity, end-of-file
  fixer, and a 1 MB max-added-file size.
