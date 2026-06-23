# Video Ingestion Agent

**Agentic workflow for ingesting robot demonstration videos into a queryable
action database.**

A LangGraph-driven pipeline that turns long demonstration videos into
temporally-bounded action clips, an entity-relation scene graph, and
visual embeddings — all queryable via a natural-language retrieval
agent or an interactive web UI.

---

## What it does

Given a directory of robot demonstration videos, the pipeline produces
two SQLite databases:

- **`graph.db`** — an entity scene graph: typed entities (person, object,
  location), pairwise relationships (picks-up, places-on, …), and
  temporally-bounded action segments per source video.
- **`vector.db`** — per-frame SigLIP-2 embeddings tagged with the
  `segment_id` of the action they belong to, so visual similarity hits
  resolve back to exact clip boundaries.

A separate retrieval agent then takes natural-language queries and
returns clip files plus answers, e.g.

> **Q:** *"Find clips where the operator picks up a mug."*
>
> **A:** Returns 3 clips, each with `[start_t, end_t]` boundaries and a
> rendered `.mp4` extracted from the source video.

### Pipeline at a glance

```
ingest:   segment → verify → refine → entity-extract → embed → DB write → report
retrieve: decompose-task → search-graph + search-frames → relax → synthesize
```

Both flows are LangGraph state graphs; nodes are pure functions over a
typed state, easy to extend or swap.

### Highlighted features

- **Chunk-based VLM segmentation** with a verify/refine loop — a critic
  VLM reviews each extracted clip and triggers re-annotation when the
  segmentation disagrees with the action.
- **Cross-database resolution** — visual hits in `vector.db` carry a
  `segment_id` that maps back to action segments in `graph.db`, so a
  CLIP-style nearest-neighbor query returns precise clip boundaries
  rather than fuzzy frame ranges.
- **EGAgent-style retrieval** with task decomposition and progressive
  query relaxation (4 levels, exact → similar action/object).
- **Sharded batch ingestion** — duration-aware LPT scheduling across
  multiple GPUs into a shared `graph.db` / `vector.db` using SQLite WAL.
- **Pluggable model backends** — vLLM (production), local HuggingFace
  (development), or remote API (NVIDIA NIM / OpenAI-compatible).

## Requirements

- **OS:** Linux (tested on Ubuntu 22.04+).
- **Python:** 3.10 or 3.11.
- **GPU:** NVIDIA with CUDA 12.x for the default vLLM backend; an 8B-class
  VLM (e.g. `Qwen/Qwen3-VL-8B-Instruct`) needs roughly 24 GB of VRAM.
- **System:** `ffmpeg` on `PATH`.
- **Auth:** a HuggingFace token (`HF_TOKEN`) for the VLM and SigLIP weight
  downloads.

## Installation

Recommend [`uv`](https://docs.astral.sh/uv/) for environment management:

```bash
git clone https://github.com/nvidia-isaac/video_to_data.git
cd video_to_data/video_ingestion_agent

# Recommended: lock-aware sync — same versions as the Dockerfile and CI.
uv sync --all-extras
source .venv/bin/activate
```

`uv sync` reads `uv.lock` and creates `.venv` with the integration-tested
versions (vLLM, torch, transformers, ...). Pass `--frozen` to make it fail
loudly if `pyproject.toml` has drifted from the lock.

For tighter environments, mix and match the per-feature extras:

```bash
uv sync                                 # core only (no GPU, no UI)
uv sync --extra server                  # vLLM (production inference backend)
uv sync --extra local                   # in-process HuggingFace inference
uv sync --extra webapp                  # Gradio UI
uv sync --extra benchmark               # EPIC-KITCHENS evaluation
uv sync --extra dev                     # tests, ruff, mypy
```

## Quickstart

The full ingestion pipeline runs end-to-end on a single video in under
two minutes (excluding initial model download):

```bash
# 1. Start the vLLM server in the background (loads the VLM, ~1 minute)
python scripts/serve.py -c configs/ingestion.yaml

# 2. Run the pipeline on one video — segmentation → entity graph → report
python scripts/run_ingestion.py path/to/video.mp4 -c configs/ingestion.yaml

# 3. Query the resulting database
python scripts/run_retrieval.py "Find clips where someone picks up a mug" \
    -d outputs/ -c configs/retrieval.yaml

# 4. Or browse interactively in the web UI
python scripts/run_webapp.py
```

Stop the server with `python scripts/serve.py --stop`.

> **Note on the verify/refine loop.** The shipped `configs/ingestion.yaml`
> sets `enable_verification: false` and `enable_refinement: false` for fast
> first-run iteration. To exercise the full LangGraph pipeline (critic VLM
> reviews each clip and triggers re-annotation when needed), copy the config
> and flip both toggles to `true`, then add `--max-iterations 3` to the
> `run_ingestion.py` invocation. On short videos the critic can be strict —
> if 0 clips pass verification the resulting database is empty.

For batch ingestion across many videos and GPUs:

```bash
python scripts/run_batch_ingestion.py --input-dir <videos> \
    -c configs/batch_ingestion.yaml --output-dir runs/batch --num-shards 8 --resume
```

## Documentation

Full documentation builds from `docs/` with Sphinx — install
`uv pip install -e ".[docs]"` then run `cd docs && make html`. Highlights:

- `docs/pages/architecture.rst` — design overview and pipeline diagram.
- `docs/pages/ingestion_pipeline.rst` — stage-by-stage walkthrough.
- `docs/pages/retrieval_agent.rst` — the LangGraph retrieval flow.
- `docs/pages/database_design.rst` — SQLite schemas for `graph.db` and
  `vector.db`.
- `docs/pages/configuration.rst` — full reference for the YAML configs.
- `docs/pages/model_backends.rst` — picking and configuring inference
  backends.

## License

Dual-licensed under [CC-BY-4.0 AND Apache-2.0](LICENSE): source code
under Apache-2.0, documentation and mixed-content files under
CC-BY-4.0. Third-party dependency notices are in [NOTICE](NOTICE).

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
PR workflow and style guide. **Every commit must be signed off via
`git commit -s`** under the
[Developer Certificate of Origin](https://developercertificate.org/) —
see the [Signing Your Work section](CONTRIBUTING.md#signing-your-work)
for details.

## Citation

A paper describing the system is in preparation. Until then, please
cite this repository directly:

```bibtex
@misc{video_ingestion_agent,
  title  = {Video Ingestion Agent: Agentic workflow for robot demonstration video ingestion},
  author = {{NVIDIA Isaac Team}},
  year   = {2026},
  url    = {https://github.com/nvidia-isaac/video_to_data/tree/main/video_ingestion_agent},
}
```
