---
name: ingestion_agent_doctor
description: Diagnostic checklist for the Video Ingestion Agent environment — run when something is broken, before a first ingest, or as a sanity check. Use this skill whenever the user reports the pipeline isn't working, the vLLM server won't start or crashes during model loading, gets `CUDA out of memory` during ingestion, sees `401 Unauthorized` from HuggingFace, the webapp fails to launch, retrieval returns no results despite a populated database, or asks generally to "diagnose", "debug", "verify", or "check" the video ingestion setup. Also trigger on phrases like "ingestion environment broken", "is my install correct", "vllm server health", or descriptions of half-failing runs (`runs/<ts>/` directory exists but `report.html` is missing).
---

# Video Ingestion Agent — Environment Doctor

Run through these checks in order when something's wrong, or before a first run as a sanity sweep. Each step has a command, what a pass looks like, and what to do on a fail. Stop at the first failing check and apply its fix before continuing — most failures cascade.

All commands assume the user is in `/home/liuw/Projects/video_to_data/video_ingestion_agent/` (substitute your clone path) and has activated the project's `.venv`.

## Before You Start

Ask the user briefly:

- **What's the symptom?** (server won't start / OOM / no retrieval results / webapp errors / pipeline hangs / unsure)
- **When did it last work?** (never / yesterday / right after I ran X)

Skip ahead to the relevant section if the symptom is specific. Otherwise run sections 1 -> 6 in order. Section 7 covers escalation when nothing here resolves it.

For the full first-time setup, use the `ingestion_agent_onboard` skill instead — this skill assumes the user has at least attempted setup and is troubleshooting.

## 1. vLLM Server Health

The pipeline needs a running vLLM server at `vllm_url` (default `http://localhost:8000/v1`) for the default `vllm` backend. If the user is on the `local` or `api` backend, skip to Section 2.

```bash
python scripts/serve.py --status
```

| Output | Meaning | Action |
|--------|---------|--------|
| `Server is running on port 8000` | Healthy | Continue to Section 2 |
| `Server is not running` | Daemon never started or has stopped | `python scripts/serve.py -c configs/ingestion.yaml` |
| Reports running but `curl http://localhost:8000/v1/models` hangs/fails | Daemon process alive but model load failed | `python scripts/serve.py --logs` and look for the crash |

To inspect the log directly:

```bash
python scripts/serve.py --logs
# or
tail -200 ~/.video_ingestion_agent/vllm.log
```

What to look for:

- `Application startup complete` — fully loaded, ready to serve
- `OutOfMemoryError` — see Section 2
- `401 Unauthorized` from huggingface — see Section 3
- `ValueError: Requested more deepstack tokens than available` — vLLM 0.20+ regression specific to Qwen3-VL. The `pyproject.toml` pin `vllm>=0.15.1,<0.20` avoids it; if hit on a manually upgraded vllm, downgrade: `uv pip install 'vllm==0.15.1'`
- `flashinfer-cubin version (X) does not match flashinfer version (Y)` — bypass with `FLASHINFER_DISABLE_VERSION_CHECK=1 python scripts/serve.py -c configs/ingestion.yaml`. Long-term, align with `uv pip install --reinstall` for flashinfer.

Port collision check:

```bash
lsof -i :8000
```

If something other than the vLLM server holds the port (look at the COMMAND column), either stop it or set `vllm_url` to a different port in your config and restart the server.

## 2. GPU + Driver

The default 8B VLM (`Qwen/Qwen3-VL-8B-Instruct`) needs ~16 GB VRAM at bf16; SigLIP-2 needs another ~2 GB. If both share one GPU, reserve headroom with `vllm_gpu_memory_utilization: 0.7` (or `--gpu-mem 0.7` on the CLI).

```bash
nvidia-smi
```

Check:

- All expected GPUs visible (matches `vllm_tp_size` in the config)
- Free memory ≥ ~18 GB summed across the TP shard for the default model
- Driver version supports the installed CUDA build

Then verify torch can see the GPUs:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

Pass: `True N <model_name>`. If `False`, the torch install doesn't see the driver — reinstall with `uv sync --extra local --reinstall`, or check the `nvidia-smi` output above first to make sure the driver is actually loaded.

### When OOM is the symptom

| Cause | Fix |
|-------|-----|
| Default 8B model too large for a single GPU | `vllm_tp_size: N` in config (or `--tp N` on CLI) to shard across GPUs |
| SigLIP competing for VRAM with the VLM | `vllm_gpu_memory_utilization: 0.7` (or `--gpu-mem 0.7`) |
| Embedding batch too large | Reduce `embedding_batch_size` to 4 or 8 in config |
| Just need a smaller model | Change `models.vlm_model` to a smaller HF VLM and restart the server |
| Local backend, not vLLM | Switch `models.vlm_backend: vllm` (vLLM is more memory-efficient via PagedAttention) |

## 3. Auth + Weights

Both the default VLM (`Qwen/Qwen3-VL-8B-Instruct`) and the default embedder (`google/siglip2-base-patch16-256`) are gated on HuggingFace and require the user to have accepted the license on their account.

```bash
echo $HF_TOKEN
```

Pass: `hf_xxxx...`. Fail (empty): get a token from <https://huggingface.co/settings/tokens> with **read** permission and `export HF_TOKEN=hf_xxx`. Then visit each gated model's HF page in a browser and click "agree" on the license — having a token isn't enough.

Cache check:

```bash
ls ~/.cache/huggingface/hub/ 2>/dev/null | head -20
```

Pass: subdirectories like `models--Qwen--Qwen3-VL-8B-Instruct/`, `models--google--siglip2-base-patch16-256/`. Fail (empty or missing target model): not yet downloaded — start the server or run an ingest, the download happens lazily on first use.

If you see `401 Unauthorized` in the vLLM log specifically: `HF_TOKEN` is missing, expired, or the account hasn't accepted that specific model's license. Visit the HF model page in a browser to confirm.

## 4. Database Integrity

After at least one successful ingest, two SQLite files live in the database directory (default `outputs/`).

```bash
ls -la outputs/
```

Pass: `graph.db` and `vector.db` exist. During or shortly after an active session, `*.db-wal` and `*.db-shm` may also exist — that's WAL mode and is normal.

Counts in `graph.db`:

```bash
sqlite3 outputs/graph.db "SELECT COUNT(*) FROM video_metadata;"
sqlite3 outputs/graph.db "SELECT COUNT(*) FROM action_segments;"
sqlite3 outputs/graph.db "SELECT COUNT(*) FROM entities;"
sqlite3 outputs/graph.db "SELECT COUNT(*) FROM relationships;"
```

Pass: each non-zero. Fails:

| What's zero | Diagnosis | Action |
|-------------|-----------|--------|
| `video_metadata` | Ingestion never wrote anything | Re-run `scripts/run_ingestion.py`; if it errors, check `runs/<ts>/pipeline.log` |
| `action_segments` (with `video_metadata` non-zero) | Segmentation produced no clips | Check VLM responses in `pipeline.log`; the VLM may be returning empty arrays — try a different chunk size or fps |
| `entities` (with `action_segments` non-zero) | Entity-graph stage skipped or failed | Check ingest was run with `enable_entity_graph: true` (default); look in `pipeline.log` for entity-extractor errors |

Visual embeddings:

```bash
sqlite3 outputs/vector.db "SELECT COUNT(*) FROM frame_embeddings;"
```

Non-zero -> visual search is wired up. Zero -> SigLIP didn't run; visual queries will return nothing but graph queries still work.

For retrieval to resolve visual hits to clip boundaries, frame embeddings must be tagged with `segment_id`:

```bash
sqlite3 outputs/vector.db "SELECT segment_id FROM frame_embeddings WHERE segment_id IS NOT NULL LIMIT 5;"
```

Should return non-empty IDs that match `action_segments.segment_id` in `graph.db`. If empty, either the DB is from an older version that didn't tag embeddings, or the ingest skipped the entity graph (which is what generates the segments). Re-run ingest with `enable_entity_graph: true`.

## 5. Python Environment

```bash
pip list | grep -E "vllm|torch|transformers|gradio|sentence-transformers|pre-commit"
```

Expected (versions vary; the question is "is it installed at all"):

| Package | Required for | When |
|---------|--------------|------|
| `vllm` | `[server]` extra; vLLM backend | Server-side machine only |
| `torch` | `[local]` extra; SigLIP embedder is always loaded locally | Always |
| `transformers` | tokenizers / SigLIP | Always |
| `gradio` | `[webapp]` extra | Required even for non-UI imports — see below |
| `sentence-transformers` | `[benchmark]` extra | Benchmark only |
| `pre-commit` | `[dev]` extra | Dev only |

**`[webapp]` is required even if the user never runs the webapp**, because `webapp/__init__.py` eagerly imports `app`. If a non-webapp script fails with `ModuleNotFoundError: No module named 'gradio'`:

```bash
uv sync --extra webapp
```

CI installs `[local,dev,benchmark,webapp]` for the test suite for exactly this reason.

Lockfile drift check:

```bash
uv sync --frozen
```

Pass: completes without changes. Fail (`pyproject.toml` differs from lock): either `uv lock` (regenerate the lock) or `git checkout pyproject.toml uv.lock` (revert local edits). CI runs with `--frozen`, so lock drift is a likely cause of "works on my machine but not in CI" failures.

## 6. Run-Dir Sanity

For any specific failed run:

```bash
ls -la runs/<timestamp>/
```

A successful run has:

| File | What's in it |
|------|--------------|
| `clips_stage1.jsonl` | Initial VLM-segmented clips (1 line per clip) |
| `clips_verified.jsonl` | After critic pass (with `iteration_status` field per clip) |
| `clips_final.jsonl` | Final clips, post-refinement |
| `critic_responses/` | One JSON per clip with detailed critic feedback |
| `report.html` | Summary HTML |
| `pipeline.log` | Stage-by-stage execution log |

A half-completed run is missing some of the above:

- `clips_stage1.jsonl` empty -> segmentation produced nothing. Check VLM responses in `pipeline.log` for empty / malformed JSON
- `clips_verified.jsonl` missing -> verify stage failed or was skipped (`--no-verify`). If the stage was meant to run, look for critic errors in `pipeline.log`
- `clips_final.jsonl` missing but `clips_verified.jsonl` present -> refinement was skipped or crashed
- A `temp_clips/` directory still present -> an interrupted verify/refine cycle. The cleanup is normally idempotent on re-run; if you want to manually clean up, `rm -rf runs/<ts>/temp_clips/`

If `pipeline.log` ends mid-stage without a clean shutdown message, the process was killed (OOM, `Ctrl+C`, kernel OOM-killer). Cross-check with `dmesg | tail -50` for OOM-killer entries.

## 7. When to Escalate

If after the above the issue persists, file a GitHub issue at <https://github.com/nvidia-isaac/video_to_data/issues>. Attach:

| File / output | Where it lives |
|---------------|----------------|
| Pipeline log | `runs/<ts>/pipeline.log` |
| vLLM server log | `~/.video_ingestion_agent/vllm.log` |
| Config used | `configs/ingestion.yaml` (or whichever was passed to `-c`) |
| Python env snapshot | `pip freeze` (or `uv pip freeze`) output |
| GPU info | `nvidia-smi` output |
| OS / driver | `uname -a`, `nvidia-smi --query-gpu=driver_version --format=csv` |
| Repro command | The full `python scripts/...` command that failed |

Redact any HuggingFace tokens or API keys before attaching.

For depth on each subsystem the docs are the canonical reference:

- `docs/pages/troubleshooting.rst` — symptom-keyed full troubleshooting guide
- `docs/pages/model_backends.rst` — backend-specific troubleshooting table
- `docs/pages/configuration.rst` — every config key explained
- `docs/pages/architecture.rst` — system design (LangGraph nodes, data flow)
