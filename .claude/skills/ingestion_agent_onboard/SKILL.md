---
name: ingestion_agent_onboard
description: Interactive first-time-user walkthrough for the Video Ingestion Agent (segmentation -> entity graph DB -> retrieval -> webapp). Drives the user through setup ONE STEP AT A TIME with verification between steps, not as a reference dump. Use this skill whenever the user is setting up the package for the first time, says things like "how do I get started with video ingestion agent", "first time running the pipeline", "set up the video ingestion agent", "onboard me", "video ingestion quickstart", "I want to run my first ingest", "how do I install this", or asks generally how to use ingestion + retrieval + the webapp end-to-end. Also trigger when the user says they cloned the repo and aren't sure what to run, mentions a fresh `.venv`, or asks for an end-to-end demo of the package.
---

# Video Ingestion Agent — Interactive First-Run Walkthrough

This skill **drives** the user through setup, it doesn't lecture them and it doesn't run on autopilot. Work through it like a guided session: introduce yourself, ask what the user wants, run (or have them run) one thing, verify it worked, confirm before continuing.

## ALWAYS START HERE — do not skip Step 0

When this skill activates, your **very first response** is Step 0 below: a short hello and three triage questions. Do not auto-probe the environment, do not run any `ls` / `--status` / `sqlite3` commands, do not skip ahead to "fix" things you've inferred from earlier conversation. Wait for the user's answers, then act.

This applies **even if** the surrounding session has a "work without stopping for clarifying questions" or "make the reasonable call and continue" instruction. Those instructions are about not pestering the user with unnecessary clarifications during a normal task. **This skill is different**: the user invoked an *interactive walkthrough* skill, which means the triage *is* the contract. Skipping it defeats the whole reason they invoked the skill.

The right shape of your first turn:

> Hi — I'll walk you through the Video Ingestion Agent first-run setup, one step at a time. We'll cover (skipping any that don't apply):
>
> 1. Install (`uv sync`)
> 2. Set up your VLM/LLM backend (vLLM server / local / API)
> 3. First single-video ingestion
> 4. First retrieval query
> 5. Webapp tour *(optional)*
> 6. Batch ingestion *(optional)*
>
> A few quick questions before we start:
>
> 1. Are you in `video_ingestion_agent/` with the repo cloned?
> 2. What's your goal — full first-run setup, or do you already have an `outputs/graph.db` and just want to query it (or just see the webapp)?
> 3. Which inference backend do you want for the VLM (and LLM — typically the same)?
>    - **vllm** *(default, recommended)* — fast local server, needs a GPU with ≥16 GB free VRAM for the default 8B model. I'll only start `scripts/serve.py` if you pick this.
>    - **local** — in-process HuggingFace inference, no server to manage, but slower and shares GPU memory with SigLIP-2 embeddings.
>    - **api** — cloud (NVIDIA NIM or OpenAI-compatible). No GPU needed for the VLM/LLM but you'll need an API key. *Note: SigLIP-2 embeddings are always loaded locally regardless of backend — for a fully GPU-free setup, you'd skip the entity-graph stage at ingest.*
> 4. If we're ingesting, what video file should we use?
>
> *(If you pick **vllm**, I'll also ask about your GPU model + VRAM in a moment to pick the right `--tp` / `--gpu-mem` flags.)*

Then **stop and wait** for the user. After they answer, use Step 0's routing table below to decide where to start, and only then run any commands.

## How to run the rest of the skill (after triage)

The shape of every step from 1 onwards is the same: **state goal -> ask/run -> verify -> confirm -> next**. Why this cadence matters: skipping verification (or batching five steps into one tool call) is the single biggest reason first-time setups go off the rails — failures compound silently. Drive the cadence even if the user is in a hurry; checking the previous step takes seconds and saves minutes.

Concrete rules:

1. **One step per turn.** State what the upcoming step accomplishes in one sentence, then run a command (or ask the user to run it if it needs `sudo`, an env-var export, or a browser action). Don't pre-run the next step's command.
2. **Verify with an explicit check, not vibes.** After every state-changing step, run a check that proves it worked (`python -c "import video_ingestion_agent"`, `serve.py --status`, `sqlite3 ... SELECT COUNT(*)`). The check commands are listed inside each step.
3. **Confirm before continuing.** End each step with something like "That step is done — server is up / clips are written / etc. Ready to move on to <next step>?" Then stop.
4. **Inline troubleshooting on failure.** Each step has its own "If this fails" table. Jump there immediately rather than continuing past a broken state. If the failure isn't in that table, hand off to the `ingestion_agent_doctor` skill.
5. **Adapt to the user.** If they answer Step 0 with "I already installed" or "I just want retrieval", skip past the irrelevant steps. The walkthrough is a guide rail, not a rigid script.

All commands assume the user is in `/home/liuw/Projects/video_to_data/video_ingestion_agent/` with the project's `.venv` activated. Substitute their clone path if different.

## Step 0: Triage (mandatory entry point)

You should already have sent the introduction + four questions above. Once the user answers, route into the right starting step using this table. The backend choice (`vllm` / `local` / `api`) determines what Step 2 looks like — keep it in mind throughout the rest of the walkthrough.

| User says | Action |
|-----------|--------|
| "Just want to retrieve from an existing DB" | Skip Steps 1–3, go to **Step 4: First Retrieval**. Verify the DB is non-empty first (`sqlite3 outputs/graph.db "SELECT COUNT(*) FROM action_segments;"`) |
| "Pipeline is broken, I want to debug" | Stop. Hand off to the `ingestion_agent_doctor` skill |
| "Just want to see the webapp" | Skip to **Step 5: Webapp** (it can run against an empty DB; ingest comes later) |
| "Full setup from scratch" | Continue with **Step 1: Install** |
| "I've already installed" | Skip to **Step 2: Backend setup**. Quick sanity check first: `python -c "import video_ingestion_agent; print('ok')"` |
| "Backend is already configured / server is up" | Skip to **Step 3: First Ingest**. Sanity-check the backend first (vllm: `serve.py --status`; api: `echo $NIM_API_KEY`; local: confirm config has `vlm_backend: local`) |

If they picked the `vllm` backend and we're going to start the server, ask one more question before Step 2: **"What GPU do you have? (model + VRAM, e.g., 'one H100 80GB' or 'two A100 40GB')"**. The shipped `configs/ingestion.yaml` defaults `vllm_tp_size: 8` for the OSMO multi-GPU setup, which fails on a single-GPU machine — you need this answer to pick the right `--tp` and `--gpu-mem` flags.

For `local` or `api` backends, no GPU follow-up is needed at this stage — those have different verification flows in Step 2.

## Step 1: Install

> "We'll install the package in editable mode using `uv sync`. This creates `.venv` and pins versions from `uv.lock`. One-time, takes 1–3 min."

First check `uv` is available:

```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
```

The right install profile depends on the backend the user chose in Step 0. Use this table — don't blindly pick `--all-extras`, which downloads gigabytes of unused stuff:

| Backend chosen in Step 0 + use case | Command |
|--------------------------------------|---------|
| `vllm` backend (running server locally) | `uv sync` (base ships with vLLM client + the package) |
| `local` backend (in-process HuggingFace) | `uv sync --extra local` |
| `api` backend (cloud) | `uv sync` |
| Need the webapp too (any backend) | append `--extra webapp` (e.g., `uv sync --extra local --extra webapp`) |
| Want everything (dev + benchmark + viz + docs + ...) | `uv sync --all-extras` |

If they're going to run the webapp later in Step 5, surface that now and offer to fold `--extra webapp` into the install — saves a second `uv sync`. Note that `[webapp]` is required even for non-UI imports of the package because `webapp/__init__.py` eagerly imports `app`.

Run the chosen command, then activate and verify:

```bash
source .venv/bin/activate
python -c "import video_ingestion_agent; print('install ok')"
```

Pass: prints `install ok`.

### If this step fails

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'gradio'` on `import video_ingestion_agent` | The package's `webapp/__init__.py` eagerly imports `app`, so `[webapp]` is required even for non-UI usage. Run `uv sync --extra webapp` |
| `error: failed to read uv.lock` | The user is in the wrong directory. `cd /home/liuw/Projects/video_to_data/video_ingestion_agent/` and retry |
| Lockfile drift complaints | `uv lock` to regenerate, or `git checkout pyproject.toml uv.lock` to revert local edits |
| Long pause on torch download | First-time install — torch is ~2 GB. Wait |

Confirm before moving on: **"Install verified. Want to start the vLLM server next?"**

### Extras quick-reference (if asked)

| Extra | What it adds | When you need it |
|-------|--------------|------------------|
| `[server]` | vLLM | Running the VLM inference server |
| `[local]` | torch, transformers, timm, bitsandbytes | Running models locally without a server |
| `[webapp]` | Gradio, Plotly, NetworkX | Interactive web interface (also required for any environment that imports the package) |
| `[benchmark]` | sentence-transformers, BERTScore, pandas, wandb | EPIC-KITCHENS evaluation |
| `[dev]` | pytest, ruff, mypy, pre-commit | Development and testing |
| `[docs]` | Sphinx, nvidia-sphinx-theme | Building the docs |
| `[viz]` | matplotlib, seaborn, plotly | Visualization utilities |

## Step 2: Set up Your VLM/LLM Backend

> "Now we'll configure the inference backend you chose in Step 0. Three different paths below — only one applies."

Branch on the user's answer from Step 0:

- They chose **`vllm`** -> follow **Step 2a**
- They chose **`local`** -> follow **Step 2b**
- They chose **`api`** -> follow **Step 2c**

Don't skim all three at the user; pick one and walk it. **Critical: do not start `scripts/serve.py` unless they explicitly chose `vllm`.** Starting a vLLM server they don't want wastes ~40 s and ~16 GB of VRAM.

### Step 2a: vLLM backend (default, recommended)

> "vLLM runs as a background daemon and serves the VLM/LLM via HTTP. Two things to do: set `HF_TOKEN` (the default models are gated on HuggingFace), then start the server."

#### Set HF_TOKEN

First check whether it's already set:

```bash
[ -n "$HF_TOKEN" ] && echo "HF_TOKEN is set" || echo "HF_TOKEN is NOT set"
```

If not set, ask the user to do this in *their* shell (env vars don't persist across your tool calls):

> "Get a token from <https://huggingface.co/settings/tokens> with **read** permission. Then visit each gated model's HF page and click 'Agree' on the license:
>
> - <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
> - <https://huggingface.co/google/siglip2-base-patch16-256>
>
> Then export the token in your shell:
>
> ```
> export HF_TOKEN=hf_xxx...
> ```
>
> Tell me when done and I'll re-check."

Re-run the check above to confirm.

#### Start the server

Pick the right CLI flags based on the user's GPU answer from Step 0:

| User has | Command |
|----------|---------|
| Single GPU with ≥20 GB free | `python scripts/serve.py -c configs/ingestion.yaml` |
| Single GPU with 16–20 GB free (SigLIP will compete) | `python scripts/serve.py -c configs/ingestion.yaml --gpu-mem 0.7` |
| Single GPU with <16 GB | Stop and ask: pick a smaller VLM in config first (e.g., `Qwen/Qwen2-VL-2B-Instruct`), or switch to `api` backend — the default 8B won't fit |
| Multi-GPU (e.g., 8x H100) | `python scripts/serve.py -c configs/ingestion.yaml --tp 8` |

The server downloads model weights on first run (~16 GB for the default 8B VLM, ~1 GB for SigLIP-2). Wait for it to settle, then verify:

```bash
python scripts/serve.py --status
```

Pass: `vLLM server is RUNNING at http://localhost:8000/v1`.

#### If this fails

Run `python scripts/serve.py --logs` and match:

| Log line | Cause | Fix |
|----------|-------|-----|
| Stuck at `loading weights` | First-time HF download | Wait — 16+ GB takes a few minutes |
| `401 Unauthorized` from huggingface | `HF_TOKEN` missing/expired or license not accepted | Re-check token; visit the model HF page in browser and click Agree |
| `Address already in use` / `port 8000` | Something else holds 8000 | `lsof -i :8000`; either stop it or change `vllm_url` in config to a different port |
| `OutOfMemoryError` | Default 8B VLM doesn't fit | Surface to user, propose: `--gpu-mem 0.7` to share with SigLIP, `--tp N` to shard, or pick a smaller VLM. Wait for go-ahead before restarting |
| `Requested more deepstack tokens than available` | vLLM 0.20+ regression on Qwen3-VL | `uv pip install 'vllm==0.15.1'` (the `pyproject.toml` pin should already prevent this; only happens after manual upgrade) |
| `flashinfer-cubin version (X) does not match flashinfer version (Y)` | Library/cubin drift | `FLASHINFER_DISABLE_VERSION_CHECK=1 python scripts/serve.py -c configs/ingestion.yaml`; long-term fix is `uv pip install --reinstall flashinfer*` |

If `--logs` shows nothing useful, hand off to the `ingestion_agent_doctor` skill.

Other useful lifecycle commands:

```bash
python scripts/serve.py --logs    # tail ~/.video_ingestion_agent/vllm.log
python scripts/serve.py --stop    # stop the daemon (no need to do this between runs)
```

**Confirm: "Server is up and reachable. Ready to ingest your first video?"**

### Step 2b: Local backend (in-process HuggingFace)

> "Local runs the VLM/LLM in-process — no server to manage. Slower than vLLM and uses more memory headroom (no PagedAttention), but simpler if you don't want a daemon."

#### Verify the [local] extra is installed

```bash
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

Pass: prints torch version + `cuda: True`. Fail: surface to user — "Local backend needs the `[local]` extra (torch, transformers, timm, bitsandbytes). Want me to install it with `uv sync --extra local`?" Wait for go-ahead before running.

#### Set HF_TOKEN

Same as Step 2a — Qwen3-VL and SigLIP-2 are both gated. Use the HF_TOKEN sub-section above (same check command, same ask, same license URLs).

#### Switch the config to local backend

The shipped `configs/ingestion.yaml` defaults to `vlm_backend: vllm`. The user's options:

- **Edit `configs/ingestion.yaml` in place**: change `models.vlm_backend: vllm` -> `local` and `models.llm_backend: vllm` -> `local`. Simplest if they're not coming back to vllm.
- **Copy to a new file**: `cp configs/ingestion.yaml configs/ingestion_local.yaml`, edit the copy, and pass `-c configs/ingestion_local.yaml` to all later commands. Cleaner if they want to keep both options.

Surface both options and ask which they prefer, then make the edit (or guide them to). Confirm the change took:

```bash
grep -E "vlm_backend|llm_backend" configs/ingestion.yaml   # or whichever file they edited
```

Should show `local`, `local`.

#### If this fails

| Symptom | Cause | Fix |
|---------|-------|-----|
| `import torch` fails | `[local]` extra not installed | `uv sync --extra local` (ask first) |
| `cuda: False` | Driver issue / no GPU | Hand off to `ingestion_agent_doctor` skill, Section 2 |
| `OutOfMemoryError` at ingest time | 8B VLM in-process is heavy | Surface to user: "Local backend OOM'd. Want to switch to vllm (more efficient) or pick a smaller VLM in config?" |
| `401 Unauthorized` from HF on first ingest | HF_TOKEN missing/expired or license not accepted | Same fix as 2a |

No verification command here proves the backend is wired correctly — the next ingest in Step 3 will surface any misconfiguration. **Confirm: "Local backend configured. Ready to ingest your first video?"**

### Step 2c: API backend (NVIDIA NIM / OpenAI-compatible)

> "The API backend offloads the VLM and LLM to a cloud service — no local GPU needed for inference, just an API key. Note: SigLIP-2 embeddings are *always* loaded locally regardless of backend, so if you don't have a GPU at all, you'll need to skip the entity-graph stage at ingest (`--no-entity-graph` in Step 3). Otherwise SigLIP runs on CPU and is slow but works."

#### Get an API key from the user

Surface this exactly to the user:

> "Do you have a NVIDIA NIM API key? If so, get one from <https://build.nvidia.com> and export it in your shell:
>
> ```
> export NIM_API_KEY=nvapi-...
> ```
>
> If you're using a different OpenAI-compatible endpoint (vLLM elsewhere, OpenAI proper, a private gateway), let me know the base URL and I'll wire that into config instead. Tell me when the key is set."

Wait for the user to confirm. Then verify:

```bash
[ -n "$NIM_API_KEY" ] && echo "NIM_API_KEY is set (${NIM_API_KEY:0:8}...)" || echo "NIM_API_KEY is NOT set"
```

#### HF_TOKEN — only if they want the entity graph

Ask: "Do you want the entity graph + visual search at ingest (recommended), or skip it (`--no-entity-graph`)?"

- **Yes (default)**: SigLIP-2 is gated on HF, so set `HF_TOKEN` — same flow as 2a, but only the SigLIP-2 license matters here (you can skip the Qwen3-VL license accept since you're not using it locally).
- **No**: skip the HF_TOKEN ask; remind the user to pass `--no-entity-graph` at Step 3.

#### Switch the config to api backend

Edit `configs/ingestion.yaml` (or a copy):

```yaml
models:
  vlm_backend: api
  llm_backend: api
  # If using a non-NIM endpoint:
  # api_base_url: https://your-endpoint/v1
  # If you'd rather put the key in YAML than env (env is preferred):
  # api_key: nvapi-...
```

Make the edit (or guide them to), then verify:

```bash
grep -E "vlm_backend|llm_backend|api_base_url" configs/ingestion.yaml
```

Should show `api`, `api`.

#### If this fails

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NIM_API_KEY environment variable not set` at ingest | Env var not exported in the same shell that runs ingest | Re-export in the right shell |
| `401 Unauthorized` from the API | Wrong key, expired, or model access not granted on the user's NIM account | Surface to user with the API host's error body |
| Slow per-call latency | API roundtrip is inherent | For fast iteration, vllm is faster. For one-off batch runs, API is fine and avoids GPU provisioning |
| Model not found at endpoint | The default `models.vlm_model` (`Qwen/Qwen3-VL-8B-Instruct`) may not be served by every API host | Update `models.vlm_model` to whatever the endpoint actually serves; check the host's model listing |
| `models.api_base_url` defaulting to NIM but user is on OpenAI | Need to set `api_base_url` explicitly | Add `api_base_url: https://api.openai.com/v1` (or wherever) |

No verification command here either — the next ingest will be the proof. **Confirm: "API backend configured. Ready to ingest your first video?"**

## Step 3: First Single-Video Ingest

> "Now we'll run a single video through the full pipeline: segmentation -> verify (critic loop) -> entity graph -> SigLIP embeddings -> SQLite. End state: a `runs/<ts>/` directory with logs and a report, and a populated `outputs/graph.db` + `outputs/vector.db`."

Use the video path from Step 0 triage:

```bash
python scripts/run_ingestion.py <video_path> -c configs/ingestion.yaml
```

If the user wants to iterate fast and skip verification (useful while tuning prompts):

```bash
python scripts/run_ingestion.py <video_path> -c configs/ingestion.yaml --no-verify
```

Available stage toggles (any combination):

| Flag | What it skips |
|------|---------------|
| `--no-verify` | Critic verify loop |
| `--no-refine` | Refinement (verify only, no loop) |
| `--no-entity-graph` | Entity extraction + DB write (segmentation only) |
| `--no-report` | HTML report |
| `-o runs/<name>` | Override the run directory |

Wait for the pipeline to finish — for a 60-second video at default settings, expect a few minutes. The terminal prints a final summary:

```
Pipeline Summary:
  Video: demo.mp4
  Total clips: 23
  Verified: 21/23 valid (91.3%)
  graph.db: outputs/graph.db
  vector.db: outputs/vector.db
  Report: file:///.../runs/<ts>/report.html
```

Verify the database actually got written:

```bash
sqlite3 outputs/graph.db "SELECT COUNT(*) FROM action_segments;"
sqlite3 outputs/graph.db "SELECT COUNT(*) FROM entities;"
```

Pass: both non-zero. Tell the user the counts.

### If this step fails

| Symptom | Fix |
|---------|-----|
| Backend errors at ingest start (`ConnectionError`, `401 Unauthorized`, `import torch` failure) | Backend-specific: **vllm**: server crashed — `serve.py --status` / `--logs`. **local**: `import torch; torch.cuda.is_available()` should be True. **api**: re-verify `NIM_API_KEY` is exported in *this* shell and the model named in `models.vlm_model` is served by the endpoint |
| `Total clips: 0` in summary | Segmentation produced nothing. Look at the actual VLM responses logged in `runs/<ts>/pipeline.log` — usually the VLM returned malformed JSON or an empty array. Try a smaller `chunk_size` or different `vlm_fps` |
| Clips look way too short or way too long | Tune `segmentation.min_clip_s` / `max_clip_s` in the config. Kitchen tends to want 1–15 s; robot manipulation 2–10 s |
| VLM produces generic labels ("object", "container") | Increase `models.vlm_fps` (more frames -> better visual context); or override `segmentation.system_prompt` with domain-specific guidance |
| `entities` count is 0 but `action_segments` is non-zero | Entity-graph stage was skipped (`enable_entity_graph: false` in config or `--no-entity-graph` flag) or failed. Check `pipeline.log` for entity-extractor errors |
| Pipeline takes much longer than expected | Try `--no-verify` first (skips critic); reduce `vlm_fps`; increase `chunk_size` (uses more VRAM per call but fewer calls) |

Confirm before moving on: **"Ingest complete with N clips and M entities written. Want to query them now?"**

## Step 4: First Retrieval Query

> "Now we'll query the database in natural language. The retrieval agent decomposes your query into sub-tasks, searches the entity graph and the SigLIP embeddings, and extracts the matching clip files."

Ask the user what they want to find. Examples to suggest if they don't know:

- *"Find all pick up <object> actions"*
- *"Show me when the person uses <tool>"*
- *"A hand reaching toward something red"*

Then run:

```bash
python scripts/run_retrieval.py "<user_query>" \
  -d outputs/ \
  --output-dir outputs/clips/
```

Key flags:

| Flag | Meaning |
|------|---------|
| `-d <dir>` | Database directory (the same `database.directory` from Step 3) |
| `--output-dir <dir>` | Where extracted `.mp4` clips land. **No `-o` shorthand** — the script doesn't define one |
| `-c <config>` | Optional — override default `configs/retrieval.yaml` |

Expected output shape:

```
ANSWER:
Found 3 clips of picking up a mug:
  1. [12.5s - 16.2s] Person picks up white mug from counter
  2. [45.0s - 48.8s] Person picks up white mug from table
  3. [102.3s - 106.1s] Person picks up red mug from drying rack

EXTRACTED CLIPS:
  - outputs/clips/task_1_pick_up_mug_1.mp4
  ...
```

### If retrieval returns no results

The agent automatically broadens searches across four relaxation levels (exact match -> any similar action/object). If you still get nothing:

1. **Confirm the DB has data** (you may have already done this in Step 3):
   ```bash
   sqlite3 outputs/graph.db "SELECT COUNT(*) FROM action_segments;"
   ```
   Zero -> the ingest ran with `enable_entity_graph: false` or failed silently. Re-run ingest.
2. **Rephrase action verbs**: "grab" instead of "pick up", "place" instead of "put down", "use" instead of "operate".
3. **Drop adjectives**: "mug" instead of "white mug", "tool" instead of "metal kitchen tool".
4. **Try visual phrasing**: queries with visual descriptors ("a hand reaching for something red") favor the SigLIP frame embeddings over the structured graph.
5. **Check video paths in the DB**:
   ```bash
   sqlite3 outputs/graph.db "SELECT video_path FROM video_metadata;"
   ```
   If paths look wrong (e.g., absolute paths from a different machine), retrieval can't find the source video for clip extraction.

Confirm before moving on: **"You ran a query and got <N> clips. Want to try the webapp UI, batch-process more videos, or wrap up here?"**

## Step 5: Webapp Tour (optional)

Ask: **"Try the Gradio web UI? It's nicer for browsing the database and iterating on queries. Or skip to batch ingestion?"**

If yes:

```bash
python scripts/run_webapp.py
# or with custom port / config / public link:
python scripts/run_webapp.py --port 7860 --config configs/webapp.yaml
python scripts/run_webapp.py --share
```

This launches at `http://localhost:7860` with **four tabs**. Walk the user through each briefly:

| Tab | What's there |
|-----|--------------|
| **Retrieve** | Same as `run_retrieval.py` but in the browser. Each result row has a "Reconstruct" button that optionally forwards the clip to the Reconstruct tab |
| **Database** | Visual browse of stored entities, relationships, and action segments |
| **Ingest** | Upload a video and run ingestion through the UI. Good for one-off experiments; for many videos, use `run_batch_ingestion.py` |
| **Reconstruct** | **Optional integration** with the sibling `reconstruction` package — runs a 16-stage hand+object reconstruction pipeline (~10–15 min/clip). Setup is non-trivial (weights staging, container builds, two-venv isolation). See `src/video_ingestion_agent/reconstruction_interface/ego_e2e/README.md`. The rest of the webapp works without this tab being wired up — don't push the user to set it up unless they ask |

### If the webapp fails to launch

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'gradio'` | `uv sync --extra webapp` |
| `OSError: [Errno 98] Address already in use` | Another service on 7860; `python scripts/run_webapp.py --port 7861` |
| Reconstruct tab errors on click | The reconstruction integration isn't wired up (it's optional). The other tabs still work; ignore unless they want to set it up |

Confirm before moving on: **"Webapp tour done. Ready for batch ingestion, or wrap up?"**

## Step 6: Batch Ingestion (optional)

Ask: **"Do you have a directory of videos to batch-process across multiple GPUs? If not, we can wrap up."**

If yes, get the input directory and number of shards (typically = number of GPUs):

```bash
python scripts/run_batch_ingestion.py \
  --input-dir /path/to/videos \
  -c configs/batch_ingestion.yaml \
  --output-dir runs/batch \
  --num-shards 8 --resume
```

Mention the key facts:

- All shards write to a **single shared** `graph.db` and `vector.db`. SQLite WAL mode handles concurrent writes safely.
- `--resume` skips videos whose `clips_final.jsonl` already exists in the output dir, so interruption + restart is safe.
- `configs/batch_ingestion.yaml` assumes a single TP=8 vLLM server shared by all shards.
- For OSMO-based / Docker deployment, see `docs/pages/deployment.rst`.

### If this step fails

| Symptom | Fix |
|---------|-----|
| All shards crash with `ConnectionError: vLLM server` | The shared server can't keep up. Check `serve.py --status`; consider lowering `--num-shards` |
| `disk I/O error` on graph.db | WAL mode requires the filesystem to support `fsync`. If you're on NFS, use a local disk for `outputs/` |
| One shard's videos all fail with the same VLM error | The model returned malformed JSON for those video durations. Check `runs/batch/shard-N/pipeline.log` |

Confirm: **"Batch run started — kicked off N shards. The script will report per-shard progress. You can interrupt with Ctrl+C and `--resume` later."**

## Wrap-Up

Once the user has gotten through whichever steps they wanted, point them at the docs for what to explore next:

| Topic | Where to go |
|-------|-------------|
| System design (LangGraph, two-pipeline architecture) | `docs/pages/architecture.rst` |
| Full config reference (every YAML key) | `docs/pages/configuration.rst` |
| Backends (`vllm` vs `local` vs `api`/NIM) | `docs/pages/model_backends.rst` |
| Ingestion pipeline deep dive | `docs/pages/ingestion_pipeline.rst` |
| Retrieval agent deep dive | `docs/pages/retrieval_agent.rst` |
| Database schema | `docs/pages/database_design.rst` |
| Custom prompts | `src/video_ingestion_agent/ingestion/segmentation/prompts.py`, `ingestion/entity_graph/prompts.py`, `retrieval/nodes/prompts.py` |
| EPIC-KITCHENS benchmark eval | `docs/pages/benchmark.rst` |
| OSMO / Docker deployment | `docs/pages/deployment.rst` |
| Reconstruction integration (optional) | `src/video_ingestion_agent/reconstruction_interface/ego_e2e/README.md` |
| Diagnose a broken environment | The `ingestion_agent_doctor` skill |

End with something like: **"Setup complete — you have a working pipeline and database. Anything else you want to explore?"**
