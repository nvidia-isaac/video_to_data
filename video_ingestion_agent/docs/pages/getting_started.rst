Getting Started
===============

This guide walks you through installing Video Ingestion Agent, starting the inference server,
and running your first video through the pipeline.

.. tip:: **Prefer a guided, interactive walkthrough?**

   If you're using Claude Code in the parent ``video_to_data/`` workspace, two
   skills cover this guide end-to-end and are version-controlled at
   ``video_to_data/.claude/skills/``:

   - ``/ingestion_agent_onboard`` — drives first-run setup one step at a time:
     triage, pick an inference backend (vLLM / local / API), install, start
     the server (only if you chose vLLM), run your first ingest, query the
     database, tour the webapp, and optionally batch-ingest. Verifies between
     steps so failures don't compound.
   - ``/ingestion_agent_doctor`` — diagnostic checklist for when something is
     broken or before a first run: vLLM health, GPU/driver, HuggingFace auth,
     database integrity, Python environment, run-dir sanity.

   The rest of this page is the manual reference for the same flow.

What You'll Build
-----------------

By the end of this guide you will have processed a video through the full pipeline:

.. code-block:: text

   $ python scripts/run_ingestion.py demo.mp4 -c configs/ingestion.yaml

   Pipeline Summary:
     Video: demo.mp4
     Total clips: 23
     Iterations: 2
     Final status: completed
     Verified: 21/23 valid (91.3%)
     graph.db: outputs/graph.db
     vector.db: outputs/vector.db
     Report: file:///home/user/runs/20260217_143022/report.html

   $ python scripts/run_retrieval.py "Find all pick up mug actions" \
       -d outputs/

   ANSWER:
   Found 3 clips of picking up a mug:
     1. [12.5s - 16.2s] Person picks up white mug from counter
     2. [45.0s - 48.8s] Person picks up white mug from table
     3. [102.3s - 106.1s] Person picks up red mug from drying rack

   EXTRACTED CLIPS:
     - outputs/clips/task_1_pick_up_mug_1.mp4
     - outputs/clips/task_1_pick_up_mug_2.mp4
     - outputs/clips/task_1_pick_up_mug_3.mp4

Prerequisites
-------------

- **Python 3.10+**
- **FFmpeg** (for video processing)
- **NVIDIA GPU** with CUDA support (recommended for VLM inference)
- **uv** package manager (`uv docs <https://docs.astral.sh/uv/>`_)

.. tip:: **Installing uv**

   If you don't have ``uv`` installed:

   .. code-block:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

   Or via pip: ``pip install uv``

Installation
------------

1. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^

:git_clone_code_block:

2. Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^

The repository ships a ``uv.lock`` file pinning the integration-tested
versions of vLLM, torch, transformers, and the rest of the dependency
tree. ``uv sync`` reads it, creates ``.venv``, and installs the project
in editable mode — the same flow used by the Dockerfile and CI.

.. tabs::

   .. tab:: Core (Recommended)

      The base install is lightweight and uses a vLLM server for VLM inference.
      This is the recommended setup.

      .. code-block:: bash

         uv sync
         source .venv/bin/activate

   .. tab:: All Features

      Install everything — local models, web UI, benchmark tools, and development utilities.

      .. code-block:: bash

         uv sync --all-extras
         source .venv/bin/activate

   .. tab:: Pick and Choose

      Install only the extras you need:

      .. code-block:: bash

         # vLLM server (install in server environment)
         uv sync --extra server

         # Local HuggingFace model inference (requires GPU)
         uv sync --extra local

         # Web interface (Gradio)
         uv sync --extra webapp

         # EPIC-KITCHENS benchmark evaluation
         uv sync --extra benchmark

         # Development tools (pytest, ruff, mypy)
         uv sync --extra dev

         # Documentation (Sphinx)
         uv sync --extra docs

         # Visualization (matplotlib, plotly)
         uv sync --extra viz

.. tip::

   Pass ``--frozen`` (e.g. ``uv sync --all-extras --frozen``) to make the
   install fail loudly if ``pyproject.toml`` has drifted from the lock —
   this is what CI and the Dockerfile use for strict reproducibility.

Dependency Groups
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Extra
     - What it adds
     - When you need it
   * - ``[server]``
     - vLLM
     - Running the VLM inference server
   * - ``[local]``
     - torch, transformers, timm, bitsandbytes
     - Running models locally without a server
   * - ``[webapp]``
     - Gradio, Plotly, NetworkX
     - Interactive web interface
   * - ``[benchmark]``
     - sentence-transformers, BERTScore, pandas, wandb
     - EPIC-KITCHENS evaluation
   * - ``[dev]``
     - pytest, ruff, mypy, pre-commit
     - Development and testing
   * - ``[docs]``
     - Sphinx, nvidia-sphinx-theme
     - Building this documentation
   * - ``[viz]``
     - matplotlib, seaborn, plotly
     - Visualization utilities
   * - ``[all]``
     - Everything above
     - Full installation

Starting the vLLM Server
-------------------------

The pipeline uses a vLLM server for fast VLM inference. The server runs as a background
daemon and persists across pipeline runs.

.. code-block:: bash

   # Start the server (reads model config from YAML)
   python scripts/serve.py -c configs/ingestion.yaml

   # Check if the server is running
   python scripts/serve.py --status

   # View server logs
   python scripts/serve.py --logs

   # Stop the server
   python scripts/serve.py --stop

The server downloads model weights on first run via HuggingFace. Set the
``HF_TOKEN`` environment variable if the model requires authentication
(both the default ``Qwen/Qwen3-VL-8B-Instruct`` and SigLIP-2 are gated):

.. code-block:: bash

   export HF_TOKEN=hf_xxx          # from https://huggingface.co/settings/tokens
   python scripts/serve.py -c configs/ingestion.yaml

.. note::

   **Multi-GPU Tensor Parallelism**

   For multi-GPU setups (e.g., 8x H100), set ``vllm_tp_size`` in your config:

   .. code-block:: yaml

      models:
        vllm_tp_size: 8

   Or override via CLI:

   .. code-block:: bash

      python scripts/serve.py -c configs/ingestion.yaml --tp 8

Running Your First Pipeline
----------------------------

Process a single video through segmentation, verification, entity graph building,
and report generation:

.. code-block:: bash

   python scripts/run_ingestion.py path/to/video.mp4 -c configs/ingestion.yaml

Output
^^^^^^

The pipeline creates two output locations:

- A **timestamped run directory** under ``runs/`` (configurable via ``paths.runs_dir``)
  for pipeline artifacts (logs, clips, reports).
- A **database directory** at ``outputs/`` (configurable via ``database.directory``)
  for the entity graph and vector databases used by the retrieval agent.

.. code-block:: text

   runs/20260217_143022/           # Run artifacts
   ├── clips_stage1.jsonl          # Initial segmented action clips
   ├── clips_verified.jsonl        # Clips after critic verification
   ├── clips_final.jsonl           # Final clips after refinement
   ├── critic_responses/           # Detailed critic feedback per clip
   ├── report.html                 # HTML summary report
   └── pipeline.log                # Execution log

   outputs/                        # Database directory (database.directory)
   ├── graph.db                    # Entity graph database
   └── vector.db                   # Visual embeddings for search

Pipeline Options
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Skip verification for faster iteration
   python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml --no-verify

   # Segmentation only (no entity graph)
   python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml --no-entity-graph

Querying for Clips
------------------

After ingestion, use natural language to find and extract clips:

.. code-block:: bash

   python scripts/run_retrieval.py "Find all pick up mug actions" \
     -d outputs/ \
     --output-dir outputs/clips/

``-d`` points at the **database directory** containing both ``graph.db``
and ``vector.db`` — the same directory written by ``run_ingestion.py``
(its ``database.directory`` config key). ``--output-dir`` is where the
extracted ``.mp4`` files land; the script defines no short ``-o`` form.

The retrieval agent decomposes your query into sub-tasks, searches the entity graph
and visual embeddings, and extracts the top matching clips.

Batch Ingestion
---------------

For processing large video datasets, the pipeline supports sharded batch
ingestion across multiple GPUs with resume capability:

.. code-block:: bash

   python scripts/run_batch_ingestion.py \
     --input-dir /path/to/videos \
     -c configs/batch_ingestion.yaml \
     --output-dir runs/batch \
     --num-shards 8 --resume

All shards write to a single shared ``graph.db`` and ``vector.db``. See
:doc:`/pages/ingestion_pipeline` for the full batch ingestion guide and
:doc:`/pages/deployment` for running batch ingestion on OSMO.

Launching the Web Interface
---------------------------

For an interactive experience with video upload, database browsing, and querying:

.. code-block:: bash

   python scripts/run_webapp.py

This launches a Gradio UI at ``http://localhost:7860``. See the
:doc:`/pages/webapp` guide for details.

CLI Commands Reference
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Command
     - What it does
   * - ``python scripts/serve.py -c <config>``
     - Start the vLLM inference server
   * - ``python scripts/serve.py -c <config> --gpu-mem 0.7``
     - Start with custom GPU memory utilization (default 0.8)
   * - ``python scripts/serve.py --status``
     - Check if the server is running
   * - ``python scripts/serve.py --stop``
     - Stop the server
   * - ``python scripts/run_ingestion.py <video> -c <config>``
     - Process a single video through the full pipeline
   * - ``python scripts/run_retrieval.py "<query>" -d <db_dir>``
     - Find and extract clips using natural language
   * - ``python scripts/run_webapp.py``
     - Launch the Gradio web interface
   * - ``python scripts/run_batch_ingestion.py --input-dir <dir> -c <config>``
     - Batch-process a directory of videos
   * - ``python scripts/run_benchmark.py -c <config>``
     - Run EPIC-KITCHENS benchmark evaluation

Next Steps
----------

- :doc:`/pages/architecture` — Understand the system design
- :doc:`/pages/configuration` — Customize the pipeline for your videos
- :doc:`/pages/deployment` — Run at scale with Docker and OSMO
