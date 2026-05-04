OSMO Deployment
===============

This guide covers deploying Video Ingestion Agent at scale using Docker containers and the OSMO
cluster.

Docker
------

The Dockerfile builds on a customized ``cosmos_reason_2`` base image, which includes
PyTorch and CUDA pre-installed. vLLM and pipeline dependencies are installed on top.

Build
^^^^^

.. code-block:: bash

   docker build -t nvcr.io/nvstaging/isaac-amr/video_ingestion_agent:latest .

The base image defaults to ``nvcr.io/nvstaging/isaac-amr/cosmos_reason_2`` and can be
overridden with ``--build-arg BASE_IMAGE=<image>``.

Push
^^^^

.. code-block:: bash

   docker push nvcr.io/nvstaging/isaac-amr/video_ingestion_agent:latest

.. note::

   **Model Weights**

   Model weights are **not** bundled in the image. The vLLM server downloads them at runtime
   via ``HF_TOKEN``. This keeps the image small and allows swapping models without rebuilding.

Run Locally with Docker
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   docker run --gpus all \
     -e HF_TOKEN=$HF_TOKEN \
     -v /path/to/videos:/data/videos \
     -v /path/to/outputs:/outputs \
     -p 8000:8000 -p 7860:7860 \
     nvcr.io/nvstaging/isaac-amr/video_ingestion_agent:latest \
     bash -c "python scripts/serve.py -c configs/ingestion.yaml && python scripts/run_webapp.py"

OSMO Cluster
------------

OSMO workflows are defined in ``osmo_workflows/`` and submitted via ``scripts/run_osmo.py``.

Available Workflows
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Workflow
     - Template
     - Description
   * - ``batch_ingestion``
     - ``batch_ingestion.yaml``
     - Multi-GPU batch video ingestion
   * - ``benchmark``
     - ``benchmark.yaml``
     - EPIC-KITCHENS benchmark evaluation
   * - ``webapp``
     - ``webapp.yaml``
     - Web interface with vLLM server

Batch Ingestion
^^^^^^^^^^^^^^^^

Process a large video dataset across multiple GPUs:

.. code-block:: bash

   python scripts/run_osmo.py batch_ingestion \
     --experiment-name epic_kitchens_v1 \
     --output-base-dir /mnt/nfs/outputs \
     --num-shards 8

**What this does:**

1. Discovers all videos in the input directory
2. Computes video durations for load balancing
3. Shards videos across N GPUs using Longest Processing Time (LPT) first scheduling
4. Each shard runs the ingestion pipeline on its subset
5. All shards write to a single shared ``graph.db`` and ``vector.db``
6. Resume is supported — restarted shards skip already-processed videos

**Output:** ``<output-base-dir>/<experiment-name>/graph.db``

Benchmark
^^^^^^^^^^

Run the EPIC-KITCHENS benchmark:

.. code-block:: bash

   python scripts/run_osmo.py benchmark \
     --experiment-name epic_kitchens_eval

**What this does:**

1. Clones EPIC-KITCHENS annotations
2. Symlinks videos from NFS storage
3. Starts vLLM server with tensor parallelism (TP=8)
4. Runs the benchmark pipeline
5. Keeps the container alive for result inspection

Web App
^^^^^^^^

Deploy the web interface on the cluster:

.. code-block:: bash

   python scripts/run_osmo.py webapp \
     --experiment-name my_demo \
     --nfs-db-dir /mnt/nfs/database

**What this does:**

1. Starts a vLLM server for inference
2. Symlinks the NFS entity graph database (graph.db, vector.db)
3. Launches the Gradio UI on port 7860

Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^

A development environment template is available at ``osmo_workflows/dev_env.yaml``.
Submit it directly via the ``osmo`` CLI:

.. code-block:: bash

   osmo workflow submit osmo_workflows/dev_env.yaml \
     --set workflow_name="dev_session" \
     --pool <your-pool>

Multi-GPU Scaling
-----------------

Tensor Parallelism (Single Video)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For faster inference on a single video, use tensor parallelism to split the VLM across
multiple GPUs. Set ``vllm_tp_size`` in your config:

.. code-block:: yaml

   models:
     vllm_tp_size: 8   # Shard model across 8 GPUs

Or override via CLI when starting the server:

.. code-block:: bash

   python scripts/serve.py -c configs/ingestion.yaml --tp 8

This is useful for large VLMs that don't fit on a single GPU, or when you want faster
inference on individual videos (e.g., the benchmark workflow uses TP=8).

Data Parallelism (Batch Processing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For processing many videos, the batch ingestion script spawns multiple worker
sub-processes, each pinned to a separate GPU via ``CUDA_VISIBLE_DEVICES``:

.. code-block:: bash

   python scripts/run_batch_ingestion.py \
     --input-dir /path/to/videos \
     -c configs/batch_ingestion.yaml \
     --output-dir runs/batch \
     --num-shards 8 --resume

All workers share a **single vLLM server** for VLM inference (TP=8 by default in
``configs/batch_ingestion.yaml``) and use their assigned GPU for local tasks
(SigLIP embeddings, frame extraction). The script handles video discovery,
duration-aware LPT sharding, and worker management internally.

Monitoring
----------

vLLM Server
^^^^^^^^^^^^

.. code-block:: bash

   # Check server status
   python scripts/serve.py --status

   # View server logs
   python scripts/serve.py --logs

Batch Ingestion Progress
^^^^^^^^^^^^^^^^^^^^^^^^^

Each worker writes a log file:

.. code-block:: text

   <output-dir>/worker_<id>.log

A final summary is written when all videos in the worker are processed:

.. code-block:: text

   <output-dir>/summary_worker_<id>.json

See Also
--------

- :doc:`/pages/configuration` — YAML config options for models, segmentation, and batch settings
- :doc:`/pages/ingestion_pipeline` — Pipeline stages and batch ingestion details
- :doc:`/pages/model_backends` — Choosing between local, vLLM, and API backends
- :doc:`/pages/troubleshooting` — Common issues and solutions
