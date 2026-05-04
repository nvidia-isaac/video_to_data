Model Backends
==============

Video Ingestion Agent uses three categories of models and provides three interchangeable inference
backends. This page explains how the model system is organized, which models are used where,
and how to choose and configure each backend.

Model Roles
-----------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Role
     - Config Key
     - What It Does
   * - **VLM**
     - ``models.vlm_model``
     - Vision-language model for video segmentation and clip verification. Processes
       sampled video frames + text prompts. Default: ``Qwen/Qwen3-VL-8B-Instruct``.
   * - **LLM**
     - ``models.llm_model``
     - Text-only model for entity extraction from clip descriptions. Falls back to
       the VLM model when set to ``null``.
   * - **Embedding**
     - ``models.embedding_model``
     - Visual embedding model for frame-level semantic search. Always runs locally
       via HuggingFace transformers. Default:
       `SigLIP-2 base <https://huggingface.co/google/siglip2-base-patch16-256>`_
       (768-dim embeddings).

The VLM and LLM roles each specify a **backend** (``vlm_backend`` / ``llm_backend``) that
controls *how* the model is loaded and called. The embedding model always runs locally.


Architecture
------------

.. mermaid::

   flowchart TD
       subgraph ModelManager["ModelManager (Singleton)"]
           direction TB
           Cache["Model Cache\n(one instance per config)"]
       end

       Config["YAML Config\n(vlm_model, vlm_backend)"] --> ModelManager
       ModelManager -->|backend=local| Local["LocalModelWrapper\n(CosmosReasonModel)"]
       ModelManager -->|backend=vllm| VLLM["VLLMModelWrapper\n(OpenAI client)"]
       ModelManager -->|backend=api| API["APIModelWrapper\n(NVIDIA Inference API)"]

       Local -->|generate_from_video| HF["HuggingFace Transformers\n+ torch (GPU)"]
       VLLM -->|OpenAI chat/completions| Server["vLLM Server\n(PagedAttention, TP)"]
       API -->|HTTP POST| Cloud["NVIDIA NIM / OpenAI\n(cloud endpoint)"]

       subgraph BaseModel["BaseModel Interface"]
           direction LR
           M1["generate_text()"]
           M2["generate_from_video()"]
           M3["generate_from_frames()"]
       end

**Key design points:**

- The ``ModelManager`` is a **singleton factory** that caches model instances by
  ``(backend, model_name, device)`` so each model is loaded only once and shared across
  pipeline components.
- All three wrappers implement the same ``BaseModel`` interface (``generate_text``,
  ``generate_from_video``, ``generate_from_frames``), so callers (segmenter, critic, entity
  extractor) are backend-agnostic.
- Convenience functions ``get_local_model()`` and ``get_api_model()`` are available for quick
  interactive use.

**Module:** :code_link:`<src/video_ingestion_agent/models/model_manager.py>`


Backends
--------

local — HuggingFace Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Module:** :code_link:`<src/video_ingestion_agent/models/cosmos_model.py>`

Loads the model weights directly onto the local GPU(s) using HuggingFace ``transformers``.
This is the simplest setup — no server process required — but is the slowest for batch
workloads.

.. code-block:: yaml

   models:
     vlm_model: "Qwen/Qwen3-VL-8B-Instruct"
     vlm_backend: "local"
     device: "cuda"

**How it works:**

1. ``CosmosReasonModel`` loads the model via ``Qwen3VLForConditionalGeneration.from_pretrained``
   with ``bfloat16`` precision and ``sdpa`` attention.
2. Video input: the ``transformers`` processor decodes the video directly from the file path,
   samples frames at the configured ``vlm_fps``, and tokenizes everything into a single
   input tensor.
3. Image input: PIL images are embedded inline as ``image`` content items.
4. Inference runs under ``torch.inference_mode()`` for memory efficiency.

**Requirements:**

- ``pip install video_ingestion_agent[local]`` (installs ``torch``, ``transformers``, ``accelerate``)
- Sufficient GPU VRAM — the 8B model requires ~16 GB in ``bfloat16``
- ``HF_TOKEN`` environment variable for gated models

**When to use:**

- Quick experiments with a single video
- Development and debugging (no server to manage)
- Environments where running a background server is not practical

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Pros
     - Cons
   * - No server process needed
     - Slowest backend (no batching, no PagedAttention)
   * - Simple setup
     - Requires ``[local]`` extras (large install)
   * - Full model control
     - One request at a time (no concurrent inference)


vllm — vLLM Server
^^^^^^^^^^^^^^^^^^^^

**Module:** :code_link:`<src/video_ingestion_agent/models/vllm_model.py>`

Connects to a running `vLLM <https://docs.vllm.ai>`_ server via its OpenAI-compatible API.
vLLM provides **2–5x** faster inference through PagedAttention, continuous batching, and
tensor parallelism across multiple GPUs.

.. code-block:: yaml

   models:
     vlm_model: "Qwen/Qwen3-VL-8B-Instruct"
     vlm_backend: "vllm"
     vllm_url: "http://localhost:8000/v1"
     vllm_local_media: true     # server reads video from disk (fastest)
     vllm_tp_size: 1            # GPUs for tensor parallelism
     vllm_gpu_memory_utilization: 0.8  # Fraction of GPU memory for vLLM

Starting the Server
"""""""""""""""""""

Use the built-in server manager:

.. code-block:: bash

   # Start (reads model name and TP size from config)
   python scripts/serve.py -c configs/ingestion.yaml

   # Check status
   python scripts/serve.py --status

   # View logs
   python scripts/serve.py --logs

   # Stop
   python scripts/serve.py --stop

Or start manually:

.. code-block:: bash

   vllm serve Qwen/Qwen3-VL-8B-Instruct \
     --allowed-local-media-path / \
     --max-model-len 32768 \
     --media-io-kwargs '{"video": {"num_frames": -1}}' \
     --mm-processor-kwargs '{"min_pixels": 262144, "max_pixels": 8388608}' \
     --tensor-parallel-size 1 \
     --gpu-memory-utilization 0.8 \
     --port 8000

The ``scripts/serve.py`` manager handles PID tracking, health checks, and graceful shutdown.
Server logs are written to ``~/.video_ingestion_agent/vllm.log``.

**Module:** :code_link:`<scripts/serve.py>`

Video Input Modes
"""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Mode
     - Config
     - How It Works
   * - **Local file path** (default)
     - ``vllm_local_media: true``
     - Sends a ``file://`` URL to vLLM. The server reads the video directly from disk
       and handles frame extraction server-side. **Fastest** — zero client-side preprocessing.
   * - **Base64 fallback**
     - ``vllm_local_media: false``
     - Client extracts frames at ``vlm_fps``, encodes as base64 JPEG, and sends as
       ``image_url`` entries. Required when the vLLM server runs on a different machine.

.. note::

   Local file path mode requires the vLLM server to be started with
   ``--allowed-local-media-path /`` and the video files to be accessible at the same
   filesystem paths from both the client and the server.

**Health Check:**

``VLLMModel`` pings the ``/health`` endpoint on initialisation. If the server is not
reachable, it raises a ``ConnectionError`` with instructions for starting the server.

**When to use:**

- Production workloads and batch ingestion
- Multi-GPU setups (tensor parallelism)
- Any scenario where throughput matters

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Pros
     - Cons
   * - 2–5x faster than local
     - Requires a running server process
   * - PagedAttention + continuous batching
     - Initial model load takes 1–2 minutes
   * - Tensor parallelism across GPUs
     - Must manage server lifecycle
   * - Server-side video decoding
     - File path mode needs shared filesystem


api — NVIDIA Inference API
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Module:** :code_link:`<src/video_ingestion_agent/models/api_model.py>`

Makes HTTP requests to a cloud endpoint (NVIDIA NIM, OpenAI, or any OpenAI-compatible API).
No local GPU is required.

.. code-block:: yaml

   models:
     vlm_model: "openai/openai/gpt-5.2"
     vlm_backend: "api"
     api_key: null     # reads from NIM_API_KEY env var

**How it works:**

1. ``APIModel`` sends a ``POST`` request to the NVIDIA Inference API endpoint
   (``https://inference-api.nvidia.com/v1/chat/completions``).
2. For video input, the client extracts frames at ``vlm_fps``, encodes each as a base64 JPEG,
   and includes them as ``image_url`` content items alongside a text prompt that provides
   frame-to-timestamp mapping context.
3. The response is parsed from the standard OpenAI ``choices[0].message.content`` format.

**Requirements:**

- ``NIM_API_KEY`` or ``OPENAI_API_KEY`` environment variable
- Internet access

**Supported model providers** (via NVIDIA NIM):

- ``openai/openai/gpt-5.2`` — OpenAI GPT-5.2
- ``google/gemini-1.5-pro`` — Google Gemini 1.5 Pro
- ``anthropic/claude-3`` — Anthropic Claude 3
- Any model available through the NVIDIA Inference API

**When to use:**

- No local GPU available
- Quick experiments with frontier models
- Comparing outputs across different model providers

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Pros
     - Cons
   * - No local GPU needed
     - Requires internet and API key
   * - Access to frontier models
     - API costs per request
   * - Zero setup
     - Higher latency (network round-trip + frame upload)
   * - No model weight downloads
     - Rate limits may apply


Embedding Model
---------------

The **SigLIP-2** embedding model is not configurable per-backend — it always runs locally via
HuggingFace transformers. It produces 768-dimensional visual embeddings used for semantic
frame search during retrieval.

.. code-block:: yaml

   models:
     embedding_model: "google/siglip2-base-patch16-256"
     embedding_batch_size: 16

The model is loaded lazily (on first use) by both the ingestion pipeline
(:code_link:`<src/video_ingestion_agent/ingestion/entity_graph/extractors/visual_extractor.py>`)
and the retrieval agent's frame search tool
(:code_link:`<src/video_ingestion_agent/retrieval/tools/search_frames.py>`).

During **ingestion**, video frames are sampled at ``processing.fps`` (default: 1 fps) and
encoded in batches of ``embedding_batch_size``. During **retrieval**, the user's text query
is encoded with the same model (SigLIP-2 supports text-to-image similarity) and compared
against stored embeddings via cosine similarity.

.. tip::

   If you are running the VLM via vLLM on the same GPU, set
   ``vllm_gpu_memory_utilization: 0.8`` (the default) or lower to reserve VRAM for SigLIP-2.
   You can also reduce ``embedding_batch_size`` to 8 or 4 to further lower peak memory usage.


Choosing a Backend
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Criteria
     - local
     - vllm
     - api
     - Recommendation
   * - **Throughput**
     - Low
     - High
     - Medium
     - vllm for batch; api for occasional use
   * - **Latency (per clip)**
     - ~2–5 s
     - ~0.5–2 s
     - ~3–10 s
     - vllm for interactive; local for simplest setup
   * - **GPU required**
     - Yes (~16 GB)
     - Yes (~16 GB per TP shard)
     - No
     - api if no GPU; vllm/local otherwise
   * - **Setup effort**
     - Low
     - Medium (server)
     - Low (API key only)
     - local for dev; vllm for production
   * - **Multi-GPU**
     - No
     - Yes (``vllm_tp_size``)
     - N/A
     - vllm for multi-GPU
   * - **Cost**
     - Hardware only
     - Hardware only
     - Per-request
     - local/vllm for high volume

.. mermaid::

   flowchart TD
       A["Do you have a local GPU?"] -->|No| B["Use api backend"]
       A -->|Yes| C["Batch or production?"]
       C -->|Yes| D["Use vllm backend"]
       C -->|No| E["Quick experiment?"]
       E -->|Yes| F["Use local backend"]
       E -->|No| D


Switching Backends
------------------

Switching backends requires only a YAML config change — no code modifications. The
``ModelManager`` handles instantiation transparently.

.. tabs::

   .. tab:: vLLM (recommended)

      .. code-block:: yaml

         models:
           vlm_model: "Qwen/Qwen3-VL-8B-Instruct"
           vlm_backend: "vllm"
           vllm_url: "http://localhost:8000/v1"
           vllm_local_media: true
           vllm_tp_size: 1

      .. code-block:: bash

         # Start server, then run pipeline
         python scripts/serve.py -c configs/ingestion.yaml
         python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml

   .. tab:: Local

      .. code-block:: yaml

         models:
           vlm_model: "Qwen/Qwen3-VL-8B-Instruct"
           vlm_backend: "local"
           device: "cuda"

      .. code-block:: bash

         # No server needed
         python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml

   .. tab:: API

      .. code-block:: yaml

         models:
           vlm_model: "openai/openai/gpt-5.2"
           vlm_backend: "api"

      .. code-block:: bash

         export NIM_API_KEY="your-key-here"
         python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml


Using Different Backends for VLM and LLM
-----------------------------------------

The VLM (video processing) and LLM (entity extraction) can use **different backends**. For
example, run segmentation/verification with a fast local vLLM server, but use an API model
for entity extraction:

.. code-block:: yaml

   models:
     # VLM: local vLLM for fast video processing
     vlm_model: "Qwen/Qwen3-VL-8B-Instruct"
     vlm_backend: "vllm"

     # LLM: API model for entity extraction (text-only, so API latency is acceptable)
     llm_model: "openai/openai/gpt-5.2"
     llm_backend: "api"

When ``llm_model`` is ``null`` (the default), entity extraction reuses the VLM model and its
backend. Setting it explicitly enables this split-backend configuration.


Programmatic Usage
------------------

For interactive or notebook use:

.. code-block:: python

   from video_ingestion_agent.models import ModelManager

   manager = ModelManager()

   # Local model
   local = manager.get_model("Qwen/Qwen3-VL-8B-Instruct", backend="local")

   # vLLM model (server must be running)
   vllm = manager.get_model(
       "Qwen/Qwen3-VL-8B-Instruct",
       backend="vllm",
       api_url="http://localhost:8000/v1",
   )

   # API model
   api = manager.get_model("openai/openai/gpt-5.2", backend="api")

   # All share the same interface
   result = local.generate_from_video("demo.mp4", "Describe the actions.")
   result = vllm.generate_from_video("demo.mp4", "Describe the actions.")
   result = api.generate_from_video("demo.mp4", "Describe the actions in this video.")

.. code-block:: python

   # Convenience shortcuts
   from video_ingestion_agent.models import get_local_model, get_api_model

   model = get_local_model("Qwen/Qwen3-VL-8B-Instruct")
   model = get_api_model("openai/openai/gpt-5.2")


Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Issue
     - Solution
   * - ``ConnectionError: vLLM server is not reachable``
     - Start the server: ``python scripts/serve.py -c configs/ingestion.yaml``.
       Check ``~/.video_ingestion_agent/vllm.log`` for errors.
   * - ``NIM_API_KEY environment variable not set``
     - Export the key: ``export NIM_API_KEY="nvapi-..."``
   * - ``CUDA out of memory`` (local backend)
     - Use the ``vllm`` backend instead (more memory-efficient), or reduce
       ``vlm_fps``.
   * - vLLM server exits during model load
     - Check ``~/.video_ingestion_agent/vllm.log``. Common causes: insufficient VRAM, missing
       ``HF_TOKEN`` for gated models, or incompatible vLLM version.
   * - Slow API responses
     - API latency includes frame upload time. Reduce ``vlm_fps`` to send fewer
       frames. For high-throughput use, switch to the ``vllm`` backend.
   * - ``openai`` package not installed
     - ``pip install 'openai>=1.0.0'`` (required only for the vLLM backend).

See Also
--------

- :doc:`/pages/configuration` — Full YAML config reference for model settings
- :doc:`/pages/deployment` — Running models at scale with Docker and OSMO
- :doc:`/pages/troubleshooting` — More troubleshooting tips
