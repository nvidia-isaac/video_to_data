Configuration Guide
===================

Video Ingestion Agent uses YAML configuration files validated by Pydantic models. This guide
explains every configuration option and how to adapt the pipeline for your use case.

Configuration Files
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - ``configs/ingestion.yaml``
     - Main pipeline configuration
   * - ``configs/retrieval.yaml``
     - Retrieval agent settings
   * - ``configs/batch_ingestion.yaml``
     - Batch ingestion on OSMO (TP=8, reporting disabled)
   * - ``configs/benchmark_epic_kitchens.yaml``
     - EPIC-KITCHENS benchmark config

Ingestion Pipeline Configuration
---------------------------------

Models
^^^^^^

.. code-block:: yaml

   models:
     # Vision-language model for segmentation and verification
     vlm_model: "Qwen/Qwen3-VL-8B-Instruct"
     vlm_backend: "vllm"       # "vllm", "local", or "api"
     vlm_fps: 4                # Frames per second sampled for VLM input

     # LLM for entity extraction (null = use vlm_model)
     llm_model: null
     llm_backend: "vllm"

     # Visual embedding model (always runs locally)
     embedding_model: "google/siglip2-base-patch16-256"
     embedding_batch_size: 16

     # Device
     device: "cuda"

     # API key for API backends (or set NIM_API_KEY env var)
     api_key: null

     # vLLM server settings
     vllm_url: "http://localhost:8000/v1"
     vllm_local_media: true     # Use file:// URLs (fastest, same machine)
     vllm_tp_size: 1            # Tensor parallelism GPUs for vLLM server
     vllm_gpu_memory_utilization: 0.8  # Fraction of GPU memory for vLLM (0.0-1.0)

.. note::

   The default VLM is ``Qwen/Qwen3-VL-8B-Instruct``. The pipeline also supports
   ``nvidia/Cosmos-Reason2-8B`` and other Qwen3-VL-compatible models. Adjust this to
   whichever VLM best suits your domain.

.. tip::

   **GPU Memory Sharing.** When the vLLM server and SigLIP-2 embedding model share the same
   GPU, set ``vllm_gpu_memory_utilization`` to cap vLLM's memory allocation. The default of
   ``0.8`` reserves ~20% of VRAM for the embedding model. Lower it further (e.g. ``0.7``) if
   you experience CUDA out-of-memory errors during the frame embedding stage. You can also
   override this at server start time with ``python scripts/serve.py --gpu-mem 0.7``.

Choosing a VLM Backend
""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 12 20 38 30

   * - Backend
     - Config
     - Pros
     - Cons
   * - **vllm**
     - ``vlm_backend: "vllm"``
     - Fastest (2-5x over local). PagedAttention, continuous batching, tensor parallelism.
     - Requires running a separate server process.
   * - **local**
     - ``vlm_backend: "local"``
     - No server needed. Simple setup.
     - Slower. Requires ``[local]`` extras (torch, transformers).
   * - **api**
     - ``vlm_backend: "api"``
     - No local GPU needed.
     - Requires internet. API costs. Latency.

.. tip::

   Use ``vllm`` for production and batch processing. Use ``local`` for quick development
   iteration. Use ``api`` when you don't have local GPU access.

Segmentation
^^^^^^^^^^^^

.. code-block:: yaml

   segmentation:
     chunk_size: 15.0       # Seconds per VLM processing window
     chunk_overlap: 1.5     # Overlap between consecutive chunks (seconds)
     min_clip_s: 1.0        # Minimum clip duration
     max_clip_s: 30.0       # Maximum clip duration

     # Dedup strategy: "heuristic" or "llm" (default)
     dedup_method: "llm"
     dedup_overlap_threshold: -0.1   # Seconds; null to disable

     video_extensions:      # File types to discover in batch mode
       - ".mp4"
       - ".mov"
       - ".mkv"
       - ".MP4"

**Tuning tips:**

- **chunk_size** — Larger chunks give the VLM more context but use more memory. 15s is a
  good balance for most VLMs.
- **chunk_overlap** — Prevents actions at chunk boundaries from being missed. 1.5-3.0s is
  typical.
- **min_clip_s / max_clip_s** — Adjust based on your domain. Robot manipulation actions are
  typically 2-10s. Kitchen activities may be 1-15s.
- **dedup_method** — ``"llm"`` (default) uses a language model to decide whether overlapping
  clips describe the same action before merging. ``"heuristic"`` always merges overlapping
  clips and keeps the longer clip's annotations — faster but less precise.
- **dedup_overlap_threshold** — Minimum temporal overlap in seconds to trigger a merge.
  Negative values (e.g. ``-0.1``) also merge clips separated by a small gap. Set to ``null``
  to disable dedup entirely.

Custom VLM Prompts
""""""""""""""""""

You can override the default segmentation and verification prompts directly in the config:

.. code-block:: yaml

   segmentation:
     # Override the system prompt sent to the VLM during segmentation
     system_prompt: |
       You are a robot action segmentation expert...

     # Override the user prompt (template variables: {chunk_start}, {chunk_end})
     user_prompt: |
       Analyze this video chunk from {chunk_start}s to {chunk_end}s...

   verification:
     # Override the critic system prompt
     system_prompt: |
       You are a quality control critic...

     # Override the critic user prompt
     user_prompt: |
       Evaluate this video clip...

Default prompts are defined in ``src/video_ingestion_agent/ingestion/segmentation/prompts.py``.
Override them when adapting the pipeline to a new domain (e.g., industrial assembly vs.
kitchen tasks).

Verification and Refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   verification:
     max_iterations: 3          # Maximum verify-refine cycles per clip

   enable_verification: true   # Enable/disable the critic
   enable_refinement: true     # Enable/disable refinement loop

.. tip::

   Set ``enable_verification: false`` during initial development to skip the critic and
   refinement loop. This significantly speeds up pipeline runs.

Entity Extraction
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   entity_extraction:
     max_time_gap: 30.0              # Max gap for entity merging (seconds)
     min_entity_confidence: 0.5      # Minimum confidence to keep an entity
     min_relationship_confidence: 0.5  # Minimum confidence for relationships

   enable_entity_graph: true          # Enable/disable entity graph building

Database
^^^^^^^^

.. code-block:: yaml

   database:
     directory: "outputs/"       # Where to store graph.db and vector.db
     embedding_dim: 768          # Must match embedding model output (768 for SigLIP-2)

Processing
^^^^^^^^^^

.. code-block:: yaml

   processing:
     fps: 1.0    # Frame sampling rate for embeddings (lower = faster, fewer embeddings)

Logging
^^^^^^^

.. code-block:: yaml

   logging:
     level: "INFO"              # DEBUG, INFO, WARNING, ERROR
     save_responses: false      # Save raw VLM responses for debugging
     response_dir: "outputs/debug/entity_extraction"

Feature Toggles
^^^^^^^^^^^^^^^

.. code-block:: yaml

   enable_verification: true    # Critic-based clip verification
   enable_refinement: true      # Iterative refinement loop
   enable_entity_graph: true    # Entity extraction + embeddings + DB write
   enable_reporting: true       # HTML report generation

Retrieval Agent Configuration
------------------------------

The retrieval agent uses the same Pydantic-validated configuration pattern as the ingestion
pipeline. Defaults are defined in ``src/video_ingestion_agent/retrieval/config.py``, and
``configs/retrieval.yaml`` is loaded via ``load_retrieval_config()``.

Models
^^^^^^

.. code-block:: yaml

   models:
     llm_model: "Qwen/Qwen3-VL-8B-Instruct"  # LLM for agent reasoning
     llm_backend: "vllm"                       # "local", "vllm", or "api"
     embedding_model: "google/siglip2-base-patch16-256"
     api_key: null                              # Or set NIM_API_KEY env var
     device: "cuda"

Agent
^^^^^

.. code-block:: yaml

   agent:
     max_steps: 10              # Max reasoning iterations before stopping
     temperature: 0.0           # 0.0 = deterministic agent decisions
     max_sub_tasks: 5           # Max sub-tasks to decompose a query into
     max_relaxation_levels: 3   # Max search relaxation levels (0-3)
     max_search_attempts: 9     # Max search attempts per sub-task
     parallel_tasks: true       # Run sub-tasks concurrently (false = sequential)

.. tip::

   **Parallel vs. sequential sub-tasks.** With ``parallel_tasks: true`` (default),
   sub-tasks run concurrently via the LangGraph Send API, providing up to N-fold
   wall-clock speedup for N sub-tasks. The final ``task_results`` dict sent to the
   VQA synthesizer is identical in both modes.

   Set to ``false`` if your queries involve dependent sub-tasks where earlier results
   should guide later searches (e.g. "find X, then find Y that happens right after X").
   In sequential mode, each task's working memory is visible to subsequent tasks.

   See :doc:`/pages/retrieval_agent` for a detailed comparison of both modes.

Database, Output, and Logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   database:
     directory: "outputs/"      # Must contain graph.db and vector.db

   output:
     clips_dir: "outputs/clips"
     clip_padding: 0.5          # Padding around clip boundaries (seconds)

   logging:
     level: "INFO"
     save_traces: true          # Save agent reasoning traces
     traces_dir: "outputs/traces"

Adapting for Your Application
------------------------------

Kitchen / Cooking Videos
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   segmentation:
     chunk_size: 15.0
     min_clip_s: 1.0        # Kitchen actions can be very short
     max_clip_s: 15.0       # Some actions (stirring) can be long

   entity_extraction:
     max_time_gap: 60.0     # Objects persist longer in kitchen scenes

Robot Manipulation (Tabletop)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   segmentation:
     chunk_size: 15.0
     min_clip_s: 2.0        # Manipulation actions are usually 2-10s
     max_clip_s: 10.0

   models:
     vlm_fps: 10            # Higher FPS captures fast hand movements

Industrial Assembly
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   segmentation:
     chunk_size: 20.0       # Longer context for multi-step assembly
     chunk_overlap: 3.0     # More overlap to catch transitions
     min_clip_s: 3.0
     max_clip_s: 20.0

   verification:
     max_iterations: 5      # More refinement for precision-critical tasks

   models:
     vlm_fps: 4             # Lower FPS sufficient for slower assembly

Fast Prototyping (No Verification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   enable_verification: false
   enable_refinement: false
   enable_entity_graph: false
   enable_reporting: false

   # Just get segmentation results quickly

Benchmark Mode
^^^^^^^^^^^^^^^

See ``configs/benchmark_epic_kitchens.yaml`` for a complete example:

.. code-block:: yaml

   enable_entity_graph: false    # Not evaluating graph construction
   enable_reporting: false       # Benchmark report replaces pipeline report

   models:
     vllm_tp_size: 8            # Use all GPUs for tensor parallelism

   logging:
     save_responses: true        # Save all VLM responses for analysis

Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Purpose
   * - ``HF_TOKEN``
     - HuggingFace token for downloading gated models
   * - ``NIM_API_KEY``
     - API key for NVIDIA NIM backend
   * - ``OPENAI_API_KEY``
     - API key for OpenAI backend

See Also
--------

- :doc:`/pages/model_backends` — Detailed backend comparison and setup instructions
- :doc:`/pages/prompts` — Prompt text and how to override it via config
- :doc:`/pages/troubleshooting` — Common issues and solutions
