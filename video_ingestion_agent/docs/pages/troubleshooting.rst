Troubleshooting
===============

Common issues and solutions for Video Ingestion Agent.

Pipeline Issues
---------------

Clips are too short or too long
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adjust the clip duration bounds in your ingestion config:

.. code-block:: yaml

   segmentation:
     min_clip_s: 2.0    # Minimum clip duration (seconds)
     max_clip_s: 10.0   # Maximum clip duration (seconds)

Kitchen activities may need a wider range (1--15 s), while robot manipulation is typically
2--10 s. See :doc:`/pages/configuration` for domain-specific tuning examples.

Entity graph has duplicate entities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The entity linker may not merge entities across clips if they appear too far apart in time.
Increase the merge window:

.. code-block:: yaml

   entity_extraction:
     max_time_gap: 60.0     # Seconds — entities within this gap can be merged

For long videos with recurring objects, a larger gap (60--120 s) reduces duplicates. You can
also lower the confidence threshold to be more aggressive about merging:

.. code-block:: yaml

   entity_extraction:
     min_entity_confidence: 0.3

Pipeline takes too long
^^^^^^^^^^^^^^^^^^^^^^^^

Several settings affect pipeline speed:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Change
     - Impact
   * - Use ``vlm_backend: "vllm"``
     - 2--5x faster than ``local`` backend
   * - Set ``enable_verification: false``
     - Skips critic + refinement loop (fastest iteration)
   * - Reduce ``vlm_fps`` (e.g., 2 instead of 4)
     - Fewer frames per chunk = faster VLM inference
   * - Set ``enable_entity_graph: false``
     - Skips entity extraction, embeddings, linking, and DB write
   * - Increase ``chunk_size`` (e.g., 20 instead of 15)
     - Fewer chunks per video (but uses more VRAM)

For the fastest possible prototyping:

.. code-block:: yaml

   enable_verification: false
   enable_refinement: false
   enable_entity_graph: false
   enable_reporting: false

VLM produces generic or hallucinated object names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default prompts instruct the VLM to name *actual* objects, but some models still
produce generic labels like "object" or "container". Try:

1. **Increase VLM FPS** — more frames give the model better visual context:

   .. code-block:: yaml

      models:
        vlm_fps: 15

2. **Override the segmentation prompt** with domain-specific guidance:

   .. code-block:: yaml

      segmentation:
        system_prompt: |
          You are an expert at analyzing kitchen manipulation videos.
          Objects you may see include: mugs, plates, bowls, utensils, pans,
          cutting boards, food items. Always use the specific object name.

3. **Try a different VLM** — larger or more capable models produce better labels.

Retrieval Issues
-----------------

Retrieval returns no results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Check that the database has data:**

   .. code-block:: bash

      sqlite3 outputs/graph.db "SELECT COUNT(*) FROM action_segments"
      sqlite3 outputs/graph.db "SELECT COUNT(*) FROM entities"

   If counts are zero, the ingestion pipeline may have run with
   ``enable_entity_graph: false``.

2. **Check the agent's relaxation** — the agent automatically broadens searches, but
   if your query uses very specific terms that don't match the VLM's output labels,
   try rephrasing. For example, "grab" instead of "pick up", or "cup" instead of "mug".

3. **Check vector.db exists** — if ``vector.db`` is missing, visual (embedding) search
   is disabled. Re-run ingestion with ``enable_entity_graph: true``.

Retrieval is slow
^^^^^^^^^^^^^^^^^^

The retrieval agent makes multiple LLM calls for task decomposition, search planning,
analysis, and synthesis. To speed it up:

- **Enable parallel sub-tasks** (default) — set ``agent.parallel_tasks: true`` to run
  sub-tasks concurrently. This provides up to N-fold wall-clock speedup for N sub-tasks.
- Use a faster LLM backend (``api`` with a fast cloud model, or ``vllm``)
- Reduce ``max_sub_tasks`` to limit decomposition
- Reduce ``agent.max_steps`` to limit search iterations

Server and Model Issues
-----------------------

vLLM server won't start
^^^^^^^^^^^^^^^^^^^^^^^^

Check the server log:

.. code-block:: bash

   python scripts/serve.py --logs
   # or directly: cat ~/.video_ingestion_agent/vllm.log

Common causes:

- **Insufficient VRAM** — the 8B model needs ~16 GB in bfloat16. Reduce ``vllm_tp_size``
  or use a smaller model.
- **Missing HF_TOKEN** — gated models (e.g., Llama) require authentication:
  ``export HF_TOKEN="hf_..."``
- **Port in use** — another process is using port 8000. Check with
  ``lsof -i :8000`` and either stop it or change ``vllm_url`` in your config.
- **Incompatible vLLM version** — ensure you have a version that supports video input.

``ConnectionError: vLLM server is not reachable``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline can't reach the vLLM server. Either it hasn't started yet or it crashed during
model loading:

.. code-block:: bash

   # Check if running
   python scripts/serve.py --status

   # Start it
   python scripts/serve.py -c configs/ingestion.yaml

   # If it crashed, check logs
   python scripts/serve.py --logs

``CUDA out of memory``
^^^^^^^^^^^^^^^^^^^^^^^

- **Local backend:** Switch to ``vllm`` backend (more memory-efficient due to
  PagedAttention).
- **vLLM backend:** Increase ``vllm_tp_size`` to shard the model across more GPUs.
- **vLLM + SigLIP on same GPU:** Lower ``vllm_gpu_memory_utilization`` (default ``0.8``)
  to reserve more VRAM for the SigLIP-2 embedding model:

  .. code-block:: yaml

     models:
       vllm_gpu_memory_utilization: 0.7   # Reserve ~30% for SigLIP embeddings

  Or pass ``--gpu-mem 0.7`` when starting the server:

  .. code-block:: bash

     python scripts/serve.py -c configs/ingestion.yaml --gpu-mem 0.7

- **Embedding model:** Reduce ``embedding_batch_size`` (e.g., 4 or 8) if the embedding
  model competes for VRAM with the VLM.

``NIM_API_KEY environment variable not set``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Required for the ``api`` backend:

.. code-block:: bash

   export NIM_API_KEY="nvapi-..."


How do I use a different VLM?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Change the model name in your config — no code changes needed:

.. code-block:: yaml

   models:
     vlm_model: "Qwen/Qwen3-VL-8B-Instruct"   # Any HuggingFace VLM
     vlm_backend: "vllm"                         # Or "local" or "api"

Then restart the vLLM server (if using vllm backend):

.. code-block:: bash

   python scripts/serve.py --stop
   python scripts/serve.py -c configs/ingestion.yaml

See :doc:`/pages/model_backends` for the full list of supported backends and models.

See Also
--------

- :doc:`/pages/model_backends` — Backend-specific troubleshooting table
- :doc:`/pages/configuration` — Full configuration reference
- :doc:`/pages/getting_started` — Installation and first-run guide
- `GitHub Issues <https://github.com/nvidia-isaac/video_to_data/issues>`_ — Report bugs
