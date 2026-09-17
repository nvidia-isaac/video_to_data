EPIC-KITCHENS Benchmark
=======================

Video Ingestion Agent includes a full evaluation suite for benchmarking the segmentation pipeline
against `EPIC-KITCHENS-100 <https://epic-kitchens.github.io/2024>`_, a large-scale dataset
of egocentric kitchen activity videos.

Overview
--------

The benchmark evaluates how well the pipeline can:

1. **Detect action segments** — Find where actions occur in long videos (temporal localization)
2. **Set accurate boundaries** — Identify precise start/end times for each action
3. **Annotate correctly** — Produce accurate action/object labels matching the ground truth
   vocabulary

Running the Benchmark
---------------------

Step 1: Prepare Data
^^^^^^^^^^^^^^^^^^^^

Download EPIC-KITCHENS-100 videos and annotations:

.. code-block:: text

   data/benchmark/epic_kitchens/
   ├── videos/          # EPIC-KITCHENS video files
   │   ├── P01_01.mp4
   │   ├── P01_02.mp4
   │   └── ...
   └── annotations/     # EPIC-KITCHENS-100 annotation files
       ├── EPIC_100_train.json
       └── EPIC_100_validation.json

Step 2: Run Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Start vLLM server with tensor parallelism
   python scripts/serve.py -c configs/benchmark_epic_kitchens.yaml

   # Run the benchmark
   python scripts/run_benchmark.py -c configs/benchmark_epic_kitchens.yaml

   # Run on specific videos
   python scripts/run_benchmark.py \
     -c configs/benchmark_epic_kitchens.yaml \
     --video-ids P01_01 P01_02

   # Multi-GPU processing
   python scripts/run_benchmark.py \
     -c configs/benchmark_epic_kitchens.yaml \
     --num-gpus 8

Step 3: Evaluate
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Compute all metrics
   python -m video_ingestion_agent.benchmark.evaluate \
     --predictions runs/benchmark_epic_kitchens/all_predictions.jsonl \
     --annotations-dir data/benchmark/epic_kitchens/annotations

Step 4: Generate Report
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m video_ingestion_agent.benchmark.report \
     --eval-results runs/benchmark_epic_kitchens/eval_results.json \
     --predictions runs/benchmark_epic_kitchens/mapped_predictions.jsonl \
     --annotations-dir data/benchmark/epic_kitchens/annotations

Metrics
-------

Temporal Segmentation
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - **mAP @ tIoU**
     - Mean Average Precision at temporal IoU thresholds (0.1, 0.2, 0.3, 0.4, 0.5) —
       compatible with the official
       `C2-Action-Detection <https://github.com/epic-kitchens/C2-Action-Detection>`_
       evaluation
   * - **Boundary Precision**
     - Fraction of predicted boundaries within tolerance of a ground truth boundary
   * - **Boundary Recall**
     - Fraction of ground truth boundaries matched by a prediction
   * - **Segmentation Ratio**
     - Ratio of total predicted segment time to total ground truth segment time

Annotation Accuracy
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - **Top-k Verb Accuracy**
     - Fraction of predictions whose verb matches ground truth (via sentence-transformer
       embedding matching)
   * - **Top-k Noun Accuracy**
     - Same for noun (object) labels
   * - **Semantic Similarity**
     - Cosine similarity between predicted and GT narration embeddings
   * - **BERTScore**
     - Token-level BERT-based similarity for narration quality

Verb/Noun Mapping
^^^^^^^^^^^^^^^^^^

The benchmark uses an **adapter** (``benchmark/adapter.py``) to map free-text predictions to
the EPIC-KITCHENS verb and noun vocabulary. This is necessary because the VLM produces natural
language descriptions, not EPIC class IDs.

The adapter uses sentence-transformer embeddings to find the closest EPIC class for each
prediction.

Benchmark Modules
-----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Description
   * - ``scripts/run_benchmark.py``
     - CLI entry point for running benchmark
   * - ``benchmark/evaluate.py``
     - Compute all evaluation metrics
   * - ``benchmark/adapter.py``
     - Map free-text predictions to EPIC verb/noun classes
   * - ``benchmark/load_epic_kitchens.py``
     - Parse EPIC-KITCHENS-100 annotations
   * - ``benchmark/report.py``
     - Generate HTML report with timelines and error analysis
   * - ``benchmark/wandb_logger.py``
     - Log results to Weights & Biases

Benchmark Configuration
------------------------

The benchmark config (``configs/benchmark_epic_kitchens.yaml``) differs from the standard
pipeline config in a few key ways:

.. code-block:: yaml

   # Entity graph disabled (not evaluating graph construction)
   enable_entity_graph: false

   # Pipeline report disabled (benchmark report replaces it)
   enable_reporting: false

   # Save VLM responses for analysis
   logging:
     save_responses: true

   # Use all GPUs for tensor parallelism
   models:
     vllm_tp_size: 8

OSMO Cluster Submission
------------------------

For running the benchmark on an OSMO cluster:

.. code-block:: bash

   python scripts/run_osmo.py benchmark \
     --experiment-name epic_kitchens_v1

See :doc:`/pages/deployment` for OSMO workflow details.

See Also
--------

- :doc:`/pages/ingestion_pipeline` — How the segmentation pipeline works
- :doc:`/pages/configuration` — Tuning pipeline parameters for benchmark runs
- :doc:`/pages/deployment` — OSMO cluster submission and multi-GPU scaling
