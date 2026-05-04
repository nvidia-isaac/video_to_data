Architecture Overview
=====================

Video Ingestion Agent is built around two core workflows orchestrated by
`LangGraph <https://github.com/langchain-ai/langgraph>`_ state graphs:

.. image:: /images/video_ingestion_agent_overview.jpg
   :alt: Video Ingestion Agent Architecture Overview
   :align: center

1. The **Ingestion Pipeline** processes videos into a structured entity graph database.
2. The **Retrieval Agent** queries that database to find and extract action clips.

Both are LangGraph state graphs — typed state flows through a sequence of nodes, with
conditional edges for branching (e.g., should we refine? is the search done?). Nodes are
independent functions, making them easy to test and reconfigure.


Ingestion Pipeline
------------------

The pipeline turns a raw video into two SQLite databases (``graph.db`` and ``vector.db``)
through eight stages. Three design choices shape how it works:

**Chunk-based VLM processing.**
Videos are split into overlapping windows (default 15 s, 1.5 s overlap) because VLMs cannot
process entire long videos at once. Each chunk is analysed independently and results are
merged with a configurable dedup strategy (``heuristic`` or ``llm``). The LLM strategy
(default) asks a language model whether overlapping clips describe the same object and action
before merging; the heuristic strategy always merges and keeps the longer clip's annotations.

**Critic-refinement loop.**
A separate VLM "critic" watches each extracted clip and evaluates whether the segmentation
is correct. When it finds issues, the refiner either adjusts boundaries (by re-extracting a
wider window from the source video) or re-annotates the clip. This loop runs up to
``max_iterations`` times (default 3).

**Entity graph construction.**
An LLM extracts typed entities (person, object, location) and relationships (picks-up,
places-on, etc.) from clip descriptions. A linker deduplicates entities across clips, and
everything is written to ``graph.db``. Separately, SigLIP-2 frame embeddings are stored in
``vector.db`` for visual similarity search. Each frame embedding is tagged with a
``segment_id`` linking it back to its source action clip, enabling the retrieval agent to
resolve visual search hits to exact clip boundaries via cross-database lookup.

See :doc:`/pages/ingestion_pipeline` for a stage-by-stage walkthrough.


Retrieval Agent
---------------

The retrieval agent (inspired by `EGAgent <https://arxiv.org/abs/2601.18157>`_) finds and
extracts clips from the database using natural language queries. It is built around three
ideas:

**Task decomposition.**
Instead of running a single database query, the agent breaks a natural-language request into
sub-tasks. *"Find clips of making coffee"* becomes: grab mug, use coffee machine, pour
coffee. Each sub-task is searched independently, so compound queries don't return nothing
when a single monolithic search can't match all criteria at once.

**Dual search strategy.**
The agent has two search tools and picks the right one per sub-task:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tool
     - When to use
   * - **Entity graph search** (``graph.db``)
     - Structured SQL against entities, relationships, and action segments. Fast and precise
       when the query has clear action verbs and objects (*"pick up mug"*).
   * - **Visual embedding search** (``vector.db``)
     - Cosine similarity between a text embedding and SigLIP-2 frame embeddings. Handles
       open-ended visual queries (*"a hand reaching toward something red"*). When frames carry
       a ``segment_id``, the executor resolves them against ``graph.db`` action segments to
       return precise clip boundaries.

**Progressive relaxation.**
When a search returns insufficient results the agent broadens constraints through four
levels — from exact match (level 0) to any similar action/object (level 3). If all levels
are exhausted for one search type it switches to the other before moving on. This avoids
both giving up too early and searching indefinitely.

See :doc:`/pages/retrieval_agent` for the full node-by-node walkthrough.


Storage and Model Backends
--------------------------

**SQLite databases.**
The entity graph and vector embeddings live in two SQLite files (``graph.db`` and
``vector.db``). SQLite was chosen for simplicity and portability — no external database
server required. WAL mode enables concurrent writes from multiple shards during batch
ingestion. See :doc:`/pages/database_design` for the full schema.

**Model backend abstraction.**
The ``ModelManager`` provides a single interface across three backends so pipeline code is
backend-agnostic:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Backend
     - Use case
   * - ``vllm``
     - Production — 2-5x faster inference via vLLM server with PagedAttention and tensor
       parallelism
   * - ``local``
     - Development — in-process HuggingFace inference, no server needed
   * - ``api``
     - Cloud — remote API (NVIDIA NIM, OpenAI), no local GPU needed

See :doc:`/pages/model_backends` for setup and comparison.


Project Structure
-----------------

.. code-block:: text

   src/video_ingestion_agent/
   ├── ingestion/                 # Ingestion pipeline
   │   ├── ingestion_graph.py     # LangGraph pipeline definition
   │   ├── config.py              # Pydantic configuration models
   │   ├── state.py               # Pipeline state and data types
   │   ├── segmentation_nodes.py  # Segmentation LangGraph nodes
   │   ├── entity_graph_nodes.py  # Entity graph LangGraph nodes
   │   ├── segmentation/          # Segmenter, critic, refiner
   │   └── entity_graph/          # Extractors, linker, DB writer
   ├── retrieval/                 # Retrieval agent
   │   ├── retrieval_graph.py     # Agent graph definition
   │   ├── state.py               # Agent state
   │   ├── nodes/                 # Agent decision-making nodes
   │   └── tools/                 # Search and extraction tools
   ├── models/                    # VLM/LLM model backends
   ├── benchmark/                 # EPIC-KITCHENS evaluation
   ├── webapp/                    # Gradio web interface
   └── utils/                     # Shared utilities


Data Types
----------

ClipContext
^^^^^^^^^^^

The central data structure for a segmented video clip:

.. code-block:: python

   @dataclass
   class ClipContext:
       clip_id: str          # Unique identifier
       video_path: str       # Source video file
       start_t: float        # Start timestamp (seconds)
       end_t: float          # End timestamp (seconds)
       action: str           # Action label (e.g., "pick up")
       object: str           # Object label (e.g., "mug")
       description: str      # Detailed natural language description
       metadata: dict        # fps, confidence, source info, etc.

Entity and Relationship
^^^^^^^^^^^^^^^^^^^^^^^^

Extracted from clip descriptions to form the scene graph:

.. code-block:: python

   @dataclass
   class Entity:
       entity_id: str
       entity_type: EntityType   # person, object, location
       properties: dict          # name, color, position, etc.
       first_seen: float         # Timestamp
       last_seen: float

   @dataclass
   class Relationship:
       source_id: str            # Entity ID
       target_id: str            # Entity ID
       rel_type: RelationType    # interacts-with, uses, picks-up, etc.
       start_t: float
       end_t: float
       confidence: float

See Also
--------

- :doc:`/pages/ingestion_pipeline` — Detailed walkthrough of each pipeline stage
- :doc:`/pages/retrieval_agent` — How the retrieval agent reasons and searches
- :doc:`/pages/database_design` — Entity graph and vector database schemas
- :doc:`/pages/model_backends` — Choosing and configuring inference backends
