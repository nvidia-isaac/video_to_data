Database Design
===============

Video Ingestion Agent persists all extracted data in two SQLite databases that live side-by-side in
the configured ``database.directory`` (default: ``outputs/``):

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - File
     - Purpose
     - Key Contents
   * - ``graph.db``
     - Entity scene graph
     - Videos, entities, typed relationships, action segments
   * - ``vector.db``
     - Visual embeddings
     - Per-frame SigLIP-2 embeddings for semantic visual search

Both databases use **WAL (Write-Ahead Logging)** journal mode with a 30-second busy timeout so
multiple parallel shards can write concurrently during batch ingestion without locking
conflicts.

Entity Graph — ``graph.db``
----------------------------

**Module:** :code_link:`<src/video_ingestion_agent/ingestion/entity_graph/database_writer.py>`

Schema
^^^^^^

.. mermaid::

   erDiagram
       video_metadata ||--o{ entities : "contains"
       video_metadata ||--o{ relationships : "contains"
       video_metadata ||--o{ action_segments : "contains"
       entities ||--o{ relationships : "source"
       entities ||--o{ relationships : "target"
       entities ||--o{ action_segments : "primary_object"
       entities ||--o{ action_segments : "secondary_object"

       video_metadata {
           INTEGER id PK
           TEXT video_path UK
           REAL duration
           REAL fps
           INTEGER width
           INTEGER height
           TIMESTAMP created_at
       }

       entities {
           INTEGER id PK
           TEXT entity_id UK
           TEXT entity_type
           REAL first_seen
           REAL last_seen
           TEXT properties
           INTEGER video_id FK
           TIMESTAMP created_at
       }

       relationships {
           INTEGER id PK
           TEXT source_id FK
           TEXT target_id FK
           TEXT rel_type
           REAL start_t
           REAL end_t
           REAL confidence
           TEXT supporting_evidence
           TEXT spatial_info
           INTEGER video_id FK
           TIMESTAMP created_at
       }

       action_segments {
           INTEGER id PK
           TEXT action_type
           REAL start_t
           REAL end_t
           TEXT primary_object_id FK
           TEXT secondary_object_id FK
           TEXT hand
           BOOLEAN success
           REAL quality_score
           TEXT visual_evidence
           INTEGER video_id FK
           TIMESTAMP created_at
       }

video_metadata
^^^^^^^^^^^^^^

One row per ingested video. Stores the canonical file path, duration, frame rate, and
resolution. The auto-incremented ``id`` is used as the foreign key (``video_id``) in all
other tables.

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Column
     - Type
     - Description
   * - ``id``
     - INTEGER PK
     - Auto-incremented video identifier
   * - ``video_path``
     - TEXT UNIQUE
     - Canonical (resolved) file path
   * - ``duration``
     - REAL
     - Video duration in seconds (must be > 0)
   * - ``fps``
     - REAL
     - Frames per second (must be > 0)
   * - ``width``
     - INTEGER
     - Frame width in pixels
   * - ``height``
     - INTEGER
     - Frame height in pixels
   * - ``created_at``
     - TIMESTAMP
     - Row insertion time

entities
^^^^^^^^

Entities are typed objects discovered by the LLM entity extractor. Each entity has a unique
string ``entity_id`` (e.g., ``person_1``, ``bowl_3``) and a type constrained to one of
``person``, ``object``, or ``location``.

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Column
     - Type
     - Description
   * - ``entity_id``
     - TEXT UNIQUE
     - Human-readable identifier (e.g., ``cup_1``)
   * - ``entity_type``
     - TEXT
     - One of ``person``, ``object``, ``location``
   * - ``first_seen``
     - REAL
     - Earliest timestamp where the entity appears (seconds)
   * - ``last_seen``
     - REAL
     - Latest timestamp where the entity appears (seconds)
   * - ``properties``
     - TEXT (JSON)
     - Serialised property dict (colour, size, description, etc.)
   * - ``video_id``
     - INTEGER FK
     - References ``video_metadata.id``

relationships
^^^^^^^^^^^^^

Directed, typed edges between entities. Each relationship captures a specific interaction
(e.g., ``person_1 --picks-up--> cup_3``) together with temporal bounds and a confidence
score.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Column
     - Type
     - Description
   * - ``source_id``
     - TEXT FK
     - Acting entity (references ``entities.entity_id``)
   * - ``target_id``
     - TEXT FK
     - Acted-upon entity (references ``entities.entity_id``)
   * - ``rel_type``
     - TEXT
     - Relationship type (e.g., ``picks-up``, ``places-on``, ``interacts-with``)
   * - ``start_t`` / ``end_t``
     - REAL
     - Temporal window of the interaction (seconds)
   * - ``confidence``
     - REAL
     - Extraction confidence [0.0, 1.0]
   * - ``supporting_evidence``
     - TEXT
     - Original clip description that generated this relationship
   * - ``spatial_info``
     - TEXT (JSON)
     - Optional spatial context (serialised dict)
   * - ``video_id``
     - INTEGER FK
     - References ``video_metadata.id``

**Default relationship types** (defined in
:code_link:`<src/video_ingestion_agent/ingestion/entity_graph/prompts.py>`):

``picks-up``, ``places-in``, ``places-on``, ``grasps``, ``releases``, ``opens``, ``closes``,
``uses``, ``interacts-with``, ``pushes``, ``pulls``, ``rotates``, ``located-in``,
``located-on``

action_segments
^^^^^^^^^^^^^^^

One row per verified action clip. Links the segmented temporal window to its primary and
secondary objects via entity IDs.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Column
     - Type
     - Description
   * - ``action_type``
     - TEXT
     - Action label (e.g., ``pick``, ``place``, ``pour``)
   * - ``start_t`` / ``end_t``
     - REAL
     - Temporal bounds of the action (seconds)
   * - ``primary_object_id``
     - TEXT FK
     - Main object being manipulated (references ``entities.entity_id``)
   * - ``secondary_object_id``
     - TEXT FK
     - Optional target (e.g., the bowl in "place into bowl")
   * - ``hand``
     - TEXT
     - Which hand performed the action (if applicable)
   * - ``success``
     - BOOLEAN
     - Whether the action completed successfully (default: ``TRUE``)
   * - ``quality_score``
     - REAL
     - Clip quality from the verification critic [0.0, 1.0]
   * - ``visual_evidence``
     - TEXT
     - Natural language description of the action
   * - ``video_id``
     - INTEGER FK
     - References ``video_metadata.id``

Indexes
^^^^^^^

All tables carry indexes optimised for the retrieval agent's query patterns:

.. code-block:: sql

   -- Entity lookups
   CREATE INDEX idx_entities_type  ON entities(entity_type);
   CREATE INDEX idx_entities_time  ON entities(first_seen, last_seen);
   CREATE INDEX idx_entities_id    ON entities(entity_id);
   CREATE INDEX idx_entities_video ON entities(video_id);

   -- Relationship graph traversals
   CREATE INDEX idx_relationships_source ON relationships(source_id);
   CREATE INDEX idx_relationships_target ON relationships(target_id);
   CREATE INDEX idx_relationships_type   ON relationships(rel_type);
   CREATE INDEX idx_relationships_time   ON relationships(start_t, end_t);
   CREATE INDEX idx_relationships_video  ON relationships(video_id);

   -- Action queries
   CREATE INDEX idx_actions_type   ON action_segments(action_type);
   CREATE INDEX idx_actions_time   ON action_segments(start_t, end_t);
   CREATE INDEX idx_actions_object ON action_segments(primary_object_id);
   CREATE INDEX idx_actions_video  ON action_segments(video_id);

Schema Migration
^^^^^^^^^^^^^^^^

The ``DatabaseWriter`` supports automatic migration from the original single-video schema
(which lacked ``video_id`` columns) to the current multi-video schema. When opening an
existing database, it checks for the ``video_id`` column and adds it to all tables if missing,
defaulting the value to the existing video's ID.

Similarly, ``VectorDatabase`` migrates older databases that lack the ``segment_id`` column.
The migration runs automatically on first open: the column is added via ``ALTER TABLE`` and
an index is created. Existing rows receive ``NULL`` for ``segment_id``, which the retrieval
agent treats as "unlinked" frames (falling back to plain frame-level output).


Vector Database — ``vector.db``
--------------------------------

**Module:** :code_link:`<src/video_ingestion_agent/utils/vector_database.py>`

Schema
^^^^^^

.. mermaid::

   erDiagram
       videos ||--o{ frame_embeddings : "contains"

       videos {
           TEXT id PK
           TEXT path
           REAL duration
           REAL fps
           INTEGER width
           INTEGER height
           TIMESTAMP created_at
       }

       frame_embeddings {
           INTEGER id PK
           TEXT frame_id UK
           TEXT video_id FK
           REAL timestamp
           BLOB embedding
           TEXT metadata
           TEXT segment_id
           TIMESTAMP created_at
       }

videos
^^^^^^

Mirrors core video metadata. Unlike ``graph.db`` where the primary key is an auto-incremented
integer, here ``id`` is a text string (typically the video stem, e.g., ``P01_01``).

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Column
     - Type
     - Description
   * - ``id``
     - TEXT PK
     - Video identifier (stem of filename)
   * - ``path``
     - TEXT
     - Filesystem path to video
   * - ``duration``
     - REAL
     - Duration in seconds
   * - ``fps``
     - REAL
     - Frames per second
   * - ``width`` / ``height``
     - INTEGER
     - Frame dimensions
   * - ``created_at``
     - TIMESTAMP
     - Row insertion time

frame_embeddings
^^^^^^^^^^^^^^^^

Stores one row per sampled frame. Embeddings are serialised as pickled ``numpy.float32``
arrays (768-dimensional for SigLIP-2 base).

.. list-table::
   :header-rows: 1
   :widths: 20 14 66

   * - Column
     - Type
     - Description
   * - ``frame_id``
     - TEXT UNIQUE
     - Globally unique frame identifier
   * - ``video_id``
     - TEXT FK
     - References ``videos.id``
   * - ``timestamp``
     - REAL
     - Frame time within the video (seconds)
   * - ``embedding``
     - BLOB
     - Pickled ``numpy.float32`` array (768 dims for SigLIP-2)
   * - ``metadata``
     - TEXT (JSON)
     - Optional metadata dict (e.g., ``{"location": "kitchen"}``)
   * - ``segment_id``
     - TEXT
     - Clip/segment identifier linking this frame to the action segment it was extracted
       from (e.g., ``clip-1``). ``NULL`` for frames from legacy ingestion runs.
   * - ``created_at``
     - TIMESTAMP
     - Row insertion time

Indexes:

.. code-block:: sql

   CREATE INDEX idx_frame_video_time ON frame_embeddings(video_id, timestamp);
   CREATE INDEX idx_frame_timestamp  ON frame_embeddings(timestamp);
   CREATE INDEX idx_frame_segment    ON frame_embeddings(segment_id);

Search Operations
^^^^^^^^^^^^^^^^^

The ``VectorDatabase`` class provides three search modes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``search()``
     - **Hybrid search**: cosine similarity over embeddings combined with optional
       ``video_id``, ``segment_id``, and ``time_range`` filters. Used by the retrieval
       agent's ``SearchFramesTool`` for text-to-image semantic search.
   * - ``search_by_time()``
     - **Temporal search**: returns all frames in a ``[start_t, end_t]`` window, optionally
       filtered by video. No vector similarity is computed.
   * - ``search_by_segment()``
     - **Segment lookup**: returns all frames belonging to a given ``segment_id``, optionally
       ranked by cosine similarity against a query embedding.
   * - ``get_frame()``
     - **Point lookup**: retrieve a single frame by its ``frame_id``.

For vector search, the query text is first encoded with the same SigLIP-2 model used during
ingestion to produce a 768-dimensional query embedding. Cosine similarity is computed against
all candidate frame embeddings (post-filter), and the top-k results are returned.

.. note::

   The current implementation performs a **brute-force scan** over all candidate embeddings.
   This is adequate for datasets up to ~100k frames (sub-second latency). For larger corpora,
   consider integrating an approximate nearest neighbour index (e.g., FAISS, ScaNN).


How the Retrieval Agent Uses Both Databases
--------------------------------------------

The LangGraph retrieval agent has two tools, each backed by one of these databases:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Tool
     - Database
     - What It Queries
   * - ``SearchGraphTool``
     - ``graph.db``
     - Entities, relationships, and action segments via structured SQL queries
       (e.g., "find all *pick* actions involving a *cup*")
   * - ``SearchFramesTool``
     - ``vector.db``
     - Frame embeddings via text-to-image cosine similarity
       (e.g., "show me a hand reaching for a red mug")

The agent's task decomposition node breaks a user query into sub-queries, deciding which tool
is appropriate for each. Graph search excels at structured queries with known entity types and
actions, while frame search handles open-ended visual queries.

When a visual search returns frames that carry a ``segment_id``, the executor cross-references
them against the ``action_segments`` table in ``graph.db`` (via
``SearchGraphTool.get_segments_overlapping``). This bridges the two databases so the agent
receives exact clip boundaries and action metadata instead of raw frame timestamps.

See :doc:`/pages/retrieval_agent` for details on the agent's reasoning loop.


SQLite Concurrency Settings
----------------------------

Both databases are configured with:

.. code-block:: sql

   PRAGMA journal_mode = WAL;     -- Write-Ahead Logging for concurrent reads/writes
   PRAGMA busy_timeout = 30000;   -- Wait up to 30 s on lock contention
   PRAGMA synchronous = NORMAL;   -- vector.db only: faster writes, safe with WAL

WAL mode allows multiple reader processes and a single writer without blocking. During batch
ingestion each shard writes to the shared databases; WAL ensures they can do so without
``SQLITE_BUSY`` errors as long as write bursts are shorter than the busy timeout.

.. tip::

   For very large batch runs (10,000+ videos across 32+ shards), monitor for occasional
   ``SQLITE_BUSY`` warnings in the worker logs. If they appear, increase the busy timeout
   or reduce the number of concurrent shards.

See Also
--------

- :doc:`/pages/ingestion_pipeline` — How the pipeline stages populate these databases
- :doc:`/pages/retrieval_agent` — How the agent queries both databases
- :doc:`/pages/configuration` — Database directory and embedding settings
