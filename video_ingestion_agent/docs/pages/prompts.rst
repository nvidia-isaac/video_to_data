Prompt Reference
================

Video Ingestion Agent uses carefully designed prompts to drive the VLM and LLM at every stage of
the pipeline. This page documents every prompt, explains where it is used, and shows how to
override them.

Prompts are organised into three modules:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Module
     - Pipeline Stage
     - Models Targeted
   * - :code_link:`<src/video_ingestion_agent/ingestion/segmentation/prompts.py>`
     - Segmentation & Verification
     - VLM (vision-language)
   * - :code_link:`<src/video_ingestion_agent/ingestion/entity_graph/prompts.py>`
     - Entity Graph Construction
     - LLM (text-only)
   * - :code_link:`<src/video_ingestion_agent/retrieval/nodes/prompts.py>`
     - Retrieval Agent
     - LLM (text-only)


Ingestion: Segmentation Prompts
--------------------------------

**Module:** :code_link:`<src/video_ingestion_agent/ingestion/segmentation/prompts.py>`

These prompts drive the VLM during the initial video segmentation stage. The VLM receives
sampled video frames alongside these text prompts.

System Prompt
^^^^^^^^^^^^^

Sets the VLM's persona as a manipulation video analyst and defines the segmentation rules.
Emphasises naming *actual* objects to prevent hallucinated or generic labels, defines three
clear boundary conditions for when a new segment should start, and keeps instructions
concise (VLMs perform better with short, direct system prompts).

.. dropdown:: Full system prompt text
   :color: light

   .. code-block:: text

      You are an expert at analyzing manipulation videos.
      Your task is to identify distinct action segments based on which object
      the person's hands are actively interacting with.

      CRITICAL INSTRUCTIONS:
      - Each video contains DIFFERENT objects
      - You MUST identify the ACTUAL objects visible in THIS specific video
      - Do NOT use placeholder names or generic examples
      - Be specific: instead of "container" say "glass bowl" or "plastic bin"

      A new segment begins when:
      (1) the person starts manipulating a different object,
      (2) the manipulation action changes (e.g., from picking up to placing), or
      (3) there is a clear pause between actions.

User Prompt
^^^^^^^^^^^

Provides a step-by-step task (observe objects, note interactions, record timestamps) and
specifies the output JSON schema. Requires the VLM to name actual objects in the video and
output timestamps relative to the chunk start.

.. dropdown:: Full user prompt text
   :color: light

   .. code-block:: text

      Watch this video carefully. Your task is to segment it into manipulation clips.

      STEP 1: Observe what objects appear in the video
      STEP 2: For each interaction, note the specific object and action
      STEP 3: Record precise timestamps in seconds

      For each clip, provide:
      - clip_id: Sequential number (1, 2, 3, ...)
      - start_time: In seconds (e.g., 2.5)
      - end_time: In seconds (e.g., 8.0)
      - object: The SPECIFIC, ACTUAL object name you see (be descriptive!)
      - action: The precise action performed on that object
      - description: Detailed explanation of what happens

      Output as JSON array:
      [
        {
          "clip_id": 1,
          "start_time": 2.5,
          "end_time": 8.0,
          "object": "describe the actual object you see",
          "action": "describe the actual action",
          "description": "detailed description"
        }
      ]

      CRITICAL REQUIREMENTS:
      - Name the ACTUAL objects in THIS video (not examples!)
      - Use specific, descriptive names (e.g., "red apple", "metal spoon")
      - Be accurate with timestamps (in seconds)
      - Include ALL manipulation interactions, even brief ones
      - Times should be relative to the start of this video segment

**Output schema:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``clip_id``
     - int
     - Sequential 1-based index
   * - ``start_time``
     - float
     - Start of the action in seconds (relative to chunk start)
   * - ``end_time``
     - float
     - End of the action in seconds (relative to chunk start)
   * - ``object``
     - string
     - Specific object being manipulated
   * - ``action``
     - string
     - Action verb (e.g., "pick up", "place", "pour")
   * - ``description``
     - string
     - Natural language description of the interaction


Ingestion: Verification (Critic) Prompts
------------------------------------------

**Module:** :code_link:`<src/video_ingestion_agent/ingestion/segmentation/prompts.py>`

These prompts drive the VLM during the verification stage. The critic watches the
*extracted clip* (not the full video) and evaluates segmentation quality.

System Prompt
^^^^^^^^^^^^^

Sets the VLM as a critic that checks five quality dimensions: single coherent action,
appropriate boundaries, accurate object identification, correct action description, and
whether the clip should be split or merged.

.. dropdown:: Full system prompt text
   :color: light

   .. code-block:: text

      You are an expert critic analyzing video segmentation quality.

      Your task is to verify if a video clip is correctly segmented and annotated:
      1. Does the clip contain a single, coherent action?
      2. Are the start/end boundaries appropriate (not too early/late)?
      3. Is the object identification accurate?
      4. Is the action description correct?
      5. Should this be split into multiple clips or merged with adjacent ones?

      Be critical but fair.

User Prompt
^^^^^^^^^^^

Includes template variables filled at runtime with the clip's annotation. Asks for a
structured JSON verdict with boundary assessment, annotation assessment, and a
recommendation that directly drives the refinement loop.

.. dropdown:: Full user prompt text
   :color: light

   .. code-block:: text

      Evaluate this segmented video clip:

      CLAIMED ANNOTATION:
      - Object: {object}
      - Action: {action}
      - Description: {description}
      - Duration: {duration:.2f}s

      TASK: Watch the clip and verify the segmentation quality.

      Respond in JSON format:
      {
        "is_correct": true/false,
        "confidence": 0.0-1.0,
        "issues": ["list specific issues found"],
        "boundary_assessment": {
          "start_is_good": true/false,
          "end_is_good": true/false,
          "suggested_adjustment": "description of needed adjustment"
        },
        "annotation_assessment": {
          "object_correct": true/false,
          "action_correct": true/false,
          "description_accurate": true/false,
          "suggested_correction": "corrected annotation if needed"
        },
        "overall_quality": "excellent/good/acceptable/poor",
        "recommendation": "keep_as_is/adjust_boundaries/re_annotate/discard"
      }

**Template variables:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Variable
     - Filled With
   * - ``{object}``
     - Object name from the segmenter's output
   * - ``{action}``
     - Action label from the segmenter's output
   * - ``{description}``
     - Natural language description from the segmenter
   * - ``{duration}``
     - Clip duration in seconds

**Output schema:**

The ``recommendation`` field directly drives the refinement loop:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Recommendation
     - Pipeline Behaviour
   * - ``keep_as_is``
     - Clip passes verification; proceeds to entity extraction
   * - ``re_annotate``
     - Triggers the ``ReannotateStrategy``
   * - ``discard``
     - Clip is dropped from the pipeline output


Ingestion: Entity Extraction Prompts
--------------------------------------

**Module:** :code_link:`<src/video_ingestion_agent/ingestion/entity_graph/prompts.py>`

These prompts are used by the LLM (text-only, no video) to extract structured entities and
relationships from the segmented clip descriptions. The design is inspired by the
`EGAgent paper <https://arxiv.org/abs/2601.18157>`_.

System Prompt
^^^^^^^^^^^^^

A detailed knowledge graph construction prompt that establishes:

- **Node labelling rules** — use basic types (``person``, ``object``, ``location``), not
  specific subtypes.
- **Node ID rules** — use human-readable names found in the text, never integers.
- **Relationship rules** — prefer general, timeless relationship types.
- **Coreference resolution** — always use the most complete identifier for an entity across
  mentions.

.. dropdown:: Full system prompt text
   :color: light

   .. code-block:: text

      Knowledge Graph Instructions

      1. Overview
      You are a top-tier algorithm designed for extracting information in
      structured formats to build a knowledge graph. Try to capture as much
      information from the text as possible without sacrificing accuracy.
      Do not add any information that is not explicitly mentioned in the text.
      ...

      4. Strict Compliance
      Adhere to the rules strictly. Non-compliance will result in termination.

User Prompt
^^^^^^^^^^^

Provides the entity and relationship type constraints and the clip description to analyse.

.. dropdown:: Full user prompt text
   :color: light

   .. code-block:: text

      Based on the following example, extract entities and relations from
      the provided text. Use the following entity types, don't use other
      entity that is not defined below:

      ENTITY TYPES: {allowed_nodes}

      Use the following relation types, don't use other relation that is
      not defined below:

      RELATION TYPES: {allowed_relationships}

      TEXT (Visual description from video):
      {caption}

      Output format (JSON only, no prose):
      {
        "entities": [
          {"id": "person_human", "type": "person", "properties": {}},
          {"id": "red_cup", "type": "object", "properties": {"color": "red", "category": "cup"}}
        ],
        "relationships": [
          {"source": "person_human", "target": "red_cup", "type": "picks-up"}
        ]
      }

**Template variables:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Filled With
   * - ``{allowed_nodes}``
     - Entity type list (default: ``person``, ``object``, ``location``)
   * - ``{allowed_relationships}``
     - Relationship type list (see below)
   * - ``{caption}``
     - The clip's natural language description from segmentation

Default Type Vocabularies
^^^^^^^^^^^^^^^^^^^^^^^^^

These defaults are defined as constants and can be extended programmatically.

**Entity types** (``DEFAULT_ENTITY_TYPES``):

- ``person`` — Human operator/demonstrator
- ``object`` — Manipulable objects (cups, tools, food items, etc.)
- ``location`` — Places/regions (table, counter, drawer, shelf, etc.)

**Relationship types** (``DEFAULT_RELATIONSHIP_TYPES``):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Meaning
   * - ``picks-up``
     - Person picks up object
   * - ``places-in``
     - Person places object in location
   * - ``places-on``
     - Person places object on surface
   * - ``grasps``
     - Person/hand grasps object
   * - ``releases``
     - Person releases object
   * - ``opens``
     - Person opens object/location (door, drawer, lid)
   * - ``closes``
     - Person closes object/location
   * - ``uses``
     - Person uses object (tool usage)
   * - ``interacts-with``
     - General interaction
   * - ``pushes``
     - Person pushes object
   * - ``pulls``
     - Person pulls object
   * - ``rotates``
     - Person rotates object
   * - ``located-in``
     - Object is located in location
   * - ``located-on``
     - Object is located on surface


Retrieval: Agent Prompts
-------------------------

**Module:** :code_link:`<src/video_ingestion_agent/retrieval/nodes/prompts.py>`

The retrieval agent is a LangGraph state graph with multiple LLM-powered nodes. Each node
has its own system/user prompt pair. The agent processes a user's natural language query
through this chain:

.. mermaid::

   flowchart LR
       A["User Query"] --> B["Task\nDecomposition"]
       B --> C["Search\nPlanning"]
       C --> D["Execute\nSearch"]
       D --> E["Result\nAnalysis"]
       E --> F{"Relevant?"}
       F -->|No| G["Search\nAdjustment"]
       G --> C
       F -->|Yes| H["VQA\nSynthesis"]
       H --> I["Final\nAnswer"]

Task Decomposition
^^^^^^^^^^^^^^^^^^

Breaks a high-level query into specific manipulation sub-tasks. Sets the LLM as a robot
manipulation expert and emphasises using the *minimum* number of sub-tasks.

**User template variables:** ``{query}``, ``{max_sub_tasks}``

.. dropdown:: Output schema
   :color: light

   .. code-block:: json

      {
        "task_analysis": "Brief analysis of what the task requires",
        "sub_tasks": [
          {
            "task_id": 1,
            "description": "Find clips of ...",
            "target_action": "action_verb",
            "target_object": "object_name"
          }
        ]
      }

**Example decompositions:**

- ``"pick up the mug"`` → 1 sub-task (single atomic action)
- ``"make coffee"`` → 3 sub-tasks (grab mug, use coffee machine, pour coffee)

Search Planning
^^^^^^^^^^^^^^^

Decides the search strategy for each sub-task. Chooses from three search types in priority
order:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Search Type
     - When to Use
   * - ``segments``
     - **Preferred.** Search action clips in the entity graph. Best when the task has
       a clear action verb (pick, place, open, pour).
   * - ``relationships``
     - **Fallback.** Search entity interactions. Best when segment search yields no results.
   * - ``visual``
     - **Last resort.** Cosine similarity over SigLIP-2 frame embeddings. Best for
       open-ended visual queries.

**User template variables:** ``{task_description}``, ``{target_action}``, ``{target_object}``,
``{history_context}``

.. dropdown:: Output schema
   :color: light

   .. code-block:: json

      {
        "reasoning": "Brief explanation",
        "search_type": "segments",
        "action": "action keyword",
        "object_name": "object keyword"
      }

Search Adjustment
^^^^^^^^^^^^^^^^^

After analysing search results, decides the next action when results are insufficient. Uses
a relaxation level system (0--3) and a strict escalation path:

1. If current search type is not at max relaxation → ``relax_search`` (broaden keywords)
2. If current type is exhausted, another type is not → ``change_search_type``
3. If all types exhausted at level 3 → ``next_task`` (give up on this sub-task)

**Relaxation levels:**

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Level
     - Matching Strategy
   * - 0
     - Exact action *and* exact object match
   * - 1
     - Exact action, broader object match
   * - 2
     - Broader action match, broader object match
   * - 3
     - Most relaxed — any similar action/object

**User template variables:** ``{task_description}``, ``{target_action}``, ``{target_object}``,
``{current_search_type}``, ``{relaxation_level}``, ``{max_relaxation}``, ``{search_history}``,
``{clips_found}``, ``{relevant}``, ``{needs_relaxed}``, ``{analysis_text}``

.. dropdown:: Output schema
   :color: light

   .. code-block:: json

      {
        "action": "relax_search",
        "reasoning": "Brief explanation",
        "new_search_type": "relationships"
      }

Result Analysis
^^^^^^^^^^^^^^^

Evaluates search results for relevance to the current sub-task. Extracts matching clips with
timestamps, video paths, descriptions, and confidence scores.

**User template variables:** ``{task_description}``, ``{target_action}``, ``{target_object}``,
``{search_results}``

.. dropdown:: Output schema
   :color: light

   .. code-block:: json

      {
        "relevant": true,
        "relevant_clips": [
          {
            "start_time": 10.5,
            "end_time": 15.2,
            "video_id": 3,
            "video_path": "/path/to/video.mp4",
            "description": "Person picks up red mug from table",
            "confidence": 0.92
          }
        ],
        "needs_relaxed_search": false,
        "analysis": "Found clear pick-up action matching query"
      }

VQA Synthesis
^^^^^^^^^^^^^

Final synthesis node. Receives all sub-task results and selects the **top 3 clips per
sub-task** (max 15 total) for robot imitation learning.

**Selection criteria:**

1. Clear demonstration of the manipulation action
2. Good visibility of objects and hands
3. Complete action sequences (start to finish)
4. Diversity of scenarios if multiple clips are available

**User template variables:** ``{query}``, ``{task_results}``

.. dropdown:: Output schema
   :color: light

   .. code-block:: json

      {
        "task_summary": "Brief summary (1-2 sentences)",
        "recommended_clips": [
          {
            "start_time": 10.5,
            "end_time": 15.2,
            "video_id": 3,
            "video_path": "/path/to/video.mp4",
            "action": "pick_up",
            "object": "red_mug",
            "description": "Person picks up red mug from table",
            "priority": 1
          }
        ],
        "training_notes": "Brief notes for policy training",
        "missing_demonstrations": ["actions not found in the database"]
      }


Overriding Prompts
-------------------

Segmentation and Verification Prompts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Override via the YAML config file. Empty strings (the default) fall back to the built-in
prompts above.

.. code-block:: yaml

   segmentation:
     system_prompt: |
       You are a robot action segmentation expert.
       Focus on tabletop manipulation actions only.
       ...
     user_prompt: |
       Analyze this video chunk from {chunk_start}s to {chunk_end}s.
       ...

   verification:
     system_prompt: |
       You are a quality control critic for manipulation videos.
       ...
     user_prompt: |
       Evaluate this clip showing "{action}" on "{object}".
       Duration: {duration:.2f}s.
       ...

See :doc:`/pages/configuration` for the full configuration reference.

.. tip::

   Override prompts when adapting the pipeline to a new domain (e.g., industrial assembly
   vs. kitchen tasks). The default prompts are tuned for general manipulation demonstrations.

Entity Extraction Types
^^^^^^^^^^^^^^^^^^^^^^^^

The entity and relationship type vocabularies can be extended programmatically by modifying
``DEFAULT_ENTITY_TYPES`` and ``DEFAULT_RELATIONSHIP_TYPES`` in the prompts module. The
types are passed to the LLM via the ``{allowed_nodes}`` and ``{allowed_relationships}``
template variables.

.. code-block:: python

   from video_ingestion_agent.ingestion.entity_graph.prompts import (
       DEFAULT_ENTITY_TYPES,
       DEFAULT_RELATIONSHIP_TYPES,
   )

   # Add a custom entity type
   custom_entities = DEFAULT_ENTITY_TYPES + ["tool", "container"]

   # Add a custom relationship type
   custom_relationships = DEFAULT_RELATIONSHIP_TYPES + ["fills", "empties", "stacks-on"]

Retrieval Agent Prompts
^^^^^^^^^^^^^^^^^^^^^^^^

The retrieval agent prompts are currently defined as module-level constants and are not
configurable via YAML. To customise them, modify the constants directly in
:code_link:`<src/video_ingestion_agent/retrieval/nodes/prompts.py>`.


Prompt Design Principles
--------------------------

The prompts in Video Ingestion Agent follow several design principles:

1. **Structured JSON output** — Every prompt requests a specific JSON schema, making
   responses machine-parseable. The pipeline includes robust JSON extraction with fallback
   regex parsing.

2. **Grounding in visual evidence** — Segmentation and verification prompts repeatedly
   emphasise naming *actual* objects visible in the video rather than hallucinated or
   placeholder names.

3. **Minimal sub-task inflation** — The task decomposition prompt explicitly instructs the
   LLM to use the *minimum* number of sub-tasks, preventing over-decomposition of simple
   queries.

4. **Progressive relaxation** — The search adjustment prompt implements a structured
   escalation path (exact → relaxed → different search type → give up) rather than
   leaving the strategy open-ended.

5. **Concise descriptions** — Analysis and synthesis prompts enforce a 20-word limit on clip
   descriptions to prevent verbose, repetitive outputs that degrade downstream quality.

See Also
--------

- :doc:`/pages/configuration` — Override segmentation and verification prompts via YAML
- :doc:`/pages/ingestion_pipeline` — How these prompts are used in the pipeline stages
- :doc:`/pages/retrieval_agent` — How the agent prompts drive the search loop
