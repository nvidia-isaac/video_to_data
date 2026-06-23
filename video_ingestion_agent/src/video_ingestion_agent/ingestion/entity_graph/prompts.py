# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompts for entity graph extraction.

Based on EGAgent paper (arXiv:2601.18157) - "Agentic Very Long Video Understanding"
"""

# =============================================================================
# Entity Graph Construction Prompt (Based on EGAgent Paper)
# =============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """Knowledge Graph Instructions

1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. Try to capture as much information from the text as possible without sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text.
- Nodes represent entities and concepts.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.

2. Labeling Nodes
- Consistency: Ensure you use available types for node labels. Ensure you use basic or elementary types for node labels.
- For example, when you identify an entity representing a person, always label it as 'person'. Avoid using more specific terms like 'mathematician' or 'scientist'.
- Node IDs: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
- Relationships represent connections between entities or concepts. Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!

3. Coreference Resolution
- Maintain Entity Consistency: When extracting entities, it's vital to ensure consistency. If an entity, such as John Doe, is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., Joe, he), always use the most complete identifier for that entity throughout the knowledge graph. In this example, use John Doe as the entity ID. Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination."""

ENTITY_EXTRACTION_USER_PROMPT = """Based on the following example, extract entities and relations from the provided text. Use the following entity types, don't use other entity that is not defined below:

ENTITY TYPES: {allowed_nodes}

Use the following relation types, don't use other relation that is not defined below:

RELATION TYPES: {allowed_relationships}

TEXT (Visual description from video):
{caption}

Output format (JSON only, no prose):
{{
  "entities": [
    {{"id": "person_human", "type": "person", "properties": {{}}}},
    {{"id": "red_cup", "type": "object", "properties": {{"color": "red", "category": "cup"}}}}
  ],
  "relationships": [
    {{"source": "person_human", "target": "red_cup", "type": "picks-up"}}
  ]
}}"""


# Default entity and relationship types for robot demonstrations
DEFAULT_ENTITY_TYPES = [
    "person",  # Human operator/demonstrator
    "object",  # Manipulable objects (cups, tools, food items, etc.)
    "location",  # Places/regions (table, counter, drawer, shelf, etc.)
]

DEFAULT_RELATIONSHIP_TYPES = [
    "picks-up",  # Person picks up object
    "places-in",  # Person places object in location
    "places-on",  # Person places object on surface
    "grasps",  # Person/hand grasps object
    "releases",  # Person releases object
    "opens",  # Person opens object/location (door, drawer, lid)
    "closes",  # Person closes object/location
    "uses",  # Person uses object (tool usage)
    "interacts-with",  # General interaction
    "pushes",  # Person pushes object
    "pulls",  # Person pulls object
    "rotates",  # Person rotates object
    "located-in",  # Object is located in location
    "located-on",  # Object is located on surface
]
