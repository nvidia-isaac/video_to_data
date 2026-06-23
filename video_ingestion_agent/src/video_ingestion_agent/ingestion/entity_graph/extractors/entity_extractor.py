# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity extractor using LLM."""

import json
import logging
from pathlib import Path

from video_ingestion_agent.ingestion.entity_graph.prompts import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_RELATIONSHIP_TYPES,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_USER_PROMPT,
)
from video_ingestion_agent.models.model_manager import BaseModel, get_model_manager
from video_ingestion_agent.utils.types import Entity, EntityType, Relationship, RelationType

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extract entities and relationships from visual captions using LLM.

    Uses structured prompting to extract:
    - Entities (people, objects, locations)
    - Relationships (interactions between entities)

    Supports both local and API-based LLM backends via ModelManager.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        llm_model: str = "nvidia/Cosmos-Reason2-8B",
        device: str = "cuda",
        backend: str = "local",
        api_key: str | None = None,
        api_url: str = "http://localhost:8000/v1",
        save_responses: bool = True,
        response_dir: str = "outputs/debug/entity_extraction",
        allowed_entity_types: list[str] | None = None,
        allowed_relationship_types: list[str] | None = None,
    ):
        """
        Initialize entity extractor.

        Args:
            system_prompt: System prompt for entity extraction (default: EGAgent paper prompt)
            llm_model: LLM model name
            device: Device to run model on
            backend: "local", "api", or "vllm" for LLM inference
            api_key: API key if using API backend
            api_url: vLLM server URL (only used when backend="vllm")
            save_responses: Whether to save LLM responses for debugging
            response_dir: Directory to save responses
            allowed_entity_types: List of allowed entity types (default: person, object, location)
            allowed_relationship_types: List of allowed relationship types
        """
        self.device = device
        self.llm_model_name = llm_model
        self.backend = backend
        self.api_key = api_key
        self.api_url = api_url
        self.save_responses = save_responses
        self.response_dir = Path(response_dir)

        # Use EGAgent paper prompts by default
        self.system_prompt = system_prompt or ENTITY_EXTRACTION_SYSTEM_PROMPT
        self.system_prompt = self.system_prompt.strip()

        # Entity and relationship types
        self.allowed_entity_types = allowed_entity_types or DEFAULT_ENTITY_TYPES
        self.allowed_relationship_types = allowed_relationship_types or DEFAULT_RELATIONSHIP_TYPES

        if self.save_responses:
            self.response_dir.mkdir(parents=True, exist_ok=True)

        # LLM model - lazy loaded via ModelManager
        self._model: BaseModel | None = None

    def _get_model(self) -> BaseModel:
        """Get LLM model from ModelManager (lazy loading with caching)."""
        if self._model is None:
            manager = get_model_manager()
            self._model = manager.get_model(
                model_name=self.llm_model_name,
                backend=self.backend,
                device=self.device,
                api_key=self.api_key,
                api_url=self.api_url,
            )
            logger.info(f"✓ LLM loaded successfully ({self.backend}): {self.llm_model_name}")
        return self._model

    def extract_from_caption(
        self, caption: str, caption_idx: int = 0
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract entities and relationships from a visual caption.

        Args:
            caption: Visual description text
            caption_idx: Caption index (for debugging)

        Returns:
            Tuple of (entities, relationships)
        """
        logger.info(f"Extracting entities from caption {caption_idx}")

        # Construct user prompt using EGAgent format
        user_prompt = ENTITY_EXTRACTION_USER_PROMPT.format(
            allowed_nodes=", ".join(self.allowed_entity_types),
            allowed_relationships=", ".join(self.allowed_relationship_types),
            caption=caption,
        )

        # Format as conversation
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Get model and generate
            model = self._get_model()
            response = model.generate_text(
                conversation=conversation,
                max_new_tokens=2048,
                temperature=0.0,
            )

            # Save response if debugging
            if self.save_responses:
                self._save_response(caption_idx, caption, response)

            # Parse JSON
            entities, relationships = self._parse_response(response)

            logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")

            return entities, relationships

        except Exception as e:
            import traceback

            logger.error(f"Entity extraction failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return [], []

    def _parse_response(self, response: str) -> tuple[list[Entity], list[Relationship]]:
        """
        Parse LLM response to extract entities and relationships.

        Args:
            response: LLM output text

        Returns:
            Tuple of (entities, relationships)
        """
        # Try to extract JSON from response
        try:
            # Strip <think> reasoning / extract <answer> block so that
            # chain-of-thought content does not confuse JSON extraction.
            from ....utils.parsing import _strip_think_tags

            cleaned = _strip_think_tags(response)

            # Find JSON block (between { and })
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON found in response")
                return [], []

            json_str = cleaned[start_idx:end_idx]
            data = json.loads(json_str)

            # Parse entities
            entities = []
            for entity_data in data.get("entities", []):
                try:
                    # Handle case where LLM returns strings instead of dicts
                    if isinstance(entity_data, str):
                        logger.warning(f"Entity data is string, not dict: {entity_data}")
                        continue

                    entity = Entity(
                        entity_id=entity_data["id"],
                        entity_type=EntityType(entity_data["type"]),
                        first_seen=0.0,  # Will be set by caller with segment timestamps
                        last_seen=0.0,
                        properties=entity_data.get("properties", {}),
                    )
                    entities.append(entity)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Invalid entity data: {e} - data: {entity_data}")
                    continue

            # Parse relationships
            relationships = []
            for rel_data in data.get("relationships", []):
                try:
                    # Handle case where LLM returns strings instead of dicts
                    if isinstance(rel_data, str):
                        logger.warning(f"Relationship data is string, not dict: {rel_data}")
                        continue

                    # Map relationship type to enum
                    rel_type_str = rel_data["type"]

                    # Try to match to enum
                    try:
                        rel_type = RelationType(rel_type_str)
                    except ValueError:
                        # Default to interacts-with
                        logger.warning(
                            f"Unknown relationship type: {rel_type_str}, using 'interacts-with'"
                        )
                        rel_type = RelationType.INTERACTS_WITH

                    relationship = Relationship(
                        source_id=rel_data["source"],
                        target_id=rel_data["target"],
                        rel_type=rel_type,
                        start_t=0.0,  # Will be set by caller with segment timestamps
                        end_t=0.0,
                        confidence=1.0,
                        supporting_evidence="",  # Will be set by caller
                    )
                    relationships.append(relationship)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid relationship data: {e}")
                    continue

            return entities, relationships

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response: {response[:500]}")
            return [], []

    def _save_response(self, caption_idx: int, caption: str, response: str):
        """Save LLM response for debugging."""
        filename = self.response_dir / f"caption_{caption_idx:04d}.txt"

        with open(filename, "w") as f:
            f.write("=== CAPTION ===\n")
            f.write(caption)
            f.write("\n\n=== LLM RESPONSE ===\n")
            f.write(response)
            f.write("\n")
