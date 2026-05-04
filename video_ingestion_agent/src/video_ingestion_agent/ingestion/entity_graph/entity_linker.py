# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entity linking across chunks."""

import logging
from collections import defaultdict

from video_ingestion_agent.utils.types import Entity, Relationship

logger = logging.getLogger(__name__)


class EntityLinker:
    """
    Link and merge duplicate entities across video chunks.

    Uses heuristic matching based on entity IDs and types.
    """

    def __init__(self, max_time_gap: float = 30.0):
        """
        Initialize entity linker.

        Args:
            max_time_gap: Maximum time gap for merging entities (seconds)
        """
        self.max_time_gap = max_time_gap

    def link_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        Link and merge duplicate entities.

        Strategy:
        - Group entities by ID (exact match)
        - Merge entities with same ID if time gap < max_time_gap
        - Update first_seen/last_seen to span entire range

        Args:
            entities: All extracted entities (from all captions)

        Returns:
            Merged list of unique entities
        """
        logger.info(f"Linking {len(entities)} entities")

        # Group entities by ID
        entity_groups: dict[str, list[Entity]] = defaultdict(list)

        for entity in entities:
            entity_groups[entity.entity_id].append(entity)

        # Merge entities in each group
        merged_entities = []

        for _entity_id, group in entity_groups.items():
            # Sort by first_seen
            group.sort(key=lambda e: e.first_seen)

            # Merge into clusters based on time gap
            clusters = []
            current_cluster = [group[0]]

            for entity in group[1:]:
                # Check if entity is close enough to current cluster
                cluster_last_seen = max(e.last_seen for e in current_cluster)

                if entity.first_seen - cluster_last_seen <= self.max_time_gap:
                    # Add to current cluster
                    current_cluster.append(entity)
                else:
                    # Start new cluster
                    clusters.append(current_cluster)
                    current_cluster = [entity]

            # Add last cluster
            clusters.append(current_cluster)

            # Merge each cluster
            for cluster in clusters:
                merged = self._merge_entity_cluster(cluster)
                merged_entities.append(merged)

        logger.info(
            f"Linked entities: {len(entities)} → {len(merged_entities)} "
            f"({len(entities) - len(merged_entities)} duplicates merged)"
        )

        return merged_entities

    def _merge_entity_cluster(self, cluster: list[Entity]) -> Entity:
        """
        Merge a cluster of duplicate entities.

        Args:
            cluster: List of entities with same ID

        Returns:
            Single merged entity
        """
        # Use first entity as base
        merged = cluster[0]

        # Merge properties from all entities
        merged_properties = {}
        for entity in cluster:
            merged_properties.update(entity.properties)

        # Update time range
        first_seen = min(e.first_seen for e in cluster)
        last_seen = max(e.last_seen for e in cluster)

        # Create merged entity
        from dataclasses import replace

        merged = replace(
            merged, first_seen=first_seen, last_seen=last_seen, properties=merged_properties
        )

        return merged

    def link_relationships(self, relationships: list[Relationship]) -> list[Relationship]:
        """
        Link and merge duplicate relationships.

        Strategy: Keep all relationships (even duplicates) since they
        may occur at different times.

        Args:
            relationships: All extracted relationships

        Returns:
            List of relationships (may contain duplicates)
        """
        # For now, keep all relationships
        # In future: could merge relationships with same source/target/type
        # that occur in adjacent time windows

        logger.info(f"Processed {len(relationships)} relationships (kept all)")
        return relationships
