# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common data types and models."""

from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    """Entity types in the graph."""

    PERSON = "person"
    OBJECT = "object"
    LOCATION = "location"


class RelationType(str, Enum):
    """Relationship types between entities."""

    INTERACTS_WITH = "interacts-with"
    USES = "uses"
    MENTIONS = "mentions"
    PLACES_IN = "places-in"
    PICKS_UP = "picks-up"
    OPENS = "opens"
    CLOSES = "closes"
    GRASPS = "grasps"
    RELEASES = "releases"
    TALKS_TO = "talks-to"


class ActionType(str, Enum):
    """Action types for robot policy learning."""

    PICK = "pick"
    PLACE = "place"
    OPEN = "open"
    CLOSE = "close"
    PUSH = "push"
    PULL = "pull"
    GRASP = "grasp"
    RELEASE = "release"
    ROTATE = "rotate"
    SLIDE = "slide"
    POUR = "pour"


@dataclass
class Entity:
    """Represents an entity in the graph."""

    entity_id: str
    entity_type: EntityType
    first_seen: float  # timestamp in seconds
    last_seen: float
    properties: dict[str, any]


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    source_id: str
    target_id: str
    rel_type: RelationType
    start_t: float
    end_t: float
    confidence: float
    supporting_evidence: str
    spatial_info: dict | None = None
