# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Tool for searching the entity graph database."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from video_ingestion_agent.retrieval.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class EntityResult:
    """Entity search result."""

    entity_id: str
    entity_type: str
    first_seen: float
    last_seen: float
    properties: dict[str, Any]
    video_id: int | None = None
    video_path: str | None = None

    def __str__(self) -> str:
        props = json.dumps(self.properties) if self.properties else "{}"
        video_str = f" [{self.video_path}]" if self.video_path else ""
        return (
            f"Entity: {self.entity_id} ({self.entity_type}) "
            f"[{self.first_seen:.1f}s - {self.last_seen:.1f}s] {props}{video_str}"
        )


@dataclass
class RelationshipResult:
    """Relationship search result."""

    source_id: str
    target_id: str
    rel_type: str
    start_t: float
    end_t: float
    evidence: str | None
    video_id: int | None = None
    video_path: str | None = None

    def __str__(self) -> str:
        evidence_str = f" - {self.evidence[:50]}..." if self.evidence else ""
        video_str = f" [{self.video_path}]" if self.video_path else ""
        return (
            f"{self.source_id} --[{self.rel_type}]--> {self.target_id} "
            f"[{self.start_t:.1f}s - {self.end_t:.1f}s]{evidence_str}{video_str}"
        )


@dataclass
class SegmentResult:
    """Action segment search result."""

    segment_id: int
    action: str
    object_name: str
    start_t: float
    end_t: float
    description: str | None
    video_id: int | None = None
    video_path: str | None = None

    def __str__(self) -> str:
        desc_str = f" - {self.description[:50]}..." if self.description else ""
        video_str = f" [{self.video_path}]" if self.video_path else ""
        return (
            f"Segment {self.segment_id}: {self.action} {self.object_name} "
            f"[{self.start_t:.1f}s - {self.end_t:.1f}s]{desc_str}{video_str}"
        )


class SearchGraphTool(BaseTool):
    """
    Search the entity graph database with strict-to-relaxed hierarchy.

    Supports querying:
    - Entities by type, name pattern, time range
    - Relationships by type, source/target, time range
    - Action segments by action, object, time range

    Relaxation levels (for segments):
    - 0: Exact match on action AND object
    - 1: Exact action, partial object match (LIKE '%object%')
    - 2: Partial action AND partial object match
    - 3: Any match on action OR object (most relaxed)

    Multi-video support:
    - Results include video_id and video_path fields
    - Can filter by video_id parameter
    """

    def __init__(self, db_path: str):
        """
        Initialize with entity graph database.

        Args:
            db_path: Path to entity graph SQLite database
        """
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @property
    def name(self) -> str:
        return "search_graph"

    @property
    def description(self) -> str:
        return (
            "Search the entity graph database for entities, relationships, "
            "or action segments. Use this to find what objects appear in the video, "
            "what actions were performed, and when they occurred. "
            "Supports strict-to-relaxed search with relaxation_level parameter. "
            "For multi-video databases, results include video_path."
        )

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
        return {
            "query_type": {
                "type": "string",
                "description": "Type of query: 'entities', 'relationships', or 'segments'",
                "required": True,
                "enum": ["entities", "relationships", "segments"],
            },
            "entity_type": {
                "type": "string",
                "description": "Filter entities by type: 'person', 'object', 'location'",
                "required": False,
            },
            "entity_name": {
                "type": "string",
                "description": "Filter by entity name pattern (partial match)",
                "required": False,
            },
            "rel_type": {
                "type": "string",
                "description": "Filter relationships by type (e.g., 'picks-up', 'grasps')",
                "required": False,
            },
            "action": {
                "type": "string",
                "description": "Filter segments by action (e.g., 'pick_up', 'grasp', 'place')",
                "required": False,
            },
            "object_name": {
                "type": "string",
                "description": "Filter by object name (partial match)",
                "required": False,
            },
            "start_time": {
                "type": "number",
                "description": "Filter by start time (seconds) - items after this time",
                "required": False,
            },
            "end_time": {
                "type": "number",
                "description": "Filter by end time (seconds) - items before this time",
                "required": False,
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of results (default: 20)",
                "required": False,
            },
            "relaxation_level": {
                "type": "number",
                "description": "Search relaxation: 0=strict, 1=partial object, 2=partial both, 3=any match",
                "required": False,
            },
            "video_id": {
                "type": "number",
                "description": "Filter by specific video ID (for multi-video databases)",
                "required": False,
            },
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute graph search with strict-to-relaxed hierarchy."""
        query_type = kwargs.get("query_type")

        if not query_type:
            return ToolResult(success=False, data=None, error="query_type is required")

        try:
            if query_type == "entities":
                results = self._search_entities(**kwargs)
            elif query_type == "relationships":
                results = self._search_relationships(**kwargs)
            elif query_type == "segments":
                results = self._search_segments_relaxed(**kwargs)
            else:
                return ToolResult(
                    success=False, data=None, error=f"Unknown query_type: {query_type}"
                )

            return ToolResult(success=True, data=results)

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    def _search_entities(self, **kwargs) -> list[EntityResult]:
        """Search entities table with video_path."""
        conn = self._get_conn()

        sql = """
            SELECT e.entity_id, e.entity_type, e.first_seen, e.last_seen,
                   e.properties, e.video_id, v.video_path
            FROM entities e
            LEFT JOIN video_metadata v ON e.video_id = v.id
            WHERE 1=1
        """
        params = []

        if kwargs.get("entity_type"):
            sql += " AND e.entity_type = ?"
            params.append(kwargs["entity_type"])

        if kwargs.get("entity_name"):
            sql += " AND e.entity_id LIKE ?"
            params.append(f"%{kwargs['entity_name']}%")

        if kwargs.get("start_time") is not None:
            sql += " AND e.last_seen >= ?"
            params.append(kwargs["start_time"])

        if kwargs.get("end_time") is not None:
            sql += " AND e.first_seen <= ?"
            params.append(kwargs["end_time"])

        if kwargs.get("video_id") is not None:
            sql += " AND e.video_id = ?"
            params.append(kwargs["video_id"])

        limit = kwargs.get("limit", 20)
        sql += f" ORDER BY e.first_seen LIMIT {int(limit)}"

        cursor = conn.execute(sql, params)

        results = []
        for row in cursor:
            props = json.loads(row["properties"]) if row["properties"] else {}
            results.append(
                EntityResult(
                    entity_id=row["entity_id"],
                    entity_type=row["entity_type"],
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                    properties=props,
                    video_id=row["video_id"],
                    video_path=row["video_path"],
                )
            )

        return results

    def _search_relationships(self, **kwargs) -> list[RelationshipResult]:
        """Search relationships table with video_path."""
        conn = self._get_conn()

        sql = """
            SELECT r.source_id, r.target_id, r.rel_type, r.start_t, r.end_t,
                   r.supporting_evidence, r.video_id, v.video_path
            FROM relationships r
            LEFT JOIN video_metadata v ON r.video_id = v.id
            WHERE 1=1
        """
        params = []

        if kwargs.get("rel_type"):
            sql += " AND r.rel_type = ?"
            params.append(kwargs["rel_type"])

        if kwargs.get("entity_name"):
            sql += " AND (r.source_id LIKE ? OR r.target_id LIKE ?)"
            pattern = f"%{kwargs['entity_name']}%"
            params.extend([pattern, pattern])

        if kwargs.get("object_name"):
            sql += " AND r.target_id LIKE ?"
            params.append(f"%{kwargs['object_name']}%")

        if kwargs.get("start_time") is not None:
            sql += " AND r.end_t >= ?"
            params.append(kwargs["start_time"])

        if kwargs.get("end_time") is not None:
            sql += " AND r.start_t <= ?"
            params.append(kwargs["end_time"])

        if kwargs.get("video_id") is not None:
            sql += " AND r.video_id = ?"
            params.append(kwargs["video_id"])

        limit = kwargs.get("limit", 20)
        sql += f" ORDER BY r.start_t LIMIT {int(limit)}"

        cursor = conn.execute(sql, params)

        results = []
        for row in cursor:
            results.append(
                RelationshipResult(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    rel_type=row["rel_type"],
                    start_t=row["start_t"],
                    end_t=row["end_t"],
                    evidence=row["supporting_evidence"],
                    video_id=row["video_id"],
                    video_path=row["video_path"],
                )
            )

        return results

    def _search_segments_relaxed(self, **kwargs) -> list[SegmentResult]:
        """
        Search action_segments table with strict-to-relaxed hierarchy.

        Relaxation levels:
        - 0: Exact match on action AND object
        - 1: Exact action, partial object match
        - 2: Partial action AND partial object match
        - 3: Any match on action OR object
        """
        conn = self._get_conn()
        relaxation_level = kwargs.get("relaxation_level", 0)
        action = kwargs.get("action", "")
        object_name = kwargs.get("object_name", "")
        limit = kwargs.get("limit", 20)

        logger.info(
            f"Segment search: action='{action}', object='{object_name}', relaxation={relaxation_level}"
        )

        # Build SQL with video_path join
        sql_base = """
            SELECT a.id, a.action_type, a.primary_object_id, a.start_t, a.end_t,
                   a.visual_evidence, a.video_id, v.video_path
            FROM action_segments a
            LEFT JOIN video_metadata v ON a.video_id = v.id
            WHERE 1=1
        """
        params = []

        # Time filters (always applied)
        if kwargs.get("start_time") is not None:
            sql_base += " AND a.end_t >= ?"
            params.append(kwargs["start_time"])

        if kwargs.get("end_time") is not None:
            sql_base += " AND a.start_t <= ?"
            params.append(kwargs["end_time"])

        # Video filter
        if kwargs.get("video_id") is not None:
            sql_base += " AND a.video_id = ?"
            params.append(kwargs["video_id"])

        # Apply relaxation strategy
        if relaxation_level == 0:
            # Level 0: Strict - exact match on both action and object
            if action:
                sql_base += " AND a.action_type = ?"
                params.append(action)
            if object_name:
                sql_base += " AND a.primary_object_id = ?"
                params.append(object_name)

        elif relaxation_level == 1:
            # Level 1: Exact action, partial object match
            if action:
                sql_base += " AND a.action_type = ?"
                params.append(action)
            if object_name:
                sql_base += " AND a.primary_object_id LIKE ?"
                params.append(f"%{object_name}%")

        elif relaxation_level == 2:
            # Level 2: Partial match on both action and object
            if action:
                sql_base += " AND a.action_type LIKE ?"
                params.append(f"%{action}%")
            if object_name:
                sql_base += " AND a.primary_object_id LIKE ?"
                params.append(f"%{object_name}%")

        else:  # relaxation_level >= 3
            # Level 3: Any match - action OR object (most relaxed)
            conditions = []
            if action:
                conditions.append("a.action_type LIKE ?")
                params.append(f"%{action}%")
            if object_name:
                conditions.append("a.primary_object_id LIKE ?")
                params.append(f"%{object_name}%")
            # Also search in visual_evidence (description)
            if action or object_name:
                search_term = action or object_name
                conditions.append("a.visual_evidence LIKE ?")
                params.append(f"%{search_term}%")

            if conditions:
                sql_base += f" AND ({' OR '.join(conditions)})"

        sql_base += f" ORDER BY a.start_t LIMIT {int(limit)}"

        logger.debug(f"SQL: {sql_base}")
        logger.debug(f"Params: {params}")

        cursor = conn.execute(sql_base, params)

        results = []
        for row in cursor:
            results.append(
                SegmentResult(
                    segment_id=row["id"],
                    action=row["action_type"],
                    object_name=row["primary_object_id"] or "",
                    start_t=row["start_t"],
                    end_t=row["end_t"],
                    description=row["visual_evidence"],
                    video_id=row["video_id"],
                    video_path=row["video_path"],
                )
            )

        logger.info(f"Found {len(results)} segments at relaxation level {relaxation_level}")
        return results

    def _search_segments(self, **kwargs) -> list[SegmentResult]:
        """Legacy segment search (non-relaxed)."""
        return self._search_segments_relaxed(**kwargs)

    def get_segments_overlapping(
        self,
        start_t: float,
        end_t: float,
        video_id: int | None = None,
        video_path: str | None = None,
    ) -> list[SegmentResult]:
        """Find action segments whose time range overlaps ``[start_t, end_t]``.

        Args:
            start_t: Window start (seconds).
            end_t: Window end (seconds).
            video_id: Optional integer video_id filter.
            video_path: Optional filesystem-path filter (preferred when callers
                use vector DB IDs that don't match graph DB IDs).

        Returns:
            Matching SegmentResult list ordered by start_t.
        """
        conn = self._get_conn()

        sql = """
            SELECT a.id, a.action_type, a.primary_object_id, a.start_t, a.end_t,
                   a.visual_evidence, a.video_id, v.video_path
            FROM action_segments a
            LEFT JOIN video_metadata v ON a.video_id = v.id
            WHERE a.start_t <= ? AND a.end_t >= ?
        """
        params: list = [end_t, start_t]

        if video_id is not None:
            sql += " AND a.video_id = ?"
            params.append(video_id)
        if video_path:
            sql += " AND v.video_path = ?"
            params.append(video_path)

        sql += " ORDER BY a.start_t"

        cursor = conn.execute(sql, params)

        results = []
        for row in cursor:
            results.append(
                SegmentResult(
                    segment_id=row["id"],
                    action=row["action_type"],
                    object_name=row["primary_object_id"] or "",
                    start_t=row["start_t"],
                    end_t=row["end_t"],
                    description=row["visual_evidence"],
                    video_id=row["video_id"],
                    video_path=row["video_path"],
                )
            )

        return results

    def get_all_actions(self) -> list[str]:
        """Get all unique action types in the database."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT DISTINCT action_type FROM action_segments")
        return [row["action_type"] for row in cursor]

    def get_all_objects(self) -> list[str]:
        """Get all unique object names in the database."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT DISTINCT primary_object_id FROM action_segments WHERE primary_object_id IS NOT NULL"
        )
        return [row["primary_object_id"] for row in cursor]

    def get_all_videos(self) -> list[dict[str, Any]]:
        """Get all videos in the database."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT id, video_path, duration, fps, width, height FROM video_metadata ORDER BY id"
        )
        return [
            {
                "id": row["id"],
                "video_path": row["video_path"],
                "duration": row["duration"],
                "fps": row["fps"],
                "width": row["width"],
                "height": row["height"],
            }
            for row in cursor
        ]

    def get_video_path(self, video_id: int) -> str | None:
        """Get video path by ID."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT video_path FROM video_metadata WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        return row["video_path"] if row else None

    def get_video_duration(self) -> float | None:
        """Get video duration from metadata (first video for backward compat)."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT duration FROM video_metadata LIMIT 1")
        row = cursor.fetchone()
        return row["duration"] if row else None

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
