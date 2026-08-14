# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Database browsing service wrapping SearchGraphTool."""

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_vector_db_path(graph_db_path: str) -> Path | None:
    """Resolve vector DB path from graph DB path (same directory, vector.db)."""
    p = Path(graph_db_path)
    if p.suffix == ".db":
        vector_path = p.parent / "vector.db"
    else:
        vector_path = p / "vector.db"
    return vector_path if vector_path.exists() else None


class DatabaseService:
    """Service for browsing entity graph databases."""

    def __init__(self, db_path: str):
        """Initialize database service.

        Args:
            db_path: Path to entity graph SQLite database.
        """
        self.db_path = db_path
        self._search_tool = None
        self._embedding_tool = None

    @property
    def search_tool(self):
        """Lazy load SearchGraphTool."""
        if self._search_tool is None:
            try:
                from ...retrieval.tools.search_graph import SearchGraphTool

                self._search_tool = SearchGraphTool(self.db_path)
            except Exception as e:
                logger.error(f"Failed to load SearchGraphTool: {e}")
                self._search_tool = None
        return self._search_tool

    def get_video_metadata(self) -> list[dict[str, Any]]:
        """Get metadata for all videos in database."""
        if not Path(self.db_path).exists():
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM video_metadata").fetchall()
            return [dict(row) for row in rows]

    def get_video_path_map(self) -> list[dict[str, Any]]:
        """Get video path ↔ id mapping for debug (same as get_video_metadata with consistent keys)."""
        rows = self.get_video_metadata()
        return [
            {
                "id": r.get("id"),
                "video_path": r.get("video_path"),
                "duration": round(r.get("duration", 0), 2),
                "fps": round(r.get("fps", 0), 2),
            }
            for r in rows
        ]

    def _get_embedding_tool(
        self,
        embedding_model: str = "google/siglip2-base-patch16-256",
        device: str = "cpu",
    ):
        """Return a cached SearchFramesTool (lazy-loaded once per service instance)."""
        vector_path = _resolve_vector_db_path(self.db_path)
        if not vector_path:
            return None

        if self._embedding_tool is None:
            from ...retrieval.tools.search_frames import SearchFramesTool

            self._embedding_tool = SearchFramesTool(
                vector_db_path=str(vector_path),
                embedding_model=embedding_model,
                device=device,
            )
        return self._embedding_tool

    def search_embeddings_by_text(
        self,
        text: str,
        top_k: int = 10,
        embedding_model: str = "google/siglip2-base-patch16-256",
        device: str = "cpu",
    ) -> list[dict[str, Any]]:
        """Search frame embeddings by text query; return top-k matches with RGB images.

        Requires vector.db in the same directory as the graph DB. Frames are extracted
        from video files on disk (video path from vector DB videos table).

        Returns:
            List of dicts with keys: image (numpy RGB), frame_id, video_id, timestamp,
            similarity, video_path. image is None if frame extraction fails.
        """
        vector_path = _resolve_vector_db_path(self.db_path)
        if not vector_path:
            logger.warning("No vector.db found; embedding search unavailable")
            return []

        try:
            from ...utils.vector_database import VectorDatabase
            from ...utils.video_processor import VideoProcessor
        except ImportError as e:
            logger.error(f"Import error for embedding search: {e}")
            return []

        try:
            tool = self._get_embedding_tool(embedding_model=embedding_model, device=device)
            if tool is None:
                logger.warning("Could not create embedding search tool")
                return []

            result = tool.execute(query=text, top_k=top_k)
            if not result.success or not result.data:
                return []

            vdb = VectorDatabase(str(vector_path))

            # Resolve video paths once per video_id
            video_path_cache: dict[str, str | None] = {}
            for r in result.data:
                if r.video_id not in video_path_cache:
                    video_path_cache[r.video_id] = vdb.get_video_path(r.video_id)

            # Cache VideoProcessors per video to avoid re-opening the same file
            processor_cache: dict[str, VideoProcessor | None] = {}

            out = []
            for r in result.data:
                video_path = video_path_cache.get(r.video_id)
                image = None
                if video_path and Path(video_path).exists():
                    try:
                        if video_path not in processor_cache:
                            processor_cache[video_path] = VideoProcessor(video_path)
                        proc = processor_cache[video_path]
                        if proc is not None:
                            frame = proc.get_frame_at_time(r.timestamp)
                            if frame is not None:
                                image = frame.image
                    except Exception as e:
                        logger.debug(f"Frame extraction failed for {r.frame_id}: {e}")
                        processor_cache[video_path] = None
                out.append(
                    {
                        "image": image,
                        "frame_id": r.frame_id,
                        "video_id": r.video_id,
                        "timestamp": round(r.timestamp, 2),
                        "similarity": round(r.similarity, 4),
                        "video_path": video_path or "",
                        "segment_id": getattr(r, "segment_id", None),
                    }
                )
            vdb.close()
            return out
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []

    def get_entities(
        self,
        video_id: int | None = None,
        entity_type: str | None = None,
        name_pattern: str | None = None,
        time_range: tuple[float, float] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get entities with filtering.

        Args:
            video_id: Filter by video ID.
            entity_type: Filter by entity type (person, object, location).
            name_pattern: Filter by name pattern (SQL LIKE).
            time_range: Filter by time range (start, end).
            limit: Maximum results.
            offset: Result offset.

        Returns:
            List of entity dictionaries.
        """
        if not Path(self.db_path).exists():
            return []

        conditions = []
        values = []

        if video_id is not None:
            conditions.append("video_id = ?")
            values.append(video_id)

        if entity_type:
            conditions.append("entity_type = ?")
            values.append(entity_type)

        if name_pattern:
            conditions.append("entity_id LIKE ?")
            values.append(f"%{name_pattern}%")

        if time_range:
            conditions.append("first_seen <= ? AND last_seen >= ?")
            values.extend([time_range[1], time_range[0]])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM entities
                {where_clause}
                ORDER BY first_seen
                LIMIT ? OFFSET ?
                """,
                values + [limit, offset],
            ).fetchall()

            return [dict(row) for row in rows]

    def get_relationships(
        self,
        video_id: int | None = None,
        rel_type: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        time_range: tuple[float, float] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get relationships with filtering."""
        if not Path(self.db_path).exists():
            return []

        conditions = []
        values = []

        if video_id is not None:
            conditions.append("video_id = ?")
            values.append(video_id)

        if rel_type:
            conditions.append("rel_type = ?")
            values.append(rel_type)

        if source_id:
            conditions.append("source_id LIKE ?")
            values.append(f"%{source_id}%")

        if target_id:
            conditions.append("target_id LIKE ?")
            values.append(f"%{target_id}%")

        if time_range:
            conditions.append("start_t <= ? AND end_t >= ?")
            values.extend([time_range[1], time_range[0]])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM relationships
                {where_clause}
                ORDER BY start_t
                LIMIT ? OFFSET ?
                """,
                values + [limit, offset],
            ).fetchall()

            return [dict(row) for row in rows]

    def get_segments(
        self,
        video_id: int | None = None,
        action_type: str | None = None,
        object_name: str | None = None,
        time_range: tuple[float, float] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get action segments with filtering."""
        if not Path(self.db_path).exists():
            return []

        conditions = []
        values = []

        if video_id is not None:
            conditions.append("video_id = ?")
            values.append(video_id)

        if action_type:
            conditions.append("action_type LIKE ?")
            values.append(f"%{action_type}%")

        if object_name:
            conditions.append("primary_object_id LIKE ?")
            values.append(f"%{object_name}%")

        if time_range:
            conditions.append("start_t <= ? AND end_t >= ?")
            values.extend([time_range[1], time_range[0]])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM action_segments
                {where_clause}
                ORDER BY start_t
                LIMIT ? OFFSET ?
                """,
                values + [limit, offset],
            ).fetchall()

            return [dict(row) for row in rows]

    def count_segments(
        self,
        video_id: int | None = None,
        action_type: str | None = None,
        object_name: str | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> int:
        """Count action segments matching filters (without LIMIT)."""
        if not Path(self.db_path).exists():
            return 0

        conditions: list[str] = []
        values: list[Any] = []

        if video_id is not None:
            conditions.append("video_id = ?")
            values.append(video_id)
        if action_type:
            conditions.append("action_type LIKE ?")
            values.append(f"%{action_type}%")
        if object_name:
            conditions.append("primary_object_id LIKE ?")
            values.append(f"%{object_name}%")
        if time_range:
            conditions.append("start_t <= ? AND end_t >= ?")
            values.extend([time_range[1], time_range[0]])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM action_segments {where_clause}", values
            ).fetchone()
            return row[0] if row else 0

    def count_entities(
        self,
        video_id: int | None = None,
        entity_type: str | None = None,
        name_pattern: str | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> int:
        """Count entities matching filters (without LIMIT)."""
        if not Path(self.db_path).exists():
            return 0

        conditions: list[str] = []
        values: list[Any] = []

        if video_id is not None:
            conditions.append("video_id = ?")
            values.append(video_id)
        if entity_type:
            conditions.append("entity_type = ?")
            values.append(entity_type)
        if name_pattern:
            conditions.append("entity_id LIKE ?")
            values.append(f"%{name_pattern}%")
        if time_range:
            conditions.append("first_seen <= ? AND last_seen >= ?")
            values.extend([time_range[1], time_range[0]])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM entities {where_clause}", values).fetchone()
            return row[0] if row else 0

    def count_relationships(
        self,
        video_id: int | None = None,
        rel_type: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> int:
        """Count relationships matching filters (without LIMIT)."""
        if not Path(self.db_path).exists():
            return 0

        conditions: list[str] = []
        values: list[Any] = []

        if video_id is not None:
            conditions.append("video_id = ?")
            values.append(video_id)
        if rel_type:
            conditions.append("rel_type = ?")
            values.append(rel_type)
        if source_id:
            conditions.append("source_id LIKE ?")
            values.append(f"%{source_id}%")
        if target_id:
            conditions.append("target_id LIKE ?")
            values.append(f"%{target_id}%")
        if time_range:
            conditions.append("start_t <= ? AND end_t >= ?")
            values.extend([time_range[1], time_range[0]])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM relationships {where_clause}", values
            ).fetchone()
            return row[0] if row else 0

    def get_filter_options(self) -> dict[str, list[str]]:
        """Get available filter options from database."""
        if not Path(self.db_path).exists():
            return {"entity_types": [], "rel_types": [], "actions": [], "objects": []}

        with sqlite3.connect(self.db_path) as conn:
            # Entity types
            entity_types = [
                row[0]
                for row in conn.execute("SELECT DISTINCT entity_type FROM entities").fetchall()
            ]

            # Relationship types
            rel_types = [
                row[0]
                for row in conn.execute("SELECT DISTINCT rel_type FROM relationships").fetchall()
            ]

            # Action types from segments
            actions = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT action_type FROM action_segments WHERE action_type IS NOT NULL"
                ).fetchall()
            ]

            # Objects from segments
            objects = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT primary_object_id FROM action_segments WHERE primary_object_id IS NOT NULL"
                ).fetchall()
            ]

        return {
            "entity_types": sorted(entity_types),
            "rel_types": sorted(rel_types),
            "actions": sorted(actions),
            "objects": sorted(objects),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        if not Path(self.db_path).exists():
            return {
                "videos": 0,
                "entities": 0,
                "relationships": 0,
                "segments": 0,
                "total_duration": 0,
            }

        with sqlite3.connect(self.db_path) as conn:
            videos = conn.execute("SELECT COUNT(*) FROM video_metadata").fetchone()[0]
            entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            segments = conn.execute("SELECT COUNT(*) FROM action_segments").fetchone()[0]
            duration = conn.execute("SELECT SUM(duration) FROM video_metadata").fetchone()[0] or 0

        return {
            "videos": videos,
            "entities": entities,
            "relationships": relationships,
            "segments": segments,
            "total_duration": round(duration, 2),
        }

    def get_graph_data(
        self,
        video_id: int | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Get data for entity graph visualization.

        Returns:
            Dict with 'nodes' and 'edges' for graph visualization.
        """
        entities = self.get_entities(video_id=video_id, time_range=time_range, limit=500)
        relationships = self.get_relationships(video_id=video_id, time_range=time_range, limit=500)

        # Build nodes
        nodes = []
        for e in entities:
            nodes.append(
                {
                    "id": e["entity_id"],
                    "label": e["entity_id"],
                    "type": e.get("entity_type", "unknown"),
                    "first_seen": e.get("first_seen", 0),
                    "last_seen": e.get("last_seen", 0),
                }
            )

        # Build edges
        edges = []
        node_ids = {n["id"] for n in nodes}
        for r in relationships:
            if r["source_id"] in node_ids and r["target_id"] in node_ids:
                edges.append(
                    {
                        "source": r["source_id"],
                        "target": r["target_id"],
                        "type": r.get("rel_type", "related"),
                        "start_t": r.get("start_t", 0),
                        "end_t": r.get("end_t", 0),
                    }
                )

        return {"nodes": nodes, "edges": edges}

    def search(
        self,
        query_type: str = "segments",
        action: str | None = None,
        object_name: str | None = None,
        entity_type: str | None = None,
        rel_type: str | None = None,
        time_range: tuple[float, float] | None = None,
        relaxation_level: int = 0,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Unified search interface using SearchGraphTool.

        Args:
            query_type: 'entities', 'relationships', or 'segments'.
            action: Action type filter.
            object_name: Object name filter.
            entity_type: Entity type filter.
            rel_type: Relationship type filter.
            time_range: Time range filter (start, end).
            relaxation_level: Search relaxation level (0-3).
            top_k: Maximum results.

        Returns:
            List of matching results.
        """
        if self.search_tool is None:
            logger.warning("SearchGraphTool not available, using direct SQL")
            if query_type == "segments":
                return self.get_segments(
                    action_type=action,
                    object_name=object_name,
                    time_range=time_range,
                    limit=top_k,
                )
            elif query_type == "entities":
                return self.get_entities(
                    entity_type=entity_type,
                    time_range=time_range,
                    limit=top_k,
                )
            elif query_type == "relationships":
                return self.get_relationships(
                    rel_type=rel_type,
                    time_range=time_range,
                    limit=top_k,
                )
            return []

        # Use SearchGraphTool
        result = self.search_tool.execute(
            query_type=query_type,
            action=action,
            object_name=object_name,
            entity_type=entity_type,
            rel_type=rel_type,
            start_time=time_range[0] if time_range else None,
            end_time=time_range[1] if time_range else None,
            relaxation_level=relaxation_level,
            top_k=top_k,
        )

        if result.success:
            return result.data if isinstance(result.data, list) else []
        return []
