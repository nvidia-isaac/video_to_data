# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vector database for storing and retrieving frame embeddings."""

import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result from vector database search."""

    frame_id: str
    video_id: str
    timestamp: float
    similarity: float
    embedding: np.ndarray | None = None
    metadata: dict | None = None
    segment_id: str | None = None


class VectorDatabase:
    """
    Vector database for frame embeddings with hybrid search.

    Supports:
    - Efficient vector similarity search
    - Attribute-based filtering (time, location, etc.)
    - Hybrid search (vector + attributes)

    Storage:
    - SQLite for metadata and attributes
    - Serialized numpy arrays for embeddings

    Example:
        db = VectorDatabase("embeddings.db", embedding_dim=768)

        # Add frame embedding
        db.add_frame(
            frame_id="frame_001",
            timestamp=10.5,
            embedding=embedding_vector,
            metadata={"location": "kitchen"}
        )

        # Search
        results = db.search(
            query_embedding=query_vector,
            filters={"timestamp": (10.0, 20.0)},
            top_k=50
        )
    """

    def __init__(self, db_path: str, embedding_dim: int = 768):
        """
        Initialize vector database.

        Args:
            db_path: Path to SQLite database file
            embedding_dim: Dimension of embeddings (default: 768 for SigLIP)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        db_existed = self.db_path.exists()

        self.embedding_dim = embedding_dim
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name

        # Enable WAL mode for better concurrency (supports multi-shard writes)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=30000")

        self._create_tables(migrate=db_existed)
        logger.info(f"Vector database initialized: {db_path}")

    def _create_tables(self, *, migrate: bool = False):
        """Create database tables if they don't exist.

        Args:
            migrate: When True the database pre-existed and may lack newer
                columns.  The migration step runs *between* table creation and
                index creation so that indexes on new columns don't fail.
        """

        # Video metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                duration REAL NOT NULL,
                fps REAL NOT NULL,
                width INTEGER,
                height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Frame embeddings table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS frame_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT UNIQUE NOT NULL,
                video_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                segment_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(video_id) REFERENCES videos(id)
            )
        """)

        if migrate:
            self._migrate_if_needed()

        # Create indexes for efficient querying
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_video_time
            ON frame_embeddings(video_id, timestamp)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_timestamp
            ON frame_embeddings(timestamp)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_segment
            ON frame_embeddings(segment_id)
        """)

        self.conn.commit()

    def _migrate_if_needed(self):
        """Add segment_id column to existing databases that lack it."""
        cursor = self.conn.execute("PRAGMA table_info(frame_embeddings)")
        columns = {row[1] for row in cursor.fetchall()}

        if "segment_id" not in columns:
            logger.info("Migrating vector DB: adding segment_id column to frame_embeddings")
            try:
                self.conn.execute("ALTER TABLE frame_embeddings ADD COLUMN segment_id TEXT")
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_frame_segment ON frame_embeddings(segment_id)"
                )
                self.conn.commit()
                logger.info("Migration complete: segment_id column added")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

    def add_video(
        self, video_id: str, path: str, duration: float, fps: float, width: int, height: int
    ):
        """
        Add video metadata.

        Args:
            video_id: Unique video identifier
            path: Path to video file
            duration: Duration in seconds
            fps: Frames per second
            width: Video width
            height: Video height
        """
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO videos (id, path, duration, fps, width, height)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (video_id, path, duration, fps, width, height),
            )
            self.conn.commit()
            logger.debug(f"Added video metadata: {video_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to add video metadata: {e}")
            raise

    def add_frame(
        self,
        frame_id: str,
        video_id: str,
        timestamp: float,
        embedding: np.ndarray,
        metadata: dict | None = None,
        segment_id: str | None = None,
    ):
        """
        Add frame embedding to database.

        Args:
            frame_id: Unique frame identifier
            video_id: Video this frame belongs to
            timestamp: Timestamp in seconds
            embedding: Frame embedding vector
            metadata: Optional metadata dict
            segment_id: Optional segment/clip this frame belongs to
        """
        # Validate embedding dimension
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )

        # Serialize embedding
        embedding_blob = pickle.dumps(embedding.astype(np.float32))

        # Serialize metadata
        metadata_json = json.dumps(metadata) if metadata else None

        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO frame_embeddings
                (frame_id, video_id, timestamp, embedding, metadata, segment_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (frame_id, video_id, timestamp, embedding_blob, metadata_json, segment_id),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to add frame embedding: {e}")
            raise

    def add_frames_batch(
        self,
        frames: list[tuple[str, str, float, np.ndarray, dict | None, str | None]],
    ):
        """
        Add multiple frames in batch for efficiency.

        Args:
            frames: List of (frame_id, video_id, timestamp, embedding, metadata, segment_id).
                    The segment_id element is optional -- 5-element tuples are accepted
                    for backward compatibility.
        """
        data = []
        for item in frames:
            if len(item) == 6:
                frame_id, video_id, timestamp, embedding, metadata, segment_id = item
            else:
                frame_id, video_id, timestamp, embedding, metadata = item[:5]
                segment_id = None

            # Validate and serialize
            if embedding.shape[0] != self.embedding_dim:
                logger.warning(f"Skipping frame {frame_id}: wrong embedding dim")
                continue

            embedding_blob = pickle.dumps(embedding.astype(np.float32))
            metadata_json = json.dumps(metadata) if metadata else None

            data.append((frame_id, video_id, timestamp, embedding_blob, metadata_json, segment_id))

        try:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO frame_embeddings
                (frame_id, video_id, timestamp, embedding, metadata, segment_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                data,
            )
            self.conn.commit()
            logger.info(f"Added {len(data)} frame embeddings in batch")
        except sqlite3.Error as e:
            logger.error(f"Failed to add frames batch: {e}")
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        video_id: str | None = None,
        segment_id: str | None = None,
        time_range: tuple[float, float] | None = None,
        metadata_filters: dict | None = None,
        top_k: int = 50,
        return_embeddings: bool = False,
    ) -> list[FrameResult]:
        """
        Hybrid search: vector similarity + attribute filters.

        Args:
            query_embedding: Query vector
            video_id: Filter by video (None = search all videos)
            segment_id: Filter by segment/clip (None = search all segments)
            time_range: (start_t, end_t) to filter by time
            metadata_filters: Dict of metadata filters (future: JSON queries)
            top_k: Number of results to return
            return_embeddings: Whether to return embedding vectors

        Returns:
            List of FrameResult, sorted by similarity (descending)
        """
        # Build SQL query
        sql = (
            "SELECT frame_id, video_id, timestamp, embedding, metadata, segment_id "
            "FROM frame_embeddings WHERE 1=1"
        )
        params: list = []

        # Apply video filter
        if video_id:
            sql += " AND video_id = ?"
            params.append(video_id)

        # Apply segment filter
        if segment_id:
            sql += " AND segment_id = ?"
            params.append(segment_id)

        # Apply time filter
        if time_range:
            start_t, end_t = time_range
            sql += " AND timestamp >= ? AND timestamp <= ?"
            params.extend([start_t, end_t])

        # Execute query
        cursor = self.conn.execute(sql, params)

        # Compute similarities
        results = []
        for row in cursor:
            # Deserialize embedding
            embedding = pickle.loads(row["embedding"])

            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)

            # Parse metadata
            metadata = json.loads(row["metadata"]) if row["metadata"] else None

            # Apply metadata filters if specified
            if metadata_filters and metadata:
                if not self._matches_metadata_filters(metadata, metadata_filters):
                    continue

            results.append(
                FrameResult(
                    frame_id=row["frame_id"],
                    video_id=row["video_id"],
                    timestamp=row["timestamp"],
                    similarity=similarity,
                    embedding=embedding if return_embeddings else None,
                    metadata=metadata,
                    segment_id=row["segment_id"],
                )
            )

        # Sort by similarity (descending)
        results.sort(key=lambda x: x.similarity, reverse=True)

        # Return top-k
        return results[:top_k]

    def search_by_time(
        self, start_t: float, end_t: float, video_id: str | None = None
    ) -> list[FrameResult]:
        """
        Get all frames in time window (no vector search).

        Args:
            start_t: Start time in seconds
            end_t: End time in seconds
            video_id: Optional video filter

        Returns:
            List of FrameResult, sorted by timestamp
        """
        sql = """
            SELECT frame_id, video_id, timestamp, metadata, segment_id
            FROM frame_embeddings
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: list = [start_t, end_t]

        if video_id:
            sql += " AND video_id = ?"
            params.append(video_id)

        sql += " ORDER BY timestamp"

        cursor = self.conn.execute(sql, params)

        results = []
        for row in cursor:
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            results.append(
                FrameResult(
                    frame_id=row["frame_id"],
                    video_id=row["video_id"],
                    timestamp=row["timestamp"],
                    similarity=1.0,  # No similarity score for time-based search
                    metadata=metadata,
                    segment_id=row["segment_id"],
                )
            )

        return results

    def search_by_segment(
        self,
        segment_id: str,
        video_id: str | None = None,
        query_embedding: np.ndarray | None = None,
        top_k: int = 50,
    ) -> list[FrameResult]:
        """
        Get all frames belonging to a segment, optionally ranked by similarity.

        Args:
            segment_id: Segment/clip identifier
            video_id: Optional additional video filter
            query_embedding: If provided, results are ranked by cosine similarity
            top_k: Maximum results to return

        Returns:
            List of FrameResult, sorted by similarity (if query given) or timestamp
        """
        sql = (
            "SELECT frame_id, video_id, timestamp, embedding, metadata, segment_id "
            "FROM frame_embeddings WHERE segment_id = ?"
        )
        params: list = [segment_id]

        if video_id:
            sql += " AND video_id = ?"
            params.append(video_id)

        cursor = self.conn.execute(sql, params)

        results = []
        for row in cursor:
            embedding = pickle.loads(row["embedding"])
            similarity = (
                self._cosine_similarity(query_embedding, embedding)
                if query_embedding is not None
                else 1.0
            )
            metadata = json.loads(row["metadata"]) if row["metadata"] else None
            results.append(
                FrameResult(
                    frame_id=row["frame_id"],
                    video_id=row["video_id"],
                    timestamp=row["timestamp"],
                    similarity=similarity,
                    metadata=metadata,
                    segment_id=row["segment_id"],
                )
            )

        if query_embedding is not None:
            results.sort(key=lambda x: x.similarity, reverse=True)
        else:
            results.sort(key=lambda x: x.timestamp)

        return results[:top_k]

    def get_frame(self, frame_id: str, return_embedding: bool = False) -> FrameResult | None:
        """
        Get specific frame by ID.

        Args:
            frame_id: Frame identifier
            return_embedding: Whether to return embedding vector

        Returns:
            FrameResult or None if not found
        """
        sql = (
            "SELECT frame_id, video_id, timestamp, embedding, metadata, segment_id "
            "FROM frame_embeddings WHERE frame_id = ?"
        )
        cursor = self.conn.execute(sql, (frame_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        embedding = None
        if return_embedding:
            embedding = pickle.loads(row["embedding"])

        metadata = json.loads(row["metadata"]) if row["metadata"] else None

        return FrameResult(
            frame_id=row["frame_id"],
            video_id=row["video_id"],
            timestamp=row["timestamp"],
            similarity=1.0,
            embedding=embedding,
            metadata=metadata,
            segment_id=row["segment_id"],
        )

    def count_frames(self, video_id: str | None = None) -> int:
        """Count frames in database."""
        if video_id:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM frame_embeddings WHERE video_id = ?", (video_id,)
            )
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM frame_embeddings")

        return cursor.fetchone()[0]

    def get_video_path(self, video_id: str) -> str | None:
        """Get filesystem path for a video by its id.

        Args:
            video_id: Video identifier (e.g. from frame_embeddings.video_id).

        Returns:
            Path string or None if not found.
        """
        cursor = self.conn.execute("SELECT path FROM videos WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        return row["path"] if row else None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # Normalize
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)

        # Dot product
        return float(np.dot(a_norm, b_norm))

    @staticmethod
    def _matches_metadata_filters(metadata: dict, filters: dict) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Vector database closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
