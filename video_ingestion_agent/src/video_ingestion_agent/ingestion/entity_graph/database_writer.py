# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Database writer for entity graph with multi-video support."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from video_ingestion_agent.utils.types import Entity, Relationship
from video_ingestion_agent.utils.video_processor import VideoMetadata

if TYPE_CHECKING:
    from ..state import ActionSegment

logger = logging.getLogger(__name__)


# SQL schema for entity graph (v3: enforced FKs with ON DELETE CASCADE).
#
# Child tables reference video_metadata(id) with CASCADE so deleting a
# video can never orphan its rows (enforced via PRAGMA foreign_keys=ON
# on every connection). The former FOREIGN KEY(...) REFERENCES
# entities(entity_id) declarations on relationships/action_segments were
# dropped: entity_id is not unique (it repeats across videos, and
# *_object_id columns hold free-text object names), so under enforcement
# SQLite rejects every insert against such a parent with "foreign key
# mismatch" -- the declarations were never enforceable.
SCHEMA_SQL = """
-- Core Entities Table
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    properties TEXT,
    video_id INTEGER REFERENCES video_metadata(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CHECK (entity_type IN ('person', 'object', 'location')),
    CHECK (first_seen >= 0),
    CHECK (last_seen >= first_seen)
);

-- Relationships Table
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    start_t REAL NOT NULL,
    end_t REAL NOT NULL,
    confidence REAL DEFAULT 1.0,
    supporting_evidence TEXT,
    spatial_info TEXT,
    video_id INTEGER REFERENCES video_metadata(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CHECK (start_t >= 0),
    CHECK (end_t >= start_t),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Action Segments Table
CREATE TABLE IF NOT EXISTS action_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,
    start_t REAL NOT NULL,
    end_t REAL NOT NULL,
    primary_object_id TEXT,
    secondary_object_id TEXT,
    hand TEXT,
    success BOOLEAN DEFAULT TRUE,
    quality_score REAL,
    visual_evidence TEXT,
    video_id INTEGER REFERENCES video_metadata(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CHECK (start_t >= 0),
    CHECK (end_t >= start_t),
    CHECK (quality_score IS NULL OR (quality_score >= 0.0 AND quality_score <= 1.0))
);

-- Video Metadata Table
CREATE TABLE IF NOT EXISTS video_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path TEXT UNIQUE NOT NULL,
    duration REAL NOT NULL,
    fps REAL NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CHECK (duration > 0),
    CHECK (fps > 0)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_time ON entities(first_seen, last_seen);
CREATE INDEX IF NOT EXISTS idx_entities_id ON entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_entities_video ON entities(video_id);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(rel_type);
CREATE INDEX IF NOT EXISTS idx_relationships_time ON relationships(start_t, end_t);
CREATE INDEX IF NOT EXISTS idx_relationships_video ON relationships(video_id);

CREATE INDEX IF NOT EXISTS idx_actions_type ON action_segments(action_type);
CREATE INDEX IF NOT EXISTS idx_actions_time ON action_segments(start_t, end_t);
CREATE INDEX IF NOT EXISTS idx_actions_object ON action_segments(primary_object_id);
CREATE INDEX IF NOT EXISTS idx_actions_video ON action_segments(video_id);
"""


class DatabaseWriter:
    """Write entity graph to SQLite database with multi-video support."""

    def __init__(self, db_path: str):
        """
        Initialize database writer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if database already exists (for migration)
        db_exists = self.db_path.exists()

        # Initialize database with WAL mode for concurrent multi-shard writes
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

        # Migrate existing databases if needed
        if db_exists:
            self._migrate_if_needed()
            self._migrate_fk_cascade_if_needed()

        # SQLite leaves foreign key enforcement OFF per connection; enable it
        # so a dangling video_id can never be written and deletes cascade.
        # Must come after migration (legacy tables may hold orphans and
        # unenforceable FK declarations until rebuilt).
        self.conn.execute("PRAGMA foreign_keys=ON")

        logger.info(f"Database initialized: {self.db_path}")

    def _create_schema(self):
        """Create database schema."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def _migrate_if_needed(self):
        """Migrate existing single-video database to multi-video schema."""
        # Check if video_id column exists in entities table
        cursor = self.conn.execute("PRAGMA table_info(entities)")
        columns = {row[1] for row in cursor.fetchall()}

        if "video_id" not in columns:
            logger.info("Migrating database to multi-video schema...")

            # Get existing video's id (should be only one in old DBs)
            cursor = self.conn.execute("SELECT id FROM video_metadata LIMIT 1")
            row = cursor.fetchone()
            default_video_id = row[0] if row else None

            # Add video_id column to each table
            tables = ["entities", "relationships", "action_segments"]
            for table in tables:
                try:
                    if default_video_id is not None:
                        self.conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN video_id INTEGER DEFAULT {default_video_id}"
                        )
                    else:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN video_id INTEGER")
                    logger.info(f"  Added video_id column to {table}")
                except sqlite3.OperationalError as e:
                    # Column may already exist
                    if "duplicate column" not in str(e).lower():
                        raise

            # Create indexes (IF NOT EXISTS handles duplicates)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_video ON entities(video_id)")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relationships_video ON relationships(video_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_actions_video ON action_segments(video_id)"
            )

            self.conn.commit()
            logger.info("Migration complete")

    # Tables whose video_id FK must be enforced with ON DELETE CASCADE.
    _CHILD_TABLES = ("entities", "relationships", "action_segments")

    def _fk_layout_is_current(self, table: str) -> bool:
        """True if *table* has exactly one FK: video_metadata CASCADE."""
        fks = self.conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        return (
            len(fks) == 1
            and fks[0]["table"] == "video_metadata"
            and fks[0]["on_delete"] == "CASCADE"
        )

    def _migrate_fk_cascade_if_needed(self):
        """Rebuild legacy child tables to the enforced-FK schema.

        Legacy databases either declare the video_metadata FK with NO
        ACTION (schema v2), have no FK at all on video_id (v1 databases
        that gained the column via ALTER TABLE), or carry unenforceable
        FKs against the non-unique entities.entity_id column. SQLite
        cannot alter FKs in place, so each stale table is rebuilt:
        orphans are removed first (enforcement cannot be enabled over
        dangling rows), then rows are copied into a fresh table created
        from the current schema.
        """
        stale = [t for t in self._CHILD_TABLES if not self._fk_layout_is_current(t)]
        if not stale:
            return

        logger.info(f"Migrating {stale} to enforced ON DELETE CASCADE schema...")
        # Individual statements from the canonical schema (executescript
        # would implicitly COMMIT the migration transaction, breaking
        # atomicity, so each statement is executed on its own).
        statements = [s.strip() for s in SCHEMA_SQL.split(";") if s.strip()]

        def create_table_stmt(table: str) -> str:
            for stmt in statements:
                if f"CREATE TABLE IF NOT EXISTS {table} (" in stmt:
                    return stmt
            raise RuntimeError(f"No schema statement found for table {table}")

        try:
            for table in stale:
                # 1. Enforcement cannot be enabled over dangling rows.
                purged = self.conn.execute(
                    f"DELETE FROM {table} "
                    "WHERE video_id IS NOT NULL "
                    "AND video_id NOT IN (SELECT id FROM video_metadata)"
                ).rowcount
                if purged:
                    logger.info(f"  {table}: removed {purged} orphaned rows")

                # 2. Rebuild: move the old table aside, recreate from the
                #    canonical schema, copy the shared columns across.
                self.conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
                self.conn.execute(create_table_stmt(table))

                old_cols = {
                    row[1]
                    for row in self.conn.execute(f"PRAGMA table_info({table}_legacy)").fetchall()
                }
                new_cols = [
                    row[1]
                    for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
                    if row[1] in old_cols
                ]
                cols = ", ".join(new_cols)
                self.conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {table}_legacy")
                self.conn.execute(f"DROP TABLE {table}_legacy")

            # Indexes on the rebuilt tables were dropped with the legacy
            # tables; recreate them (IF NOT EXISTS makes this idempotent).
            for stmt in statements:
                if "CREATE INDEX" in stmt:
                    self.conn.execute(stmt)

            self.conn.commit()
            logger.info("FK cascade migration complete")
        except Exception:
            self.conn.rollback()
            raise

    def write_video_metadata(self, metadata: VideoMetadata) -> int:
        """
        Write video metadata and return video_id.

        Re-ingesting a video that is already in the database replaces its
        previous run: the video keeps its stable id (updated in place) and
        all of its existing action_segments, entities, and relationships
        are deleted in the same transaction. The old implementation used
        ``INSERT OR REPLACE``, whose delete-then-insert gave the video a
        fresh AUTOINCREMENT id while (with SQLite FK enforcement off) the
        previous run's child rows survived pointing at the dead id --
        orphans that stayed searchable but could never resolve a
        video_path again.

        Args:
            metadata: Video metadata object

        Returns:
            video_id: The database ID for this video
        """
        cursor = self.conn.execute(
            "SELECT id FROM video_metadata WHERE video_path = ?", (metadata.path,)
        )
        row = cursor.fetchone()

        if row is not None:
            video_id = row[0]
            purged = {}
            for table in ("action_segments", "entities", "relationships"):
                cur = self.conn.execute(f"DELETE FROM {table} WHERE video_id = ?", (video_id,))
                purged[table] = cur.rowcount
            self.conn.execute(
                """
                UPDATE video_metadata
                SET duration = ?, fps = ?, width = ?, height = ?
                WHERE id = ?
                """,
                (metadata.duration, metadata.fps, metadata.width, metadata.height, video_id),
            )
            logger.info(
                f"Re-ingest of {metadata.path} (video_id={video_id}): purged previous run "
                f"({purged['action_segments']} segments, {purged['entities']} entities, "
                f"{purged['relationships']} relationships)"
            )
        else:
            cursor = self.conn.execute(
                """
                INSERT INTO video_metadata
                (video_path, duration, fps, width, height)
                VALUES (?, ?, ?, ?, ?)
                """,
                (metadata.path, metadata.duration, metadata.fps, metadata.width, metadata.height),
            )
            video_id = cursor.lastrowid

        self.conn.commit()

        logger.info(f"Wrote video metadata: {metadata.path} (video_id={video_id})")
        return video_id

    def get_video_id(self, video_path: str) -> int | None:
        """
        Get video_id for a given video path.

        Args:
            video_path: Path to video file

        Returns:
            video_id or None if not found
        """
        cursor = self.conn.execute(
            "SELECT id FROM video_metadata WHERE video_path = ?", (video_path,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def write_entities(self, entities: list[Entity], video_id: int | None = None):
        """
        Write entities to database.

        Args:
            entities: List of Entity objects
            video_id: Video ID to associate entities with (required for multi-video)
        """
        for entity in entities:
            # Serialize properties as JSON
            properties_json = json.dumps(entity.properties)

            self.conn.execute(
                """
                INSERT INTO entities
                (entity_id, entity_type, first_seen, last_seen, properties, video_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity.entity_id,
                    entity.entity_type.value,
                    entity.first_seen,
                    entity.last_seen,
                    properties_json,
                    video_id,
                ),
            )

        self.conn.commit()
        logger.info(f"Wrote {len(entities)} entities (video_id={video_id})")

    def write_relationships(self, relationships: list[Relationship], video_id: int | None = None):
        """
        Write relationships to database.

        Args:
            relationships: List of Relationship objects
            video_id: Video ID to associate relationships with (required for multi-video)
        """
        for rel in relationships:
            # Serialize spatial_info as JSON
            spatial_json = json.dumps(rel.spatial_info) if rel.spatial_info else None

            self.conn.execute(
                """
                INSERT INTO relationships
                (source_id, target_id, rel_type, start_t, end_t,
                 confidence, supporting_evidence, spatial_info, video_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel.source_id,
                    rel.target_id,
                    rel.rel_type.value,
                    rel.start_t,
                    rel.end_t,
                    rel.confidence,
                    rel.supporting_evidence,
                    spatial_json,
                    video_id,
                ),
            )

        self.conn.commit()
        logger.info(f"Wrote {len(relationships)} relationships (video_id={video_id})")

    def write_action_segments(self, segments: list["ActionSegment"], video_id: int | None = None):
        """
        Write action segments to database.

        Args:
            segments: List of ActionSegment objects
            video_id: Video ID to associate segments with (required for multi-video)
        """
        for seg in segments:
            self.conn.execute(
                """
                INSERT INTO action_segments
                (action_type, start_t, end_t, primary_object_id, visual_evidence, quality_score, video_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seg.action,
                    seg.start_t,
                    seg.end_t,
                    seg.object_name,
                    seg.description,
                    seg.confidence,
                    video_id,
                ),
            )

        self.conn.commit()
        logger.info(f"Wrote {len(segments)} action segments (video_id={video_id})")

    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")
