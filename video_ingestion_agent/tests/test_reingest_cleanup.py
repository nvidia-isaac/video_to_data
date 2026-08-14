# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Re-ingest referential integrity tests.

Re-ingesting a video that already exists in graph.db previously orphaned
the prior run's children: INSERT OR REPLACE deleted the video_metadata
row and re-inserted it under a fresh AUTOINCREMENT id while the old
action_segments/entities/relationships survived (SQLite FK enforcement
is off), still pointing at the dead id. Orphans stayed searchable but
could never resolve a video_path again.
"""

import numpy as np
import pytest

from video_ingestion_agent.ingestion.entity_graph.database_writer import DatabaseWriter
from video_ingestion_agent.utils.vector_database import VectorDatabase
from video_ingestion_agent.utils.video_processor import VideoMetadata


def _meta(path: str = "videos/a.mp4", duration: float = 10.0) -> VideoMetadata:
    return VideoMetadata(
        path=path, duration=duration, fps=30.0, width=640, height=480, frame_count=300
    )


def _insert_children(writer: DatabaseWriter, video_id: int, n: int = 3) -> None:
    for i in range(n):
        writer.conn.execute(
            "INSERT INTO action_segments (action_type, start_t, end_t, video_id) "
            "VALUES (?, ?, ?, ?)",
            (f"action_{i}", float(i), float(i + 1), video_id),
        )
        writer.conn.execute(
            "INSERT INTO entities (entity_id, entity_type, first_seen, last_seen, video_id) "
            "VALUES (?, 'object', 0.0, 1.0, ?)",
            (f"e{video_id}_{i}", video_id),
        )
        writer.conn.execute(
            "INSERT INTO relationships (source_id, target_id, rel_type, start_t, end_t, video_id) "
            "VALUES (?, ?, 'interacts-with', 0.0, 1.0, ?)",
            (f"e{video_id}_{i}", f"e{video_id}_{(i + 1) % n}", video_id),
        )
    writer.conn.commit()


def _orphan_count(writer: DatabaseWriter, table: str) -> int:
    return writer.conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE video_id NOT IN (SELECT id FROM video_metadata)"
    ).fetchone()[0]


class TestGraphDbReingest:
    def test_reingest_keeps_stable_video_id(self, tmp_path):
        writer = DatabaseWriter(tmp_path / "graph.db")
        first_id = writer.write_video_metadata(_meta())
        second_id = writer.write_video_metadata(_meta(duration=12.0))
        assert second_id == first_id

        duration = writer.conn.execute(
            "SELECT duration FROM video_metadata WHERE id = ?", (first_id,)
        ).fetchone()[0]
        assert duration == pytest.approx(12.0)

    def test_reingest_purges_previous_children(self, tmp_path):
        writer = DatabaseWriter(tmp_path / "graph.db")
        video_id = writer.write_video_metadata(_meta())
        _insert_children(writer, video_id, n=4)

        writer.write_video_metadata(_meta())

        for table in ("action_segments", "entities", "relationships"):
            total = writer.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert total == 0, f"{table} should be purged on re-ingest"
            assert _orphan_count(writer, table) == 0

    def test_reingest_leaves_no_orphans_after_new_children(self, tmp_path):
        writer = DatabaseWriter(tmp_path / "graph.db")
        video_id = writer.write_video_metadata(_meta())
        _insert_children(writer, video_id, n=4)

        new_id = writer.write_video_metadata(_meta())
        _insert_children(writer, new_id, n=2)

        for table in ("action_segments", "entities", "relationships"):
            total = writer.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert total == 2, f"{table} should contain only the new run's rows"
            assert _orphan_count(writer, table) == 0

    def test_reingest_does_not_touch_other_videos(self, tmp_path):
        writer = DatabaseWriter(tmp_path / "graph.db")
        id_a = writer.write_video_metadata(_meta("videos/a.mp4"))
        id_b = writer.write_video_metadata(_meta("videos/b.mp4"))
        _insert_children(writer, id_a, n=3)
        _insert_children(writer, id_b, n=5)

        writer.write_video_metadata(_meta("videos/a.mp4"))

        b_segments = writer.conn.execute(
            "SELECT COUNT(*) FROM action_segments WHERE video_id = ?", (id_b,)
        ).fetchone()[0]
        assert b_segments == 5
        for table in ("action_segments", "entities", "relationships"):
            assert _orphan_count(writer, table) == 0


# Pre-hardening (v2) child-table schema, used to fabricate legacy databases:
# NO ACTION on the video_metadata FK plus unenforceable FKs against the
# non-unique entities.entity_id column.
_LEGACY_SCHEMA = """
CREATE TABLE video_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path TEXT UNIQUE NOT NULL,
    duration REAL NOT NULL,
    fps REAL NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    properties TEXT,
    video_id INTEGER REFERENCES video_metadata(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    start_t REAL NOT NULL,
    end_t REAL NOT NULL,
    confidence REAL DEFAULT 1.0,
    supporting_evidence TEXT,
    spatial_info TEXT,
    video_id INTEGER REFERENCES video_metadata(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_id) REFERENCES entities(entity_id),
    FOREIGN KEY(target_id) REFERENCES entities(entity_id)
);
CREATE TABLE action_segments (
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
    video_id INTEGER REFERENCES video_metadata(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(primary_object_id) REFERENCES entities(entity_id),
    FOREIGN KEY(secondary_object_id) REFERENCES entities(entity_id)
);
"""


class TestForeignKeyEnforcement:
    """FK enforcement + ON DELETE CASCADE make orphans structurally impossible."""

    def test_pragma_foreign_keys_is_on(self, tmp_path):
        writer = DatabaseWriter(tmp_path / "graph.db")
        assert writer.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_dangling_child_insert_is_rejected(self, tmp_path):
        import sqlite3

        writer = DatabaseWriter(tmp_path / "graph.db")
        with pytest.raises(sqlite3.IntegrityError):
            writer.conn.execute(
                "INSERT INTO action_segments (action_type, start_t, end_t, video_id) "
                "VALUES ('stir', 0.0, 1.0, 999)"
            )

    def test_parent_delete_cascades_to_children(self, tmp_path):
        writer = DatabaseWriter(tmp_path / "graph.db")
        video_id = writer.write_video_metadata(_meta())
        _insert_children(writer, video_id, n=3)

        writer.conn.execute("DELETE FROM video_metadata WHERE id = ?", (video_id,))
        writer.conn.commit()

        for table in ("action_segments", "entities", "relationships"):
            assert writer.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    def test_legacy_database_is_migrated_on_open(self, tmp_path):
        import sqlite3

        db_path = tmp_path / "graph.db"
        legacy = sqlite3.connect(db_path)
        legacy.executescript(_LEGACY_SCHEMA)
        legacy.execute(
            "INSERT INTO video_metadata (id, video_path, duration, fps) "
            "VALUES (1, 'videos/a.mp4', 10.0, 30.0)"
        )
        # live children (video 1) + orphans (video 99, parent long gone)
        for vid in (1, 99):
            legacy.execute(
                "INSERT INTO action_segments (action_type, start_t, end_t, video_id) "
                "VALUES ('stir', 0.0, 1.0, ?)",
                (vid,),
            )
            legacy.execute(
                "INSERT INTO entities (entity_id, entity_type, first_seen, last_seen, video_id) "
                "VALUES ('e1', 'object', 0.0, 1.0, ?)",
                (vid,),
            )
            legacy.execute(
                "INSERT INTO relationships (source_id, target_id, rel_type, start_t, end_t, video_id) "
                "VALUES ('e1', 'e1', 'interacts-with', 0.0, 1.0, ?)",
                (vid,),
            )
        legacy.commit()
        legacy.close()

        writer = DatabaseWriter(db_path)

        for table in ("action_segments", "entities", "relationships"):
            # orphans removed, live rows preserved
            rows = writer.conn.execute(f"SELECT video_id FROM {table}").fetchall()
            assert [r[0] for r in rows] == [1], f"{table} should keep only live rows"
            # FK layout upgraded: single CASCADE FK to video_metadata
            fks = writer.conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
            assert len(fks) == 1
            assert fks[0]["table"] == "video_metadata"
            assert fks[0]["on_delete"] == "CASCADE"

        # enforcement is live on the migrated database
        with pytest.raises(sqlite3.IntegrityError):
            writer.conn.execute(
                "INSERT INTO entities (entity_id, entity_type, first_seen, last_seen, video_id) "
                "VALUES ('e2', 'object', 0.0, 1.0, 999)"
            )

    def test_migration_is_idempotent(self, tmp_path):
        writer = DatabaseWriter(tmp_path / "graph.db")
        video_id = writer.write_video_metadata(_meta())
        _insert_children(writer, video_id, n=2)
        writer.close()

        reopened = DatabaseWriter(tmp_path / "graph.db")
        assert reopened.conn.execute("SELECT COUNT(*) FROM action_segments").fetchone()[0] == 2

    def test_vector_db_rejects_dangling_frame(self, tmp_path):
        import sqlite3

        db = VectorDatabase(str(tmp_path / "vector.db"), embedding_dim=8)
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO frame_embeddings (frame_id, video_id, timestamp, embedding) "
                "VALUES ('f1', 'no_such_video', 0.0, x'00')"
            )


class TestZeroClipRunsDoNotMarkProcessed:
    """A run with no valid clips must not write video_metadata.

    Writing it would make batch --resume treat the failed video as
    processed and skip it forever.
    """

    def _state(self, tmp_path, clips, verifications):
        from video_ingestion_agent.ingestion.config import PipelineConfig

        config = PipelineConfig()
        config.database.directory = str(tmp_path / "db")
        return {
            "video_path": "videos/a.mp4",
            "clips": clips,
            "verifications": verifications,
            "linked_entities": [],
            "linked_relationships": [],
            "frames": [],
            "embeddings": [],
            "config": config,
        }

    def test_zero_clips_skips_all_db_writes(self, tmp_path):
        from video_ingestion_agent.ingestion.entity_graph_nodes import database_write_node

        result = database_write_node(self._state(tmp_path, clips=[], verifications=[]))

        assert result["status"] == "db_write_skipped"
        assert result["db_paths"] == {}
        assert not (tmp_path / "db" / "graph.db").exists()
        assert not (tmp_path / "db" / "vector.db").exists()

    def test_all_invalid_clips_also_skip(self, tmp_path):
        from video_ingestion_agent.ingestion.entity_graph_nodes import database_write_node
        from video_ingestion_agent.ingestion.state import ClipContext, VerificationResult

        clip = ClipContext(clip_id="c1", video_path="videos/a.mp4", start_t=0.0, end_t=1.0)
        bad = VerificationResult(clip_id="c1", is_valid=False, verification_score=0.0)

        result = database_write_node(self._state(tmp_path, clips=[clip], verifications=[bad]))

        assert result["status"] == "db_write_skipped"
        assert not (tmp_path / "db" / "graph.db").exists()


class TestVectorDbReingest:
    def _add_frames(self, db: VectorDatabase, video_id: str, n: int, run: int) -> None:
        frames = [
            (
                f"{video_id}/frame_{i}",
                video_id,
                float(i),
                np.zeros(8, dtype=np.float32),
                None,
                f"clip_{run}_{i}",
            )
            for i in range(n)
        ]
        db.add_frames_batch(frames)

    def test_readd_purges_stale_frame_embeddings(self, tmp_path):
        db = VectorDatabase(str(tmp_path / "vector.db"), embedding_dim=8)
        db.add_video("vid_a", "videos/a.mp4", 10.0, 30.0, 640, 480)
        self._add_frames(db, "vid_a", n=5, run=1)

        # Re-ingest produces fewer frames; without the purge, frames 3-4
        # from run 1 would survive with stale segment ids.
        db.add_video("vid_a", "videos/a.mp4", 10.0, 30.0, 640, 480)
        self._add_frames(db, "vid_a", n=3, run=2)

        rows = db.conn.execute(
            "SELECT frame_id, segment_id FROM frame_embeddings WHERE video_id = 'vid_a'"
        ).fetchall()
        assert len(rows) == 3
        assert all(seg.startswith("clip_2_") for _, seg in rows)

    def test_readd_does_not_touch_other_videos(self, tmp_path):
        db = VectorDatabase(str(tmp_path / "vector.db"), embedding_dim=8)
        db.add_video("vid_a", "videos/a.mp4", 10.0, 30.0, 640, 480)
        db.add_video("vid_b", "videos/b.mp4", 10.0, 30.0, 640, 480)
        self._add_frames(db, "vid_a", n=4, run=1)
        self._add_frames(db, "vid_b", n=6, run=1)

        db.add_video("vid_a", "videos/a.mp4", 10.0, 30.0, 640, 480)

        b_count = db.conn.execute(
            "SELECT COUNT(*) FROM frame_embeddings WHERE video_id = 'vid_b'"
        ).fetchone()[0]
        assert b_count == 6
