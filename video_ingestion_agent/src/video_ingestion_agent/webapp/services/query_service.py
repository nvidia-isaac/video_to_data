# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query service wrapping RetrievalAgent."""

import json
import logging
import sqlite3
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from video_ingestion_agent.retrieval.config import RetrievalConfig
from video_ingestion_agent.webapp.models.query_history import ClipResult, QueryRecord, SubTaskResult

logger = logging.getLogger(__name__)


@dataclass
class NodeUpdate:
    """Update from a LangGraph node execution."""

    node_name: str
    status: str  # "started", "completed", "error"
    output: dict[str, Any] | None = None
    iteration: int = 0
    message: str = ""


@dataclass
class QueryResult:
    """Result of a video query."""

    success: bool
    query: str
    final_answer: str = ""
    clips: list[dict[str, Any]] = field(default_factory=list)
    clips_extracted: list[str] = field(default_factory=list)
    sub_tasks: list[dict[str, Any]] = field(default_factory=list)
    working_memory: list[str] = field(default_factory=list)
    task_results: dict[int, dict] = field(default_factory=dict)
    elapsed_time: float = 0.0
    error_message: str | None = None

    def to_query_record(self, project_id: int | None = None) -> QueryRecord:
        """Convert to QueryRecord for storage."""
        clips = [
            ClipResult(
                clip_id=f"clip_{i}",
                video_path=c.get("video_path", ""),
                start_time=c.get("start_time", 0),
                end_time=c.get("end_time", 0),
                description=c.get("description", ""),
                confidence=c.get("confidence", 0),
                action=c.get("action"),
                object_name=c.get("object"),
                extracted_path=self.clips_extracted[i] if i < len(self.clips_extracted) else None,
            )
            for i, c in enumerate(self.clips)
        ]

        sub_tasks = [
            SubTaskResult(
                task_id=t.get("task_id", i),
                description=t.get("description", ""),
                search_type=t.get("search_type", ""),
                target_action=t.get("target_action"),
                target_object=t.get("target_object"),
                clips_found=len(self.task_results.get(t.get("task_id", i), {}).get("clips", [])),
                analysis=self.task_results.get(t.get("task_id", i), {}).get("analysis", ""),
            )
            for i, t in enumerate(self.sub_tasks)
        ]

        return QueryRecord(
            id=0,  # Will be set by database
            query=self.query,
            project_id=project_id,
            timestamp=datetime.now(),
            duration_seconds=self.elapsed_time,
            clips=clips,
            sub_tasks=sub_tasks,
            final_answer=self.final_answer,
            working_memory=self.working_memory,
        )


class QueryService:
    """Service for querying videos using RetrievalAgent."""

    def __init__(
        self,
        graph_db_path: str,
        vector_db_path: str | None = None,
        clips_dir: str = "outputs/clips",
        config: RetrievalConfig | None = None,
    ):
        """Initialize query service.

        Args:
            graph_db_path: Path to entity graph database.
            vector_db_path: Path to vector database (optional).
            clips_dir: Directory for extracted clips.
            config: Retrieval agent configuration.
        """
        self.graph_db_path = graph_db_path
        self.vector_db_path = vector_db_path or self._auto_detect_vector_db()
        self.clips_dir = clips_dir
        self.config = config or RetrievalConfig()

        self._agent = None
        self._tools = None
        self._video_paths = None

    def _auto_detect_vector_db(self) -> str | None:
        """Auto-detect vector database path."""
        if not self.graph_db_path:
            return None

        # Try common naming patterns
        db_path = Path(self.graph_db_path)
        candidates = [
            db_path.parent / f"{db_path.stem}_vector.db",
            db_path.parent / "vector.db",
            db_path.parent / f"{db_path.stem.replace('_graph', '')}_vector.db",
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return None

    def _get_video_paths(self) -> dict[int, str]:
        """Get video paths from database."""
        if self._video_paths is None:
            self._video_paths = {}
            if Path(self.graph_db_path).exists():
                try:
                    with sqlite3.connect(self.graph_db_path) as conn:
                        rows = conn.execute("SELECT id, video_path FROM video_metadata").fetchall()
                        self._video_paths = {row[0]: row[1] for row in rows}
                except Exception as e:
                    logger.warning(f"Failed to get video paths: {e}")
        return self._video_paths

    def _init_tools(self) -> dict[str, Any]:
        """Initialize search and extraction tools."""
        if self._tools is not None:
            return self._tools

        self._tools = {}

        try:
            from ...retrieval.tools.search_graph import SearchGraphTool

            self._tools["search_graph"] = SearchGraphTool(self.graph_db_path)
            logger.info("Initialized SearchGraphTool")
        except Exception as e:
            logger.error(f"Failed to init SearchGraphTool: {e}")

        if self.vector_db_path and Path(self.vector_db_path).exists():
            try:
                from ...retrieval.tools.search_frames import SearchFramesTool

                self._tools["search_frames"] = SearchFramesTool(
                    vector_db_path=self.vector_db_path,
                    embedding_model=self.config.models.embedding_model,
                    device=self.config.models.device,
                )
                logger.info("Initialized SearchFramesTool")
            except Exception as e:
                logger.warning(f"Failed to init SearchFramesTool: {e}")

        try:
            from ...retrieval.tools.extract_clip import ExtractClipTool

            video_paths = self._get_video_paths()
            Path(self.clips_dir).mkdir(parents=True, exist_ok=True)
            self._tools["extract_clip"] = ExtractClipTool(
                video_paths=video_paths,
                output_dir=self.clips_dir,
            )
            logger.info("Initialized ExtractClipTool")
        except Exception as e:
            logger.error(f"Failed to init ExtractClipTool: {e}")

        return self._tools

    def _get_agent(self):
        """Get or create RetrievalAgent."""
        if self._agent is not None:
            return self._agent

        try:
            from ...retrieval import RetrievalAgent

            tools = self._init_tools()

            logger.info(
                f"Creating RetrievalAgent with model={self.config.models.llm_model}, "
                f"backend={self.config.models.llm_backend}"
            )

            self._agent = RetrievalAgent(
                config=self.config,
                tools=tools,
            )

            logger.info("Initialized RetrievalAgent")
            return self._agent

        except Exception as e:
            logger.error(f"Failed to create RetrievalAgent: {e}")
            raise

    def run_query(
        self,
        query: str,
        video_path: str | None = None,
        node_callback: Callable[[NodeUpdate], None] | None = None,
    ) -> QueryResult:
        """Run a query using the LangGraph agent.

        Args:
            query: Natural language query.
            video_path: Optional specific video path.
            node_callback: Callback for node execution updates.

        Returns:
            QueryResult with clips and answer.
        """
        start_time = time.time()

        # Use first video if not specified
        if video_path is None:
            video_paths = self._get_video_paths()
            video_path = next(iter(video_paths.values()), "") if video_paths else ""

        try:
            agent = self._get_agent()

            # Report node start
            if node_callback:
                node_callback(
                    NodeUpdate(
                        node_name="task_decomposer",
                        status="started",
                        message="Decomposing query into sub-tasks...",
                    )
                )

            # Run the agent
            result = agent.run(query=query, video_path=video_path)

            elapsed = time.time() - start_time

            if not result.get("success", False):
                return QueryResult(
                    success=False,
                    query=query,
                    error_message=result.get("error", "Unknown error"),
                    elapsed_time=elapsed,
                )

            # Report completion
            if node_callback:
                node_callback(
                    NodeUpdate(
                        node_name="vqa_synthesizer",
                        status="completed",
                        message="Query complete!",
                    )
                )

            return QueryResult(
                success=True,
                query=query,
                final_answer=result.get("answer", ""),
                clips=result.get("clips_to_extract", []),
                clips_extracted=result.get("clips_extracted", []),
                sub_tasks=result.get("sub_tasks", []),
                working_memory=result.get("working_memory", []),
                task_results=result.get("task_results", {}),
                elapsed_time=elapsed,
            )

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return QueryResult(
                success=False,
                query=query,
                error_message=str(e),
                elapsed_time=time.time() - start_time,
            )

    def run_query_streaming(
        self,
        query: str,
        video_path: str | None = None,
    ) -> Generator[NodeUpdate | QueryResult, None, None]:
        """Run query with streaming node updates.

        Yields NodeUpdate objects during execution, then final QueryResult.

        Usage:
            for update in service.run_query_streaming(query):
                if isinstance(update, QueryResult):
                    # Final result
                    result = update
                else:
                    # Progress update
                    print(f"{update.node_name}: {update.status}")
        """
        start_time = time.time()

        # Use first video if not specified
        if video_path is None:
            video_paths = self._get_video_paths()
            video_path = next(iter(video_paths.values()), "") if video_paths else ""

        try:
            agent = self._get_agent()

            # Use agent's streaming interface
            final_result = None
            accumulated_memory = []

            for update in agent.run_streaming(query=query, video_path=video_path):
                node_name = update.get("node_name", "unknown")
                status = update.get("status", "")

                # Accumulate working memory
                new_memory = update.get("working_memory", [])
                if new_memory:
                    accumulated_memory.extend(new_memory)

                # Check if this is the final result
                if node_name == "__final__":
                    elapsed = time.time() - start_time
                    final_result = QueryResult(
                        success=True,
                        query=query,
                        final_answer=update.get("answer", ""),
                        clips=update.get("clips_to_extract", []),
                        clips_extracted=update.get("clips_extracted", []),
                        sub_tasks=update.get("sub_tasks", []),
                        working_memory=update.get("working_memory", []),
                        task_results=update.get("task_results", {}),
                        elapsed_time=elapsed,
                    )
                elif node_name == "__error__":
                    elapsed = time.time() - start_time
                    final_result = QueryResult(
                        success=False,
                        query=query,
                        error_message=update.get("error", "Unknown error"),
                        elapsed_time=elapsed,
                    )
                else:
                    # Yield progress update
                    task_idx = update.get("current_task_idx")
                    message = f"Task {(task_idx or 0) + 1}"
                    if update.get("search_type"):
                        message += f" | {update['search_type']}"
                    if update.get("relaxation_level") is not None:
                        message += f" (level {update['relaxation_level']})"

                    yield NodeUpdate(
                        node_name=node_name,
                        status=status,
                        message=message,
                        output={"working_memory": accumulated_memory.copy()},
                    )

            # Yield final result
            if final_result:
                yield final_result
            else:
                yield QueryResult(
                    success=False,
                    query=query,
                    error_message="No result from agent",
                    elapsed_time=time.time() - start_time,
                )

        except Exception as e:
            logger.error(f"Streaming query failed: {e}", exc_info=True)
            yield QueryResult(
                success=False,
                query=query,
                error_message=str(e),
                elapsed_time=time.time() - start_time,
            )

    def get_available_videos(self) -> list[dict[str, Any]]:
        """Get list of videos available for querying."""
        videos = []
        video_paths = self._get_video_paths()

        for video_id, path in video_paths.items():
            video_info = {
                "id": video_id,
                "path": path,
                "name": Path(path).name,
                "exists": Path(path).exists(),
            }
            videos.append(video_info)

        return videos


class HistoryService:
    """Service for managing query history."""

    def __init__(self, db_path: str):
        """Initialize history service.

        Args:
            db_path: Path to history SQLite database.
        """
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Create database and tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    project_id INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    duration_seconds REAL,
                    final_answer TEXT,
                    working_memory TEXT,
                    config_json TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER NOT NULL,
                    clip_id TEXT,
                    video_path TEXT,
                    start_time REAL,
                    end_time REAL,
                    description TEXT,
                    confidence REAL,
                    action TEXT,
                    object_name TEXT,
                    extracted_path TEXT,
                    FOREIGN KEY(query_id) REFERENCES query_history(id) ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_history_project
                ON query_history(project_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_history_timestamp
                ON query_history(timestamp)
            """)

            conn.commit()

    def save_query(self, record: QueryRecord) -> int:
        """Save query record to history.

        Returns:
            ID of saved record.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO query_history
                (query, project_id, timestamp, duration_seconds, final_answer, working_memory, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.query,
                    record.project_id,
                    record.timestamp.isoformat(),
                    record.duration_seconds,
                    record.final_answer,
                    json.dumps(record.working_memory),
                    json.dumps(record.config_used),
                ),
            )
            query_id = cursor.lastrowid

            # Save clips
            for clip in record.clips:
                conn.execute(
                    """
                    INSERT INTO query_results
                    (query_id, clip_id, video_path, start_time, end_time, description,
                     confidence, action, object_name, extracted_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        clip.clip_id,
                        clip.video_path,
                        clip.start_time,
                        clip.end_time,
                        clip.description,
                        clip.confidence,
                        clip.action,
                        clip.object_name,
                        clip.extracted_path,
                    ),
                )

            conn.commit()

        logger.info(f"Saved query to history with ID {query_id}")
        return query_id

    def get_history(
        self,
        project_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[QueryRecord]:
        """Get query history."""
        where_clause = "WHERE project_id = ?" if project_id else ""
        values = [project_id] if project_id else []

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM query_history
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                values + [limit, offset],
            ).fetchall()

            records = []
            for row in rows:
                record = QueryRecord.from_db_row(row)

                # Load clips
                clip_rows = conn.execute(
                    "SELECT * FROM query_results WHERE query_id = ?",
                    (record.id,),
                ).fetchall()

                record.clips = [
                    ClipResult(
                        clip_id=r[2] or "",
                        video_path=r[3] or "",
                        start_time=r[4] or 0,
                        end_time=r[5] or 0,
                        description=r[6] or "",
                        confidence=r[7] or 0,
                        action=r[8],
                        object_name=r[9],
                        extracted_path=r[10],
                    )
                    for r in clip_rows
                ]

                records.append(record)

            return records

    def delete_query(self, query_id: int) -> bool:
        """Delete query from history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM query_results WHERE query_id = ?", (query_id,))
            conn.execute("DELETE FROM query_history WHERE id = ?", (query_id,))
            conn.commit()

        return True
