# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video ingestion service wrapping the LangGraph ingestion pipeline."""

import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IngestionProgress:
    """Progress information for video ingestion."""

    step: str
    step_index: int
    total_steps: int
    progress: float  # 0.0 - 1.0
    message: str
    is_complete: bool = False
    is_error: bool = False
    error_message: str | None = None

    @property
    def overall_progress(self) -> float:
        """Overall progress across all steps."""
        if self.total_steps == 0:
            return 0.0
        base_progress = self.step_index / self.total_steps
        step_contribution = self.progress / self.total_steps
        return min(base_progress + step_contribution, 1.0)


@dataclass
class IngestionResult:
    """Result of video ingestion."""

    success: bool
    graph_db_path: str | None = None
    vector_db_path: str | None = None
    video_duration: float | None = None
    segment_count: int | None = None
    entity_count: int | None = None
    relationship_count: int | None = None
    error_message: str | None = None
    elapsed_time: float = 0.0


class IngestionService:
    """Service for video ingestion using the LangGraph pipeline."""

    STEPS = [
        "Initializing",
        "Segmentation",
        "Verification",
        "Entity Extraction",
        "Frame Embeddings",
        "Entity Linking",
        "Database Writing",
        "Complete",
    ]

    def __init__(self, default_config_path: str = "configs/ingestion.yaml"):
        """Initialize ingestion service.

        Args:
            default_config_path: Default path to pipeline config YAML.
        """
        self.default_config_path = default_config_path
        self._current_progress: IngestionProgress | None = None
        self._cancel_requested = False

    def ingest_video(
        self,
        video_path: str,
        output_db: str,
        vector_db_path: str | None = None,
        config_path: str | None = None,
        progress_callback: Callable[[IngestionProgress], None] | None = None,
    ) -> IngestionResult:
        """Ingest a video through the full LangGraph pipeline.

        Args:
            video_path: Path to input video.
            output_db: Path for output entity graph database.
            vector_db_path: Path for vector database (auto-generated if None).
            config_path: Path to config file (uses default if None).
            progress_callback: Callback for progress updates.

        Returns:
            IngestionResult with success status and metadata.
        """
        start_time = time.time()
        cfg_path = config_path or self.default_config_path

        def report_progress(step: str, progress: float = 0.0, message: str = ""):
            step_index = self.STEPS.index(step) if step in self.STEPS else 0
            prog = IngestionProgress(
                step=step,
                step_index=step_index,
                total_steps=len(self.STEPS),
                progress=progress,
                message=message,
            )
            self._current_progress = prog
            if progress_callback:
                progress_callback(prog)

        try:
            report_progress("Initializing", 0.0, "Loading configuration...")

            # Validate paths
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            # Create output directory
            output_dir = Path(output_db).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Auto-generate vector db path if not provided
            if vector_db_path is None:
                db_stem = Path(output_db).stem
                vector_db_path = str(output_dir / f"{db_stem}_vector.db")

            # Load pipeline config
            from ...ingestion.config import PipelineConfig, load_config

            if Path(cfg_path).exists():
                config = load_config(cfg_path)
            else:
                config = PipelineConfig()

            report_progress("Initializing", 0.5, "Starting pipeline...")
            logger.info(f"Starting ingestion of {video_path}")

            # Use a per-video run directory alongside the DB
            run_dir = output_dir / "runs" / Path(video_path).stem
            run_dir.mkdir(parents=True, exist_ok=True)

            # Run the pipeline in a background thread
            result_holder: dict = {"state": None, "error": None}

            def _run():
                try:
                    from ...ingestion.ingestion_graph import run_pipeline

                    result_holder["state"] = run_pipeline(
                        video_path=video_path,
                        run_dir=run_dir,
                        config=config,
                        graph_db_path=output_db,
                        vector_db_path=vector_db_path,
                    )
                except Exception as e:
                    result_holder["error"] = str(e)

            pipeline_thread = threading.Thread(target=_run)
            pipeline_thread.start()

            # Simulate progress while pipeline runs
            step_durations = {
                "Segmentation": 60,
                "Verification": 30,
                "Entity Extraction": 30,
                "Frame Embeddings": 45,
                "Entity Linking": 5,
                "Database Writing": 5,
            }

            current_step_idx = 1  # Start after "Initializing"
            step_start = time.time()

            while pipeline_thread.is_alive():
                if self._cancel_requested:
                    logger.warning("Ingestion cancelled by user")
                    break

                current_step = self.STEPS[current_step_idx]
                expected_duration = step_durations.get(current_step, 10)
                elapsed = time.time() - step_start
                step_progress = min(elapsed / expected_duration, 0.95)

                report_progress(
                    current_step,
                    step_progress,
                    f"Processing... ({elapsed:.0f}s)",
                )

                if elapsed > expected_duration and current_step_idx < len(self.STEPS) - 2:
                    current_step_idx += 1
                    step_start = time.time()

                time.sleep(1)

            pipeline_thread.join()

            if result_holder["error"]:
                raise RuntimeError(result_holder["error"])

            report_progress("Complete", 1.0, "Ingestion complete!")

            # Get statistics from the database
            segment_count = 0
            entity_count = 0
            relationship_count = 0
            video_duration = 0.0

            try:
                with sqlite3.connect(output_db) as conn:
                    segment_count = conn.execute("SELECT COUNT(*) FROM action_segments").fetchone()[
                        0
                    ]
                    entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                    relationship_count = conn.execute(
                        "SELECT COUNT(*) FROM relationships"
                    ).fetchone()[0]
                    row = conn.execute("SELECT duration FROM video_metadata LIMIT 1").fetchone()
                    video_duration = (row[0] if row else 0) or 0
            except Exception as e:
                logger.warning(f"Failed to get stats: {e}")

            elapsed = time.time() - start_time
            logger.info(
                f"Ingestion complete in {elapsed:.1f}s: "
                f"{segment_count} segments, {entity_count} entities"
            )

            return IngestionResult(
                success=True,
                graph_db_path=output_db,
                vector_db_path=vector_db_path,
                video_duration=video_duration,
                segment_count=segment_count,
                entity_count=entity_count,
                relationship_count=relationship_count,
                elapsed_time=elapsed,
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            elapsed = time.time() - start_time

            error_progress = IngestionProgress(
                step="Error",
                step_index=0,
                total_steps=len(self.STEPS),
                progress=0,
                message=str(e),
                is_complete=True,
                is_error=True,
                error_message=str(e),
            )
            if progress_callback:
                progress_callback(error_progress)

            return IngestionResult(
                success=False,
                error_message=str(e),
                elapsed_time=elapsed,
            )

    def ingest_video_streaming(
        self,
        video_path: str,
        output_db: str,
        vector_db_path: str | None = None,
        config_path: str | None = None,
    ) -> Generator[IngestionProgress, None, IngestionResult]:
        """Ingest video with streaming progress updates.

        Yields IngestionProgress objects and returns IngestionResult.

        Usage:
            gen = service.ingest_video_streaming(video, db)
            for progress in gen:
                print(f"{progress.step}: {progress.progress*100:.0f}%")
            result = gen.value  # Final result after iteration
        """
        result_holder: dict = {"result": None}

        def run_ingestion():
            result_holder["result"] = self.ingest_video(
                video_path=video_path,
                output_db=output_db,
                vector_db_path=vector_db_path,
                config_path=config_path,
                progress_callback=lambda _p: None,
            )

        thread = threading.Thread(target=run_ingestion)
        thread.start()

        while thread.is_alive():
            if self._current_progress:
                yield self._current_progress
            time.sleep(0.5)

        thread.join()

        if self._current_progress:
            yield self._current_progress

        return result_holder["result"]

    def cancel(self):
        """Request cancellation of current ingestion."""
        self._cancel_requested = True

    def reset(self):
        """Reset service state."""
        self._cancel_requested = False
        self._current_progress = None
