# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Tool for semantic search over frame embeddings."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from video_ingestion_agent.retrieval.tools.base import BaseTool, ToolResult
from video_ingestion_agent.utils.vector_database import VectorDatabase

logger = logging.getLogger(__name__)


@dataclass
class FrameSearchResult:
    """Frame search result for display."""

    frame_id: str
    video_id: str
    video_path: str | None
    timestamp: float
    similarity: float
    segment_id: str | None = None

    def __str__(self) -> str:
        seg = f" segment={self.segment_id}" if self.segment_id else ""
        video = f" video={self.video_path}" if self.video_path else ""
        return (
            f"Frame {self.frame_id} @ {self.timestamp:.1f}s"
            f" (similarity: {self.similarity:.3f}{seg}{video})"
        )


class SearchFramesTool(BaseTool):
    """
    Semantic search over frame embeddings.

    Uses text-to-image similarity to find frames matching
    a natural language description.
    """

    def __init__(
        self,
        vector_db_path: str,
        embedding_model: str = "google/siglip2-base-patch16-256",
        device: str = "cuda",
    ):
        """
        Initialize with vector database.

        Args:
            vector_db_path: Path to vector database
            embedding_model: Model for text embeddings (SigLIP model)
            device: Device for model inference
        """
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.device = device

        self._db: VectorDatabase | None = None
        self._processor = None
        self._model = None

    def _get_db(self) -> VectorDatabase:
        """Get vector database (lazy initialization)."""
        if self._db is None:
            self._db = VectorDatabase(self.vector_db_path)
        return self._db

    def _load_embedding_model(self):
        """Load SigLIP model for text encoding (lazy initialization)."""
        if self._model is None:
            logger.info(f"Loading SigLIP embedding model: {self.embedding_model_name}")
            try:
                self._processor = AutoProcessor.from_pretrained(self.embedding_model_name)
                self._model = AutoModel.from_pretrained(
                    self.embedding_model_name, trust_remote_code=True
                ).to(self.device)
                self._model.eval()
                logger.info("✓ SigLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SigLIP model: {e}")
                raise

    def encode_text(self, query: str) -> np.ndarray:
        """Encode text query to embedding using SigLIP."""
        self._load_embedding_model()

        # SigLIP text encoding
        inputs = self._processor(text=[query], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embedding = text_features.cpu().numpy()[0]

        return embedding

    @property
    def name(self) -> str:
        return "search_frames"

    @property
    def description(self) -> str:
        return (
            "Search for video frames using natural language description. "
            "Returns frames that visually match the query, sorted by similarity. "
            "Use this to find specific visual content in the video."
        )

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
        return {
            "query": {
                "type": "string",
                "description": "Natural language description of what to find (e.g., 'person holding a red cup')",
                "required": True,
            },
            "start_time": {
                "type": "number",
                "description": "Only search frames after this time (seconds)",
                "required": False,
            },
            "end_time": {
                "type": "number",
                "description": "Only search frames before this time (seconds)",
                "required": False,
            },
            "top_k": {
                "type": "number",
                "description": "Number of results to return (default: 10)",
                "required": False,
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum similarity threshold 0-1 (default: 0.0)",
                "required": False,
            },
            "segment_id": {
                "type": "string",
                "description": "Only search frames belonging to this segment/clip",
                "required": False,
            },
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute semantic frame search."""
        query = kwargs.get("query")

        if not query:
            return ToolResult(success=False, data=None, error="query is required")

        try:
            # Get text embedding for query using SigLIP
            query_embedding = self.encode_text(query)

            # Build time filter
            time_range = None
            if kwargs.get("start_time") is not None or kwargs.get("end_time") is not None:
                start_t = kwargs.get("start_time", 0.0)
                end_t = kwargs.get("end_time", float("inf"))
                time_range = (start_t, end_t)

            # Search
            db = self._get_db()
            top_k = int(kwargs.get("top_k", 10))
            segment_id = kwargs.get("segment_id")

            results = db.search(
                query_embedding=query_embedding,
                segment_id=segment_id,
                time_range=time_range,
                top_k=top_k,
            )

            # Filter by minimum similarity
            min_sim = kwargs.get("min_similarity", 0.0)
            results = [r for r in results if r.similarity >= min_sim]

            # Convert to display format
            output = [
                FrameSearchResult(
                    frame_id=r.frame_id,
                    video_id=r.video_id,
                    video_path=db.get_video_path(r.video_id),
                    timestamp=r.timestamp,
                    similarity=r.similarity,
                    segment_id=r.segment_id,
                )
                for r in results
            ]

            return ToolResult(success=True, data=output)

        except Exception as e:
            logger.error(f"Frame search failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    def close(self):
        """Close resources."""
        if self._db:
            self._db.close()
            self._db = None
