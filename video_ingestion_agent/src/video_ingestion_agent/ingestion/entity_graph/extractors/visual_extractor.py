# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual feature extractor using VLM and SigLIP-2."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from video_ingestion_agent.models.model_manager import get_model_manager
from video_ingestion_agent.utils.video_processor import Frame

logger = logging.getLogger(__name__)


@dataclass
class Caption:
    """Generated caption for a video chunk."""

    text: str
    start_t: float
    end_t: float
    chunk_frames: list[str]  # Frame IDs in this chunk
    location: str | None = None  # Detected scene location


class VisualExtractor:
    """
    Extract visual features from video frames.

    Responsibilities:
    1. Generate captions using VLM (30s chunks)
    2. Extract frame embeddings using SigLIP-2
    3. Detect scene locations from captions

    Uses ModelManager for VLM (supports local and API backends).
    """

    # Default VLM prompt for video captioning
    DEFAULT_VLM_PROMPT = (
        "Describe what is happening in this video sequence. "
        "Focus on:\n"
        "1. What objects are visible and being manipulated\n"
        "2. What actions the person is performing\n"
        "3. The sequence of events\n"
        "4. The location/setting if identifiable\n\n"
        "Be specific about object names and actions."
    )

    def __init__(
        self,
        vlm_model: str = "nvidia/Cosmos-Reason2-8B",
        embedding_model: str = "google/siglip2-base-patch16-256",
        device: str = "cuda",
        chunk_size: float = 30.0,
        chunk_overlap: float = 5.0,
        vlm_backend: str = "local",
        api_key: str | None = None,
        vlm_prompt: str | None = None,
    ):
        """
        Initialize visual extractors.

        Args:
            vlm_model: VLM model name for captioning
            embedding_model: SigLIP model name for embeddings
            device: Device to run models on
            chunk_size: Size of video chunks for captioning (seconds)
            chunk_overlap: Overlap between chunks (seconds)
            vlm_backend: "local" or "api" for VLM
            api_key: API key if using API backend
            vlm_prompt: Custom prompt for VLM captioning (uses default if None)
        """
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vlm_model_name = vlm_model
        self.vlm_backend = vlm_backend
        self.api_key = api_key
        self.vlm_prompt = vlm_prompt.strip() if vlm_prompt else self.DEFAULT_VLM_PROMPT

        # SigLIP for embeddings (always local)
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            self.embedding_processor = AutoProcessor.from_pretrained(embedding_model)
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model, trust_remote_code=True
            ).to(device)
            self.embedding_model.eval()
            logger.info("✓ SigLIP-2 loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SigLIP model: {e}")
            raise

        # VLM for captioning - lazy load via ModelManager
        self._vlm_model = None

        # Location keywords for scene detection
        self.location_keywords = {
            "kitchen",
            "bedroom",
            "living room",
            "bathroom",
            "garage",
            "office",
            "dining room",
            "hallway",
            "outdoor",
            "workshop",
            "lab",
            "warehouse",
            "store",
            "restaurant",
        }

    def _get_vlm_model(self):
        """Get VLM from ModelManager (lazy loading with caching)."""
        if self._vlm_model is None:
            manager = get_model_manager()
            self._vlm_model = manager.get_model(
                model_name=self.vlm_model_name,
                backend=self.vlm_backend,
                device=self.device,
                fps=1,  # 1 FPS for frame-based captioning
                api_key=self.api_key,
            )
        return self._vlm_model

    def extract_captions(self, frames: list[Frame], video_duration: float) -> list[Caption]:
        """
        Generate captions for video chunks using VLM.

        Args:
            frames: List of frames (assumed to be at 1 FPS)
            video_duration: Total video duration in seconds

        Returns:
            List of captions with timestamps
        """
        captions = []
        step = int(self.chunk_size - self.chunk_overlap)  # Step in seconds

        logger.info(f"Generating captions for {len(frames)} frames")
        logger.info(f"Chunk size: {self.chunk_size}s, overlap: {self.chunk_overlap}s")

        # Process video in chunks
        for chunk_start in range(0, int(video_duration), step):
            chunk_end = min(chunk_start + self.chunk_size, video_duration)

            # Get frames in this chunk
            chunk_frames = [f for f in frames if chunk_start <= f.timestamp < chunk_end]

            if not chunk_frames:
                continue

            # Generate caption for this chunk
            caption_text = self._generate_caption_for_chunk(chunk_frames)

            # Detect location
            location = self._detect_location(caption_text)

            caption = Caption(
                text=caption_text,
                start_t=float(chunk_start),
                end_t=float(chunk_end),
                chunk_frames=[f.frame_id for f in chunk_frames],
                location=location,
            )
            captions.append(caption)

            logger.info(
                f"Caption {len(captions)}: [{chunk_start:.1f}s - {chunk_end:.1f}s] "
                f"{len(chunk_frames)} frames, location={location}"
            )

        logger.info(f"Generated {len(captions)} captions")
        return captions

    def _generate_caption_for_chunk(self, frames: list[Frame]) -> str:
        """
        Generate caption for a chunk of frames using VLM.

        Args:
            frames: Frames in this chunk (sampled at 1 FPS)

        Returns:
            Generated caption text
        """
        # Sample frames uniformly (max 8-10 frames per chunk)
        max_frames = 8
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames

        # Convert frames to PIL Images
        images = [Image.fromarray(f.image) for f in sampled_frames]

        try:
            # Use ModelManager to get the model
            vlm_model = self._get_vlm_model()

            # Use generate_from_frames if available (local model)
            if hasattr(vlm_model, "_model") and hasattr(vlm_model._model, "generate_from_frames"):
                caption = vlm_model._model.generate_from_frames(
                    frames=images,
                    prompt=self.vlm_prompt,
                    max_new_tokens=512,
                    temperature=0.0,
                )
            else:
                # Fallback: Use generate_text with image content
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": img} for img in images],
                            {"type": "text", "text": self.vlm_prompt},
                        ],
                    }
                ]
                caption = vlm_model.generate_text(
                    conversation=conversation,
                    max_new_tokens=512,
                    temperature=0.0,
                )

            return caption

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            # Fallback caption
            return f"Video segment from {frames[0].timestamp:.1f}s to {frames[-1].timestamp:.1f}s"

    def _detect_location(self, caption: str) -> str | None:
        """
        Detect scene location from caption text.

        Args:
            caption: Generated caption

        Returns:
            Detected location or None
        """
        caption_lower = caption.lower()

        for location in self.location_keywords:
            if location in caption_lower:
                return location

        return None

    def extract_embeddings(
        self, frames: list[Frame], batch_size: int = 32
    ) -> list[tuple[str, np.ndarray]]:
        """
        Extract SigLIP-2 embeddings for frames.

        Args:
            frames: List of frames
            batch_size: Batch size for processing

        Returns:
            List of (frame_id, embedding) tuples
        """
        logger.info(f"Extracting embeddings for {len(frames)} frames")

        embeddings = []

        # Process in batches
        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start : batch_start + batch_size]

            # Convert to PIL Images
            images = [Image.fromarray(f.image) for f in batch_frames]

            try:
                # Process images
                inputs = self.embedding_processor(images=images, return_tensors="pt").to(
                    self.device
                )

                # Extract embeddings
                with torch.no_grad():
                    outputs = self.embedding_model.get_image_features(**inputs)
                    # Normalize embeddings
                    batch_embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
                    batch_embeddings = batch_embeddings.cpu().numpy()

                # Store with frame IDs
                for i, frame in enumerate(batch_frames):
                    embeddings.append((frame.frame_id, batch_embeddings[i]))

                logger.info(
                    f"Processed batch {batch_start // batch_size + 1}: {len(batch_frames)} frames"
                )

            except Exception as e:
                logger.error(f"Embedding extraction failed for batch: {e}")
                # Add zero embeddings as fallback
                embedding_dim = 768  # SigLIP2 base-patch16-256 dimension
                for frame in batch_frames:
                    embeddings.append((frame.frame_id, np.zeros(embedding_dim)))

        logger.info(f"Extracted {len(embeddings)} embeddings")
        return embeddings

    def extract_single_embedding(self, frame: Frame) -> np.ndarray:
        """
        Extract embedding for a single frame.

        Args:
            frame: Single frame

        Returns:
            Embedding vector
        """
        image = Image.fromarray(frame.image)

        inputs = self.embedding_processor(images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model.get_image_features(**inputs)
            embedding = outputs / outputs.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy()[0]

        return embedding
