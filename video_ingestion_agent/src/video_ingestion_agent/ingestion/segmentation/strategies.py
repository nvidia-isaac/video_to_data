# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Base classes and strategies for clip refinement.

Operates on ClipContext objects with absolute timestamps and uses the
shared ModelManager for VLM inference.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from video_ingestion_agent.ingestion.config import PipelineConfig
from video_ingestion_agent.ingestion.segmentation.prompts import (
    SEGMENTATION_SYSTEM_PROMPT,
    SEGMENTATION_USER_PROMPT,
)
from video_ingestion_agent.ingestion.state import ClipContext, VerificationResult
from video_ingestion_agent.models.model_manager import BaseModel as ModelBase
from video_ingestion_agent.models.model_manager import get_model_manager
from video_ingestion_agent.utils.parsing import parse_llm_json

logger = logging.getLogger(__name__)


class RefinementStrategy(ABC):
    """
    Abstract base class for refinement strategies.

    A refinement strategy defines how to improve an invalid clip
    based on critic feedback.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._system_prompt = (
            config.segmentation.system_prompt.strip() or SEGMENTATION_SYSTEM_PROMPT
        )

    @abstractmethod
    def refine_clip(
        self,
        clip: ClipContext,
        verification: VerificationResult,
        clip_video_path: Path,
        iteration: int,
    ) -> tuple[ClipContext | None, str | None]:
        """
        Refine a single invalid clip.

        Args:
            clip: The original invalid clip
            verification: Verification result with critic feedback
            clip_video_path: Path to the extracted clip video (.mp4)
            iteration: Current refinement iteration number

        Returns:
            Tuple of (refined_clip, raw_response) or (None, None) if failed
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass


class ReannotateStrategy(RefinementStrategy):
    """
    Reannotation refinement strategy.

    Re-runs VLM on the clip video with an enhanced prompt that
    incorporates specific critic feedback about what was wrong.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self._model: ModelBase | None = None

    def _get_model(self) -> ModelBase:
        """Get VLM model from ModelManager."""
        if self._model is None:
            manager = get_model_manager()
            api_url = (
                self.config.models.vllm_url if self.config.models.vlm_backend == "vllm" else None
            )
            self._model = manager.get_model(
                model_name=self.config.models.vlm_model,
                backend=self.config.models.vlm_backend,
                device=self.config.models.device,
                fps=self.config.models.vlm_fps,
                api_key=self.config.models.api_key,
                api_url=api_url,
                use_local_media=self.config.models.vllm_local_media,
            )
        return self._model

    @property
    def name(self) -> str:
        return "reannotate"

    def refine_clip(
        self,
        clip: ClipContext,
        verification: VerificationResult,
        clip_video_path: Path,
        iteration: int,
    ) -> tuple[ClipContext | None, str | None]:
        """
        Reannotate a clip using VLM with enhanced prompt.

        The clip video file (temporary) is re-watched by the VLM, and
        the annotation is updated based on critic feedback.
        """
        if not clip_video_path.exists():
            logger.warning(f"Clip video not found: {clip_video_path}")
            return None, None

        logger.info(f"Re-annotating {clip.clip_id} with enhanced prompt")

        # Build enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(verification)

        raw_response: str | None = None
        try:
            model = self._get_model()

            raw_response = model.generate_from_video(
                video_path=str(clip_video_path),
                prompt=enhanced_prompt,
                system_prompt=self._system_prompt,
                max_new_tokens=4096,
                temperature=0.0,
            )

            # Parse response
            clips_data = parse_llm_json(raw_response, expect_array=True)

            if not clips_data:
                logger.warning(f"No refined clips generated for {clip.clip_id}")
                return None, raw_response

            # Take the first entry and update annotations
            refined_data = clips_data[0]

            # Update annotations only -- boundaries are left unchanged.
            update: dict = {
                "object": refined_data.get("object", clip.object),
                "action": refined_data.get("action", clip.action),
                "description": refined_data.get("description", clip.description),
                "metadata": {
                    **clip.metadata,
                    "refined": True,
                    "refinement_iteration": iteration,
                    "refinement_strategy": self.name,
                },
            }

            refined_clip = clip.model_copy(update=update)

            logger.info(f"  Successfully refined {clip.clip_id}")
            return refined_clip, raw_response

        except Exception as e:
            logger.error(f"Failed to refine {clip.clip_id}: {e}")
            logger.debug(f"Raw VLM response: {raw_response[:500] if raw_response else '<empty>'}")
            return None, None

    def _build_enhanced_prompt(self, verification: VerificationResult) -> str:
        """
        Build an enhanced prompt from critic feedback.

        Args:
            verification: Verification result with critic feedback

        Returns:
            Enhanced prompt string
        """
        base_prompt = self.config.segmentation.user_prompt.strip() or SEGMENTATION_USER_PROMPT

        # Extract critic feedback
        critic_response = verification.metadata.get("critic_response", {})
        issues = critic_response.get("issues", [])
        boundary_assessment = critic_response.get("boundary_assessment", {})
        annotation_assessment = critic_response.get("annotation_assessment", {})

        feedback_lines = ["", "CRITICAL FEEDBACK from previous attempt:"]

        if issues:
            feedback_lines.append(f"Issues identified: {'; '.join(issues)}")

        if not annotation_assessment.get("object_correct", True):
            feedback_lines.append("WARNING: Object identification was incorrect.")

        if not annotation_assessment.get("action_correct", True):
            feedback_lines.append("WARNING: Action description was incorrect.")
            if annotation_assessment.get("suggested_correction"):
                feedback_lines.append(f"Suggested: {annotation_assessment['suggested_correction']}")

        if not boundary_assessment.get("start_is_good", True):
            feedback_lines.append("WARNING: Start boundary needs adjustment.")

        if not boundary_assessment.get("end_is_good", True):
            feedback_lines.append("WARNING: End boundary needs adjustment.")

        if boundary_assessment.get("suggested_adjustment"):
            feedback_lines.append(
                f"Timing adjustment: {boundary_assessment['suggested_adjustment']}"
            )

        feedback_lines.append("")
        feedback_lines.append("Pay special attention to these issues when re-annotating this clip.")

        return base_prompt + "\n".join(feedback_lines)
