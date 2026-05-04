# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
COSMOS Reason Critic for segmentation verification.

Uses VLM as a critic to verify if segmentation boundaries and annotations
are correct. Operates on individual extracted (temporary) clip .mp4 files.

Uses the shared ModelManager for VLM inference.
"""

import json
import logging
from pathlib import Path

from video_ingestion_agent.ingestion.config import PipelineConfig
from video_ingestion_agent.ingestion.segmentation.prompts import (
    VERIFICATION_SYSTEM_PROMPT,
    VERIFICATION_USER_PROMPT,
)
from video_ingestion_agent.ingestion.state import ClipContext, VerificationResult
from video_ingestion_agent.models.model_manager import BaseModel as ModelBase
from video_ingestion_agent.models.model_manager import get_model_manager
from video_ingestion_agent.utils.parsing import parse_llm_json

logger = logging.getLogger(__name__)


def parse_critic_response(text: str) -> dict:
    """Parse JSON response from critic model.

    Delegates to ``common.parsing.parse_llm_json`` (object mode).
    """
    return parse_llm_json(text, expect_array=False)


class Critic:
    """
    VLM-based critic for verifying segmentation quality.

    Verifies each clip by watching its extracted .mp4 and comparing
    against the claimed annotation.

    Uses video_ingestion_agent's ModelManager for model management.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the critic.

        Args:
            config: Unified pipeline configuration
        """
        self.config = config
        self.verification_config = config.verification
        self.model_config = config.models
        self._model: ModelBase | None = None

        # Fall back to built-in defaults when the config leaves prompts empty
        self._system_prompt = (
            self.verification_config.system_prompt.strip() or VERIFICATION_SYSTEM_PROMPT
        )
        self._user_prompt = self.verification_config.user_prompt.strip() or VERIFICATION_USER_PROMPT

    def _get_model(self) -> ModelBase:
        """Get VLM model from ModelManager (lazy loaded, cached)."""
        if self._model is None:
            manager = get_model_manager()
            api_url = (
                self.model_config.vllm_url if self.model_config.vlm_backend == "vllm" else None
            )
            self._model = manager.get_model(
                model_name=self.model_config.vlm_model,
                backend=self.model_config.vlm_backend,
                device=self.model_config.device,
                fps=self.model_config.vlm_fps,
                api_key=self.model_config.api_key,
                api_url=api_url,
                use_local_media=self.model_config.vllm_local_media,
            )
        return self._model

    def verify_clip(
        self,
        clip: ClipContext,
        clip_video_path: Path,
    ) -> tuple[VerificationResult, str]:
        """
        Verify a single clip using VLM as critic.

        Args:
            clip: ClipContext with annotation to verify
            clip_video_path: Path to the extracted clip video (.mp4)

        Returns:
            Tuple of (VerificationResult, raw_response)
        """
        if not clip_video_path.exists():
            raise FileNotFoundError(f"Clip video not found: {clip_video_path}")

        # Prepare user prompt with clip metadata
        user_prompt = self._user_prompt.format(
            object=clip.object or "unknown",
            action=clip.action or "unknown",
            description=clip.description or "unknown",
        )

        # Get model and generate
        model = self._get_model()

        logger.debug(f"Verifying clip: {clip.clip_id}")
        raw_response = model.generate_from_video(
            video_path=str(clip_video_path),
            prompt=user_prompt,
            system_prompt=self._system_prompt,
            max_new_tokens=4096,
            temperature=0.0,
        )

        logger.debug(f"  Raw critic response for {clip.clip_id}:\n{raw_response}")

        # Parse critic response
        try:
            critic_result = parse_critic_response(raw_response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse critic response: {e}")
            logger.debug(f"Raw response: {raw_response}")
            critic_result = {
                "is_correct": False,
                "confidence": 0.0,
                "issues": ["Failed to parse critic response"],
                "overall_quality": "unknown",
                "recommendation": "manual_review",
            }

        # Convert to VerificationResult
        is_valid = critic_result.get("is_correct", False)
        confidence = critic_result.get("confidence", 0.5)
        issues = critic_result.get("issues", [])

        violations = []
        if not is_valid:
            violations.extend(issues)

            boundary = critic_result.get("boundary_assessment", {})
            if not boundary.get("start_is_good", True):
                violations.append("Poor start boundary")
            if not boundary.get("end_is_good", True):
                violations.append("Poor end boundary")

            annotation = critic_result.get("annotation_assessment", {})
            if not annotation.get("object_correct", True):
                violations.append("Incorrect object identification")
            if not annotation.get("action_correct", True):
                violations.append("Incorrect action labeling")

        verification = VerificationResult(
            clip_id=clip.clip_id,
            is_valid=is_valid,
            verification_score=confidence,
            violations=violations,
            metadata={
                "critic_response": critic_result,
                "overall_quality": critic_result.get("overall_quality", "unknown"),
                "recommendation": critic_result.get("recommendation", "manual_review"),
                "boundary_assessment": critic_result.get("boundary_assessment", {}),
                "annotation_assessment": critic_result.get("annotation_assessment", {}),
            },
        )

        return verification, raw_response

    def verify_clips_batch(
        self,
        clips: list[ClipContext],
        clip_path_map: dict[str, Path],
    ) -> list[tuple[VerificationResult, str]]:
        """
        Verify multiple clips in batch.

        Args:
            clips: List of ClipContext objects to verify
            clip_path_map: Mapping of clip_id to temporary .mp4 file path

        Returns:
            List of (VerificationResult, raw_response) tuples
        """
        results = []

        logger.info(f"Verifying {len(clips)} clips with VLM critic...")

        for i, clip in enumerate(clips, 1):
            logger.info(f"[{i}/{len(clips)}] Verifying {clip.clip_id}")

            clip_video = clip_path_map.get(clip.clip_id)
            if clip_video is None or not clip_video.exists():
                logger.error(f"Clip video not found for: {clip.clip_id}")
                failed_result = VerificationResult(
                    clip_id=clip.clip_id,
                    is_valid=False,
                    verification_score=0.0,
                    violations=["Clip video file not found"],
                    metadata={"error": "video_not_found"},
                )
                results.append((failed_result, ""))
                continue

            try:
                verification, response = self.verify_clip(clip, clip_video)
                results.append((verification, response))

                status = "VALID" if verification.is_valid else "INVALID"
                logger.info(f"  {status} (score: {verification.verification_score:.2f})")

                if not verification.is_valid and verification.violations:
                    logger.info(f"  Issues: {', '.join(verification.violations[:2])}")

            except Exception as e:
                logger.error(f"Failed to verify {clip.clip_id}: {e}")
                failed_result = VerificationResult(
                    clip_id=clip.clip_id,
                    is_valid=False,
                    verification_score=0.0,
                    violations=[f"Verification error: {str(e)}"],
                    metadata={"error": str(e)},
                )
                results.append((failed_result, ""))

        # Summary
        valid_count = sum(1 for r, _ in results if r.is_valid)
        avg_score = sum(r.verification_score for r, _ in results) / len(results) if results else 0

        logger.info("=" * 60)
        logger.info(f"Verification complete: {valid_count}/{len(results)} valid")
        logger.info(f"Average confidence: {avg_score:.2f}")
        logger.info("=" * 60)

        return results
