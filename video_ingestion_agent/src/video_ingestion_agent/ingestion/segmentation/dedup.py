# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Clip deduplication for overlapping segments.

Provides :class:`ClipDeduplicator` which supports two merge strategies
selectable via the ``method`` parameter:

* ``"heuristic"`` -- always merges overlapping pairs, keeping the longer
  clip's annotations.
* ``"llm"`` -- asks a language model whether two overlapping clips
  describe the same action and synthesises combined annotations when they
  do.  Requires a *model* to be supplied.
"""

from __future__ import annotations

import logging
from typing import Literal

from video_ingestion_agent.ingestion.segmentation.prompts import (
    DEDUP_MERGE_SYSTEM_PROMPT,
    DEDUP_MERGE_USER_PROMPT,
)
from video_ingestion_agent.ingestion.state import ClipContext
from video_ingestion_agent.models.model_manager import BaseModel as ModelBase
from video_ingestion_agent.utils.parsing import parse_llm_json as _parse_llm_json

logger = logging.getLogger(__name__)


class ClipDeduplicator:
    """Merge overlapping clip segments.

    Args:
        overlap_threshold: Minimum overlap in seconds to trigger a merge.
            Positive values require actual overlap; negative values also
            consider clips separated by a small gap (e.g. ``-0.1`` merges
            clips up to 0.1 s apart).  ``None`` disables merging entirely.
        method: ``"heuristic"`` or ``"llm"``.
        model: LLM used for merge decisions when *method* is ``"llm"``.
    """

    def __init__(
        self,
        overlap_threshold: float | None,
        method: Literal["heuristic", "llm"] = "heuristic",
        model: ModelBase | None = None,
    ) -> None:
        if method == "llm" and model is None:
            raise ValueError("method='llm' requires a model to be provided")
        self.overlap_threshold = overlap_threshold
        self.method = method
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, clips: list[ClipContext]) -> list[ClipContext]:
        """Deduplicate *clips* and return the merged result sorted by ``start_t``."""
        if self.overlap_threshold is None or len(clips) <= 1:
            return clips

        sorted_clips = sorted(clips, key=lambda c: c.start_t)

        if self.method == "llm":
            result = self._llm_pass(sorted_clips)
        else:
            result = self._heuristic_pass(sorted_clips)

        if len(result) < len(clips):
            tag = "LLM merge" if self.method == "llm" else "Overlap merge"
            logger.info(
                f"{tag} dedup: {len(clips)} -> {len(result)} clips "
                f"(merged {len(clips) - len(result)}, "
                f"threshold={self.overlap_threshold}s)"
            )

        return result

    # ------------------------------------------------------------------
    # Heuristic merge (no LLM)
    # ------------------------------------------------------------------

    def _heuristic_pass(self, sorted_clips: list[ClipContext]) -> list[ClipContext]:
        """Iteratively merge until no overlaps above threshold remain."""
        assert self.overlap_threshold is not None
        changed = True
        while changed:
            changed = False
            result: list[ClipContext] = [sorted_clips[0]]
            for clip in sorted_clips[1:]:
                prev = result[-1]
                overlap = prev.end_t - clip.start_t
                if overlap > self.overlap_threshold:
                    result[-1] = self._merge_heuristic(prev, clip)
                    changed = True
                else:
                    result.append(clip)
            sorted_clips = result
        return result

    # ------------------------------------------------------------------
    # LLM-based merge
    # ------------------------------------------------------------------

    def _llm_pass(self, sorted_clips: list[ClipContext]) -> list[ClipContext]:
        """Single-pass merge asking the LLM for each overlapping pair."""
        assert self.overlap_threshold is not None
        result: list[ClipContext] = [sorted_clips[0]]
        for clip in sorted_clips[1:]:
            prev = result[-1]
            overlap = prev.end_t - clip.start_t
            if overlap <= self.overlap_threshold:
                result.append(clip)
                continue

            should_merge, merged_clip = self._ask_llm(prev, clip, overlap)
            if should_merge:
                result[-1] = merged_clip
            else:
                result.append(clip)

        return result

    def _ask_llm(
        self,
        prev: ClipContext,
        clip: ClipContext,
        overlap: float,
    ) -> tuple[bool, ClipContext]:
        """Ask the LLM whether *prev* and *clip* should be merged.

        Returns ``(should_merge, merged_clip)``.  When ``should_merge`` is
        ``False`` the second element should be ignored by the caller.
        """
        assert self.model is not None

        prompt = DEDUP_MERGE_USER_PROMPT.format(
            a_start_t=prev.start_t,
            a_end_t=prev.end_t,
            a_object=prev.object,
            a_action=prev.action,
            a_description=prev.description,
            b_start_t=clip.start_t,
            b_end_t=clip.end_t,
            b_object=clip.object,
            b_action=clip.action,
            b_description=clip.description,
            overlap=overlap,
        )

        conversation = [
            {"role": "system", "content": DEDUP_MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        fallback = self._merge_heuristic(prev, clip)

        try:
            raw = self.model.generate_text(conversation, max_new_tokens=512, temperature=0.0)
            data = _parse_llm_json(raw)
        except Exception:
            logger.warning(
                f"LLM merge failed for {prev.clip_id} + {clip.clip_id}, "
                "keeping clips separate (conservative fallback)"
            )
            return False, fallback

        if not isinstance(data, dict):
            logger.warning(
                "LLM merge returned non-dict, keeping clips separate (conservative fallback)"
            )
            return False, fallback

        should_merge = bool(data.get("merge", False))
        if not should_merge:
            return False, fallback

        merged_start = min(prev.start_t, clip.start_t)
        merged_end = max(prev.end_t, clip.end_t)

        donor = prev if (prev.end_t - prev.start_t) >= (clip.end_t - clip.start_t) else clip

        merged = donor.model_copy(
            update={
                "start_t": merged_start,
                "end_t": merged_end,
                "object": data.get("object", donor.object),
                "action": data.get("action", donor.action),
                "description": data.get("description", donor.description),
            }
        )

        logger.info(
            f"LLM merged {prev.clip_id} + {clip.clip_id} -> "
            f"[{merged_start:.1f}s-{merged_end:.1f}s] {merged.action}"
        )
        return True, merged

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_heuristic(prev: ClipContext, clip: ClipContext) -> ClipContext:
        """Union time range, keep the longer clip's annotations."""
        donor = prev if (prev.end_t - prev.start_t) >= (clip.end_t - clip.start_t) else clip
        return donor.model_copy(
            update={
                "start_t": min(prev.start_t, clip.start_t),
                "end_t": max(prev.end_t, clip.end_t),
            }
        )
