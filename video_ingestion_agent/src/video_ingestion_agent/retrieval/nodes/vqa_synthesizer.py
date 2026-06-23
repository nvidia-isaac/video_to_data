# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VQA Synthesizer node.

Synthesizes final answer and selects clips for robot policy training.
"""

import logging
from pathlib import Path
from typing import Any

from video_ingestion_agent.retrieval.nodes.base import BaseNode
from video_ingestion_agent.retrieval.nodes.prompts import (
    VQA_SYNTHESIZER_SYSTEM,
    VQA_SYNTHESIZER_USER,
)
from video_ingestion_agent.retrieval.state import AgentState

logger = logging.getLogger(__name__)


class VQASynthesizerNode(BaseNode):
    """Synthesize final answer and select clips for training.

    This is the final node that:
    1. Analyzes all collected search results
    2. Recommends the best clips for robot policy training
    3. Extracts the recommended clips using the extract_clip tool
    4. Generates a comprehensive answer
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Synthesize final answer and extract clips.

        Args:
            state: Current agent state with all search results

        Returns:
            State updates with final_answer and clips_extracted
        """
        logger.info("=== VQA SYNTHESIZER ===")

        task_results = state.get("task_results", {})

        # Format task results and build clip index for ID-based selection
        task_results_str, clip_index = self._format_task_results(task_results)
        logger.info(f"Built clip index with {len(clip_index)} entries")

        user_prompt = VQA_SYNTHESIZER_USER.format(
            query=state["query"],
            task_results=task_results_str,
        )

        response = self._call_llm(
            user_prompt, system_prompt=VQA_SYNTHESIZER_SYSTEM, max_tokens=4096
        )
        synthesis = self._parse_json(response)

        task_summary = synthesis.get("task_summary", "")
        raw_selections = synthesis.get("recommended_clips", [])
        training_notes = synthesis.get("training_notes", "")
        missing = synthesis.get("missing_demonstrations", [])

        logger.info(f"Task summary: {task_summary}")
        logger.info(f"LLM selected {len(raw_selections)} clips")
        if missing:
            logger.warning(f"Missing demonstrations: {missing}")

        # Resolve clip IDs to full metadata from clip_index
        recommended_clips = self._resolve_clips(raw_selections, clip_index)

        # Extract clips if tool available
        default_video_path = state.get("video_path", "")
        extracted_clips, extracted_clip_infos = self._extract_clips(
            recommended_clips, default_video_path
        )

        # Build final answer using only successfully extracted clips
        final_answer = self._build_answer(
            task_summary=task_summary,
            recommended_clips=extracted_clip_infos if extracted_clip_infos else recommended_clips,
            missing=missing,
            training_notes=training_notes,
            default_video_path=default_video_path,
        )

        return {
            "final_answer": final_answer,
            "clips_to_extract": extracted_clip_infos,
            "clips_extracted": extracted_clips,
        }

    def _format_task_results(self, task_results: dict) -> tuple[str, dict[str, dict]]:
        """Format task results and build a clip index for ID-based selection.

        Each clip is assigned a stable ID like ``T1-C1`` so the LLM can
        reference clips by ID instead of reproducing numeric metadata.

        Returns:
            (formatted_string, clip_index) where clip_index maps clip ID
            to the full clip dict (start_time, end_time, video_path, ...).
        """
        clip_index: dict[str, dict] = {}
        task_results_str = ""

        for task_id, result in task_results.items():
            task = result.get("task", {})
            clips = result.get("clips", [])
            analysis = result.get("analysis", "")

            task_results_str += f"\n--- Sub-task {task_id}: {task.get('description', '')} ---\n"
            task_results_str += f"Target action: {task.get('target_action', 'N/A')}\n"
            task_results_str += f"Target object: {task.get('target_object', 'N/A')}\n"
            task_results_str += f"Analysis: {analysis}\n"
            task_results_str += f"Clips found: {len(clips)}\n"

            valid_idx = 0
            for clip in clips:
                start, end = self._sanitize_timestamps(clip.get("start_time"), clip.get("end_time"))
                if start is None:
                    continue
                valid_idx += 1
                clip_id = f"T{task_id}-C{valid_idx}"
                clip_index[clip_id] = {
                    **clip,
                    "start_time": start,
                    "end_time": end,
                    "action": task.get("target_action", ""),
                    "object": task.get("target_object", ""),
                }
                task_results_str += f"  [{clip_id}] {clip.get('description', '')}\n"

        return task_results_str, clip_index

    @staticmethod
    def _sanitize_timestamps(start: Any, end: Any) -> tuple[float | None, float | None]:
        """Validate timestamps produced by the analyzer LLM.

        Returns ``(start, end)`` if valid, or ``(None, None)`` to drop
        the clip entirely.
        """
        try:
            s = float(str(start).rstrip("s"))
            e = float(str(end).rstrip("s"))
        except (ValueError, TypeError):
            logger.warning(f"Unparseable timestamps: start={start!r}, end={end!r} -- dropping clip")
            return None, None

        if e <= s:
            logger.warning(f"Invalid timestamps: start={s}, end={e} -- dropping clip")
            return None, None

        _MAX_DURATION = 60.0
        if e - s > _MAX_DURATION:
            logger.warning(
                f"Clip too long: {e - s:.0f}s (max {_MAX_DURATION:.0f}s) -- dropping clip"
            )
            return None, None

        return s, e

    def _resolve_clips(
        self,
        recommended_clips: list[dict],
        clip_index: dict[str, dict],
    ) -> list[dict]:
        """Resolve LLM-selected clip IDs into full clip metadata.

        The LLM output contains only ``clip_id``, ``priority``, and
        ``description``.  All numeric metadata (timestamps, paths) is
        looked up from *clip_index* which was built during formatting.
        """
        resolved: list[dict] = []
        for selection in recommended_clips:
            clip_id = selection.get("clip_id", "")
            if clip_id not in clip_index:
                logger.warning(f"Unknown clip_id '{clip_id}' from LLM output -- skipping")
                continue

            metadata = clip_index[clip_id]
            resolved.append(
                {
                    "start_time": metadata.get("start_time"),
                    "end_time": metadata.get("end_time"),
                    "video_id": metadata.get("video_id"),
                    "video_path": metadata.get("video_path", ""),
                    "action": metadata.get("action", ""),
                    "object": metadata.get("object", ""),
                    "description": selection.get("description", metadata.get("description", "")),
                    "priority": selection.get("priority", 99),
                    "clip_id": clip_id,
                }
            )

        logger.info(f"Resolved {len(resolved)}/{len(recommended_clips)} clip selections")
        return resolved

    def _extract_clips(
        self, recommended_clips: list[dict], default_video_path: str
    ) -> tuple[list[str], list[dict]]:
        """Extract recommended clips using the extract_clip tool.

        Returns:
            (extracted_paths, extracted_clip_infos) — two parallel lists
            so the webapp can map each path to its clip metadata.
        """
        extracted_paths: list[str] = []
        extracted_infos: list[dict] = []

        if not recommended_clips or "extract_clip" not in self.tools:
            return extracted_paths, extracted_infos

        logger.info(f"Extracting {len(recommended_clips)} clips...")

        for clip in sorted(recommended_clips, key=lambda x: x.get("priority", 99)):
            start = clip.get("start_time")
            end = clip.get("end_time")
            clip_id = clip.get("clip_id", "?")

            if start is None or end is None:
                logger.warning(f"  [{clip_id}] Skipped: missing start or end time")
                continue

            start = float(str(start).rstrip("s"))
            end = float(str(end).rstrip("s"))

            clip_video_path = clip.get("video_path") or default_video_path
            video_name = Path(clip_video_path).stem if clip_video_path else "video"

            logger.info(f"  [{clip_id}] {start:.1f}s-{end:.1f}s video={video_name}")

            output_name = self._create_clip_filename(
                video_name=video_name,
                action=clip.get("action", "clip"),
                obj=clip.get("object", ""),
                start=start,
                end=end,
                clip_idx=len(extracted_paths),
            )

            clip_result = self.tools["extract_clip"].execute(
                start_time=start, end_time=end, output_name=output_name, video_path=clip_video_path
            )

            if clip_result.success:
                logger.info(f"    ✓ Extracted: {clip_result.data}")
                extracted_paths.append(clip_result.data.output_path)
                extracted_infos.append(clip)
            else:
                logger.error(f"    ✗ Failed: {clip_result.error}")

        return extracted_paths, extracted_infos

    def _create_clip_filename(
        self, video_name: str, action: str, obj: str, start: Any, end: Any, clip_idx: int
    ) -> str:
        """Create a unique filename for the extracted clip.

        Args:
            video_name: Name of source video
            action: Action in the clip
            obj: Object in the clip
            start: Start timestamp
            end: End timestamp
            clip_idx: Index for fallback naming

        Returns:
            Filename (without extension)
        """
        action = action.replace(" ", "_")
        obj = obj.replace(" ", "_")

        try:
            start_sec = float(str(start).rstrip("s"))
            end_sec = float(str(end).rstrip("s"))
            time_str = f"{start_sec:.0f}s-{end_sec:.0f}s"
        except (ValueError, TypeError):
            time_str = f"{clip_idx + 1:03d}"

        return f"{video_name}_{action}_{obj}_{time_str}"[:60]

    def _build_answer(
        self,
        task_summary: str,
        recommended_clips: list[dict],
        missing: list[str],
        training_notes: str,
        default_video_path: str,
    ) -> str:
        """Build the final answer string.

        Args:
            task_summary: Summary of the task
            recommended_clips: Clips recommended for training
            missing: Missing demonstrations
            training_notes: Notes for training
            default_video_path: Default video path

        Returns:
            Formatted answer string
        """
        answer = f"**Task Summary:** {task_summary}\n\n"
        answer += "**Recommended Clips for Training:**\n"

        for i, clip in enumerate(recommended_clips, 1):
            clip_video = clip.get("video_path") or default_video_path
            video_name = Path(clip_video).name if clip_video else "unknown"
            answer += (
                f"{i}. [{clip.get('start_time', '?')}s - {clip.get('end_time', '?')}s] "
                f"{clip.get('action', '')} {clip.get('object', '')}: "
                f"{clip.get('description', '')} (from {video_name})\n"
            )

        if missing:
            answer += f"\n**Missing Demonstrations:** {', '.join(missing)}\n"

        if training_notes:
            answer += f"\n**Training Notes:** {training_notes}"

        return answer
