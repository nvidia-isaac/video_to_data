# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyzer node.

Analyzes search results and reports findings. Does NOT make decisions about
next actions - that's the SearchPlanner's responsibility.
"""

import logging
from typing import Any

from video_ingestion_agent.retrieval.nodes.base import BaseNode
from video_ingestion_agent.retrieval.nodes.prompts import (
    ANALYZE_RESULTS_SYSTEM,
    ANALYZE_RESULTS_USER,
)
from video_ingestion_agent.retrieval.state import AgentState

logger = logging.getLogger(__name__)


class AnalyzerNode(BaseNode):
    """Analyze search results and report findings.

    Evaluates whether the search results are relevant and extracts clips.
    Stores analysis in state for SearchPlanner to make decisions.
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Analyze results and report findings.

        Args:
            state: Current agent state

        Returns:
            State updates with analysis results (for SearchPlanner to act on)
        """
        logger.info("=== ANALYZER ===")

        sub_tasks = state.get("sub_tasks", [])
        task_idx = state.get("current_task_idx", 0)
        search_results = state.get("current_search_results", "")

        if task_idx >= len(sub_tasks):
            return {}

        current_task = sub_tasks[task_idx]
        task_id = current_task["task_id"]

        # Analyze results using LLM
        user_prompt = ANALYZE_RESULTS_USER.format(
            task_description=current_task["description"],
            target_action=current_task.get("target_action", ""),
            target_object=current_task.get("target_object", ""),
            search_results=search_results,
        )

        response = self._call_llm(user_prompt, system_prompt=ANALYZE_RESULTS_SYSTEM)
        logger.debug(f"Analyzer raw response: {response[:500]}...")
        analysis = self._parse_json(response)

        relevant = analysis.get("relevant", False)
        relevant_clips = analysis.get("relevant_clips", [])
        needs_relaxed = analysis.get("needs_relaxed_search", False)

        logger.info(f"Analysis: {analysis.get('analysis', '')}")
        logger.info(f"Relevant: {relevant}, Found {len(relevant_clips)} clips")
        for clip in relevant_clips:
            logger.info(
                f"  Clip: [{clip.get('start_time', '?')}s - {clip.get('end_time', '?')}s] "
                f"video_id={clip.get('video_id', '?')} "
                f"{clip.get('description', '')}"
            )

        # Update task results with found clips
        task_results = state.get("task_results", {})
        if task_id in task_results:
            task_results[task_id]["clips"].extend(relevant_clips)
            task_results[task_id]["analysis"] = analysis.get("analysis", "")

        # Store analysis for SearchPlanner to make decisions
        current_analysis = {
            "task_id": task_id,
            "relevant": relevant,
            "clips_found": len(relevant_clips),
            "needs_relaxed_search": needs_relaxed,
            "analysis_text": analysis.get("analysis", ""),
        }

        return {
            "task_results": {task_id: task_results.get(task_id, {})},
            "current_analysis": current_analysis,
            "working_memory": [
                f"[Analysis for task {task_id}]\n"
                f"Relevant: {relevant}, Clips: {len(relevant_clips)}\n"
                f"{analysis.get('analysis', '')}"
            ],
        }
