# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Search planner node.

Central decision maker for all search strategy:
- Plans initial search (type and parameters) using LLM
- Decides next action based on analyzer results using LLM
- Decides when to move to next task
"""

import logging
from typing import Any

from video_ingestion_agent.retrieval.nodes.base import BaseNode
from video_ingestion_agent.retrieval.nodes.prompts import (
    SEARCH_ADJUSTMENT_SYSTEM,
    SEARCH_ADJUSTMENT_USER,
    SEARCH_PLANNING_SYSTEM,
    SEARCH_PLANNING_USER,
)
from video_ingestion_agent.retrieval.state import AgentState

logger = logging.getLogger(__name__)


class SearchPlannerNode(BaseNode):
    """Central search strategy planner using LLM.

    Makes all search-related decisions via LLM:
    - Initial search type and parameters for new tasks
    - Whether to relax, change search type, or move on based on analyzer feedback
    - When to move to the next task

    Includes safeguards against infinite loops:
    - Tracks which search types have been tried
    - Limits total search attempts per task
    """

    def __init__(self, **kwargs):
        """Initialize the search planner.

        Args:
            **kwargs: Passed to BaseNode (config, tools, debug, debug_dir)
        """
        super().__init__(**kwargs)
        self.max_relaxation_levels = self.config.agent.max_relaxation_levels
        self.max_search_attempts = self.config.agent.max_search_attempts

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Plan search strategy or decide next action based on analysis.

        Args:
            state: Current agent state

        Returns:
            State updates with search plan or task progression
        """
        logger.info("=== SEARCH PLANNER ===")

        sub_tasks = state.get("sub_tasks", [])
        task_idx = state.get("current_task_idx", 0)
        current_analysis = state.get("current_analysis", {})

        # All tasks completed
        if task_idx >= len(sub_tasks):
            logger.info("All sub-tasks completed")
            return {}

        current_task = sub_tasks[task_idx]
        task_id = current_task["task_id"]

        # Check if we have analysis from a previous search for this task
        if current_analysis and current_analysis.get("task_id") == task_id:
            # Analyzer has provided results - decide what to do next
            return self._decide_based_on_analysis(state, current_task, current_analysis)
        else:
            # No analysis yet - plan initial search for this task
            return self._plan_initial_search(state, current_task)

    def _plan_initial_search(self, state: AgentState, task: dict) -> dict[str, Any]:
        """Plan the initial search for a new task.

        Args:
            state: Current agent state
            task: Current task dict

        Returns:
            State updates with search plan
        """
        task_id = task["task_id"]
        working_memory = state.get("working_memory", [])

        logger.info(f"Planning initial search for task {task_id}: {task['description']}")

        # Build context from working memory (recent history)
        history_context = self._build_history_context(working_memory)

        # Use LLM to plan search strategy
        plan = self._plan_search(task, history_context)

        logger.info(f"Search plan: type={plan['search_type']}")
        logger.info(f"Action: {plan.get('action', '')}, Object: {plan.get('object_name', '')}")
        logger.info(f"Reasoning: {plan.get('reasoning', '')}")

        return {
            "current_search_type": plan["search_type"],
            "current_search_plan": plan,
            "search_relaxation_level": 0,  # Start at level 0
            "current_search_results": "",  # Clear any old results
            "current_analysis": {},  # Clear any old analysis
            "search_attempts": 1,  # First attempt
            "search_history": [],  # Start fresh history (will be populated after search)
            "working_memory": [
                f"[Search Plan for task {task_id}]\n"
                f"Type: {plan['search_type']}\n"
                f"Action: {plan.get('action', '')}, Object: {plan.get('object_name', '')}\n"
                f"Reasoning: {plan.get('reasoning', '')}"
            ],
        }

    def _decide_based_on_analysis(
        self, state: AgentState, task: dict, analysis: dict
    ) -> dict[str, Any]:
        """Decide next action based on analyzer's results using LLM.

        Args:
            state: Current agent state
            task: Current task dict
            analysis: Analysis results from AnalyzerNode

        Returns:
            State updates (relax search, change type, or move to next task)
        """
        task_id = task["task_id"]
        task_idx = state.get("current_task_idx", 0)
        relaxation_level = state.get("search_relaxation_level", 0)
        current_plan = state.get("current_search_plan", {})
        search_attempts = state.get("search_attempts", 0)
        search_history = state.get("search_history", [])

        # Record current search attempt in history
        current_attempt = {
            "search_type": current_plan.get("search_type", ""),
            "relaxation": relaxation_level,
            "action": current_plan.get("action", ""),
            "object": current_plan.get("object_name", ""),
            "clips_found": analysis.get("clips_found", 0),
            "relevant": analysis.get("relevant", False),
            "summary": analysis.get("analysis_text", "")[:100],  # Truncate
        }
        updated_history = search_history + [current_attempt]

        # Safeguard: check max attempts
        # Hard limit to prevent infinite loops
        if search_attempts >= self.max_search_attempts:
            logger.warning(
                f"Max search attempts ({self.max_search_attempts}) reached for task {task_id}. "
                "Forcing move to next task."
            )
            return self._move_to_next_task(task_idx, task_id, "Max search attempts reached")

        logger.info(f"Asking LLM to decide next action for task {task_id}")
        logger.info(f"Attempts: {search_attempts}/{self.max_search_attempts}")

        # Use LLM to decide next action (pass full search history)
        decision = self._get_adjustment_decision(
            task, analysis, current_plan, relaxation_level, updated_history
        )

        action = decision.get("action")
        reasoning = decision.get("reasoning", "")

        # Warn if LLM didn't return required action field
        if not action:
            logger.warning("LLM response missing 'action' field, defaulting to 'next_task'")
            action = "next_task"

        logger.info(f"LLM decision: {action}")
        logger.info(f"Reasoning: {reasoning}")

        if action == "next_task":
            return self._move_to_next_task(task_idx, task_id, reasoning)
        elif action == "relax_search":
            # Relax the search (increment level)
            new_level = min(relaxation_level + 1, self.max_relaxation_levels)
            logger.info(f"Relaxing search to level {new_level}")
            return {
                "search_relaxation_level": new_level,
                "current_search_results": "",
                "current_analysis": {},
                "search_attempts": search_attempts + 1,
                "search_history": updated_history,
                "working_memory": [f"[Relaxing search for task {task_id}]\n{reasoning}"],
            }
        elif action == "change_search_type":
            # Change search type
            new_type = decision.get("new_search_type", "visual")
            new_action = decision.get("new_action", current_plan.get("action", ""))
            new_object = decision.get("new_object", current_plan.get("object_name", ""))

            # Validate new search type
            valid_types = ("segments", "entities", "relationships", "visual")
            if new_type not in valid_types:
                new_type = "visual"

            logger.info(f"Changing search type to: {new_type}")

            new_plan = {
                "search_type": new_type,
                "action": new_action,
                "object_name": new_object,
                "reasoning": reasoning,
            }

            return {
                "current_search_type": new_type,
                "current_search_plan": new_plan,
                "search_relaxation_level": 0,  # Reset relaxation for new type
                "current_search_results": "",
                "current_analysis": {},
                "search_attempts": search_attempts + 1,
                "search_history": updated_history,
                "working_memory": [
                    f"[Changing search type for task {task_id}]\nNew type: {new_type}\n{reasoning}"
                ],
            }
        else:
            # Unknown action - default to next task
            logger.warning(f"Unknown action '{action}', defaulting to next_task")
            return self._move_to_next_task(task_idx, task_id, f"Unknown action: {action}")

    def _move_to_next_task(self, task_idx: int, task_id: int, reasoning: str) -> dict[str, Any]:
        """Move to the next task and reset tracking state.

        Args:
            task_idx: Current task index
            task_id: Current task ID
            reasoning: Reason for moving

        Returns:
            State updates for moving to next task
        """
        logger.info("Moving to next sub-task")
        return {
            "current_task_idx": task_idx + 1,
            "search_relaxation_level": 0,
            "current_search_results": "",
            "current_search_type": "",
            "current_search_plan": {},
            "current_analysis": {},
            "search_attempts": 0,  # Reset for next task
            "search_history": [],  # Reset for next task
            "working_memory": [f"[Decision for task {task_id}]\n{reasoning}"],
        }

    def _get_adjustment_decision(
        self,
        task: dict,
        analysis: dict,
        current_plan: dict,
        relaxation_level: int,
        search_history: list[dict],
    ) -> dict:
        """Get LLM decision on next action based on analysis.

        Args:
            task: Current task dict
            analysis: Analysis results from AnalyzerNode
            current_plan: Current search plan
            relaxation_level: Current relaxation level
            search_history: Detailed history of all search attempts

        Returns:
            Decision dict with action, reasoning, and optional new parameters
        """
        # Format detailed search history
        history_str = self._format_search_history(search_history)

        user_prompt = SEARCH_ADJUSTMENT_USER.format(
            task_description=task["description"],
            target_action=task.get("target_action", ""),
            target_object=task.get("target_object", ""),
            current_search_type=current_plan.get("search_type", "segments"),
            relaxation_level=relaxation_level,
            max_relaxation=self.max_relaxation_levels,
            clips_found=analysis.get("clips_found", 0),
            relevant=analysis.get("relevant", False),
            needs_relaxed=analysis.get("needs_relaxed_search", False),
            analysis_text=analysis.get("analysis_text", ""),
            search_history=history_str,
        )

        response = self._call_llm(user_prompt, system_prompt=SEARCH_ADJUSTMENT_SYSTEM)
        return self._parse_json(response)

    def _format_search_history(self, search_history: list[dict]) -> str:
        """Format search history for the prompt.

        Args:
            search_history: List of search attempt dicts

        Returns:
            Formatted string showing what was tried and results
        """
        if not search_history:
            return "No previous attempts"

        # Track max relaxation reached per type
        max_relaxation_per_type: dict[str, int] = {}

        lines = []
        for i, attempt in enumerate(search_history, 1):
            search_type = attempt.get("search_type", "?")
            relaxation = attempt.get("relaxation", 0)
            action = attempt.get("action", "?")
            obj = attempt.get("object", "?")
            clips = attempt.get("clips_found", 0)
            relevant = attempt.get("relevant", False)

            # Track max level per type
            if search_type not in max_relaxation_per_type:
                max_relaxation_per_type[search_type] = relaxation
            else:
                max_relaxation_per_type[search_type] = max(
                    max_relaxation_per_type[search_type], relaxation
                )

            result = f"{clips} clips" if clips > 0 else "no results"
            if relevant:
                result += " (RELEVANT)"

            lines.append(
                f"  {i}. {search_type}(relax={relaxation}): "
                f"action='{action}', object='{obj}' → {result}"
            )

        # Add summary of max levels reached
        lines.append("")
        lines.append("MAX LEVELS REACHED:")
        for stype in ["segments", "relationships", "visual"]:
            max_level = max_relaxation_per_type.get(stype, -1)
            if max_level >= 0:
                exhausted = "(EXHAUSTED)" if max_level >= 3 else ""
                lines.append(f"  - {stype}: level {max_level}/3 {exhausted}")
            else:
                lines.append(f"  - {stype}: not tried yet")

        return "\n".join(lines)

    def _build_history_context(self, working_memory: list[str], max_entries: int = 5) -> str:
        """Build a context string from recent working memory entries.

        Args:
            working_memory: List of memory entries
            max_entries: Maximum number of recent entries to include

        Returns:
            Context string summarizing recent history
        """
        if not working_memory:
            return ""

        # Take the most recent entries
        recent = working_memory[-max_entries:]
        return "PREVIOUS SEARCH HISTORY:\n" + "\n---\n".join(recent)

    def _plan_search(self, task: dict, history_context: str) -> dict:
        """Plan the search strategy using LLM.

        Args:
            task: Current task dict
            history_context: Context from previous searches

        Returns:
            Search plan dict with search_type, action, object_name, reasoning
        """
        user_prompt = SEARCH_PLANNING_USER.format(
            task_description=task["description"],
            target_action=task.get("target_action", ""),
            target_object=task.get("target_object", ""),
            history_context=history_context,
        )

        response = self._call_llm(
            user_prompt, system_prompt=SEARCH_PLANNING_SYSTEM, max_tokens=4096
        )
        plan = self._parse_json(response)

        # Validate search_type against the 4 flat types
        valid_types = ("segments", "entities", "relationships", "visual")
        search_type = plan.get("search_type", "segments")
        if search_type not in valid_types:
            logger.warning(f"Invalid search_type '{search_type}', defaulting to 'segments'")
            search_type = "segments"

        return {
            "search_type": search_type,
            "action": plan.get("action", task.get("target_action", "")),
            "object_name": plan.get("object_name", task.get("target_object", "")),
            "reasoning": plan.get("reasoning", ""),
        }
