# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Agent state definitions and data classes.

This module defines the state that flows through the LangGraph agent,
along with supporting data classes and reducers.
"""

from dataclasses import dataclass
from typing import Annotated, TypedDict

# =============================================================================
# Custom Reducers
# =============================================================================


def append_list(left: list, right: list) -> list:
    """Append right list to left list."""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


def merge_dict(left: dict, right: dict) -> dict:
    """Merge right dict into left dict."""
    if left is None:
        left = {}
    if right is None:
        right = {}
    result = left.copy()
    result.update(right)
    return result


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SubTask:
    """A decomposed sub-task for retrieval."""

    task_id: int
    description: str
    search_query: str
    target_action: str | None = None  # e.g., "pick_up", "place", "open"
    target_object: str | None = None  # e.g., "mug", "coffee_machine"
    completed: bool = False
    results: str | None = None


# =============================================================================
# State Definition
# =============================================================================


class AgentState(TypedDict):
    """State that flows through the LangGraph agent.

    Attributes:
        query: The original user query
        video_path: Path to source video (or default video)
        sub_tasks: List of decomposed sub-tasks as dicts
        current_task_idx: Index of currently executing sub-task
        working_memory: Accumulated search results and analysis (append-only)
        task_results: Per-task results mapping task_id -> results
        current_search_query: Query for current search iteration
        current_search_results: Results from current search
        search_relaxation_level: 0=strict, 1=wider, 2=partial, 3=type-only
        final_answer: Final synthesized answer
        clips_to_extract: List of clips to extract
        clips_extracted: List of extracted clip paths
        max_sub_tasks: Maximum number of sub-tasks
        error: Error message if any
    """

    # Input
    query: str
    video_path: str

    # Task decomposition
    sub_tasks: list[dict]
    current_task_idx: int

    # Working memory (accumulated search results)
    working_memory: Annotated[list[str], append_list]

    # Per-task results mapping: task_id -> results
    task_results: Annotated[dict[int, dict], merge_dict]

    # Current search state (set by search_planner)
    current_search_query: str
    current_search_results: str
    current_search_type: str  # "entity_graph" or "visual"
    current_search_plan: (
        dict  # Full search plan from planner (type, query_type, action, object_name)
    )
    search_relaxation_level: int

    # Analysis results (set by analyzer, read by search_planner)
    current_analysis: dict  # {task_id, relevant, clips_found, needs_relaxed_search, analysis_text}

    # Search history tracking (to prevent infinite loops)
    search_attempts: int  # Total search attempts for current task
    search_history: list[dict]  # Detailed history: [{type, relaxation, action, object, result}]

    # Output
    final_answer: str
    clips_to_extract: list[dict]
    clips_extracted: list[str]

    # Control
    max_sub_tasks: int
    error: str | None


def create_initial_state(query: str, video_path: str = "", max_sub_tasks: int = 5) -> AgentState:
    """Create an initial agent state.

    Args:
        query: The user's query
        video_path: Path to source video
        max_sub_tasks: Maximum sub-tasks to decompose into

    Returns:
        Initialized AgentState
    """
    return {
        "query": query,
        "video_path": video_path,
        "sub_tasks": [],
        "current_task_idx": 0,
        "working_memory": [],
        "task_results": {},
        "current_search_query": "",
        "current_search_results": "",
        "current_search_type": "",
        "current_search_plan": {},
        "search_relaxation_level": 0,
        "current_analysis": {},
        "search_attempts": 0,
        "search_history": [],
        "final_answer": "",
        "clips_to_extract": [],
        "clips_extracted": [],
        "max_sub_tasks": max_sub_tasks,
        "error": None,
    }
