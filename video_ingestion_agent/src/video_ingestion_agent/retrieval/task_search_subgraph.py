# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-task search subgraph for parallel execution.

Encapsulates the search_planner -> executor -> analyzer loop for a single
sub-task as a compiled LangGraph subgraph.  Used by the Send API to fan out
independent sub-tasks in parallel.

The subgraph reuses the existing node classes (SearchPlannerNode, ExecutorNode,
AnalyzerNode) without modification.  Each parallel branch receives a
``TaskSearchState`` containing a *single* task in ``sub_tasks`` (as a
one-element list so that ``sub_tasks[current_task_idx]`` works unchanged).
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from video_ingestion_agent.retrieval.config import RetrievalConfig
from video_ingestion_agent.retrieval.nodes import AnalyzerNode, ExecutorNode, SearchPlannerNode
from video_ingestion_agent.retrieval.state import append_list, merge_dict
from video_ingestion_agent.retrieval.tools.base import BaseTool


class TaskSearchState(TypedDict):
    """State for a single sub-task's search loop.

    Contains the same field names that SearchPlannerNode, ExecutorNode, and
    AnalyzerNode read/write via ``state.get(...)``, so those nodes work
    unmodified inside this subgraph.

    ``sub_tasks`` is always a single-element list and ``current_task_idx``
    starts at 0.  When the search planner "moves to next task" (idx → 1),
    the routing function detects ``idx >= len(sub_tasks)`` and ends.
    """

    sub_tasks: list[dict]
    current_task_idx: int

    working_memory: Annotated[list[str], append_list]
    task_results: Annotated[dict[int, dict], merge_dict]

    current_search_query: str
    current_search_results: str
    current_search_type: str
    current_search_plan: dict
    search_relaxation_level: int

    current_analysis: dict

    search_attempts: int
    search_history: list[dict]


def _route_after_search_planner(state: TaskSearchState) -> str:
    """Route within the subgraph after the search planner runs.

    Returns ``END`` (instead of ``"vqa_synthesizer"``) when the single
    task is complete.
    """
    sub_tasks = state.get("sub_tasks", [])
    task_idx = state.get("current_task_idx", 0)
    search_plan = state.get("current_search_plan", {})

    if task_idx >= len(sub_tasks):
        return END

    if not search_plan:
        return "search_planner"

    return "executor"


def build_task_search_subgraph(
    config: RetrievalConfig,
    tools: dict[str, BaseTool],
    debug: bool = False,
    debug_dir: str | None = None,
) -> Any:
    """Build and compile a subgraph for one sub-task's search loop.

    Args:
        config: Retrieval agent configuration.
        tools: Shared tool instances (read-only DB queries, thread-safe).
        debug: Enable debug logging.
        debug_dir: Directory for debug logs.

    Returns:
        Compiled LangGraph subgraph.
    """
    node_kwargs: dict[str, Any] = {
        "config": config,
        "tools": tools,
        "debug": debug,
        "debug_dir": debug_dir,
    }

    search_planner = SearchPlannerNode(**node_kwargs)
    executor = ExecutorNode(**node_kwargs)
    analyzer = AnalyzerNode(**node_kwargs)

    wf = StateGraph(TaskSearchState)

    wf.add_node("search_planner", search_planner)
    wf.add_node("executor", executor)
    wf.add_node("analyzer", analyzer)

    wf.add_edge(START, "search_planner")

    wf.add_conditional_edges(
        "search_planner",
        _route_after_search_planner,
        {
            "executor": "executor",
            "search_planner": "search_planner",
            END: END,
        },
    )

    wf.add_edge("executor", "analyzer")
    wf.add_edge("analyzer", "search_planner")

    return wf.compile()


def create_task_search_state(task: dict) -> TaskSearchState:
    """Create initial state for a single sub-task's search subgraph.

    Args:
        task: Sub-task dict (must contain ``task_id``).

    Returns:
        Initialised TaskSearchState ready for the subgraph.
    """
    task_id = task["task_id"]
    return {
        "sub_tasks": [task],
        "current_task_idx": 0,
        "working_memory": [],
        "task_results": {task_id: {"task": task, "clips": []}},
        "current_search_query": "",
        "current_search_results": "",
        "current_search_type": "",
        "current_search_plan": {},
        "search_relaxation_level": 0,
        "current_analysis": {},
        "search_attempts": 0,
        "search_history": [],
    }
