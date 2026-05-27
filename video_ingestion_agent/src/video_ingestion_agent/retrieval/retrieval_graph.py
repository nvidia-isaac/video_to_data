# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""LangGraph-based agent for video clip retrieval.

Enhanced with EGAgent paper (arXiv:2601.18157) patterns:
1. Multi-step task decomposition - breaks high-level tasks into sub-tasks
2. Search planner - central decision maker for search strategy and relaxation
3. Strict-to-relaxed search - hierarchical query relaxation
4. Analyzer - analyzes search results (reports to search_planner)
5. Dedicated VQA synthesizer - for robot policy training clip extraction

Supports two execution modes (controlled by ``config.agent.parallel_tasks``):

**Parallel** (default):
    START → task_decomposer → [Send per task → task_search subgraph] → vqa_synthesizer → END

    Sub-tasks run concurrently via the LangGraph Send API.  Each sub-task
    gets its own compiled subgraph containing the search loop.

**Sequential** (``parallel_tasks: false``):
    START → task_decomposer → search_planner ⟷ executor ⟷ analyzer → vqa_synthesizer → END

    Sub-tasks run one at a time via a ``current_task_idx`` counter.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from video_ingestion_agent.retrieval.config import RetrievalConfig
from video_ingestion_agent.retrieval.nodes import (
    AnalyzerNode,
    ExecutorNode,
    SearchPlannerNode,
    TaskDecomposerNode,
    VQASynthesizerNode,
)
from video_ingestion_agent.retrieval.state import AgentState, create_initial_state
from video_ingestion_agent.retrieval.task_search_subgraph import (
    build_task_search_subgraph,
    create_task_search_state,
)
from video_ingestion_agent.retrieval.tools.base import BaseTool

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """LangGraph-based retrieval agent for robot policy training.

    Supports parallel and sequential sub-task execution (see module docstring).
    """

    def __init__(
        self,
        config: RetrievalConfig,
        tools: dict[str, BaseTool],
        debug: bool = False,
        debug_dir: str | None = None,
    ):
        """Initialize LangGraph agent.

        Args:
            config: Retrieval agent configuration
            tools: Dict of tool_name -> BaseTool instances
            debug: Enable debug logging of LLM inputs/outputs
            debug_dir: Directory to save debug logs
        """
        self.config = config
        self.tools = tools
        self.max_sub_tasks = config.agent.max_sub_tasks

        self.debug = debug or os.environ.get("VIDEO_INGESTION_AGENT_DEBUG", "").lower() in (
            "1",
            "true",
        )

        if self.debug:
            base_debug_dir = Path(debug_dir or "./debug_logs")
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = base_debug_dir / session_timestamp
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug logs will be saved to: {self.debug_dir}")
        else:
            self.debug_dir = None

        self._node_kwargs: dict[str, Any] = {
            "config": config,
            "tools": tools,
            "debug": debug,
            "debug_dir": str(self.debug_dir) if self.debug_dir else None,
        }

        self.task_decomposer = TaskDecomposerNode(**self._node_kwargs)
        self.vqa_synthesizer = VQASynthesizerNode(**self._node_kwargs)

        self._graph = None

    # =========================================================================
    # Parallel graph (Send API)
    # =========================================================================

    def _fan_out_tasks(self, state: AgentState) -> list[Send]:
        """Fan out sub-tasks to parallel search subgraph branches.

        Called as a conditional edge after ``task_decomposer``.  Returns one
        ``Send`` per sub-task, each targeting the ``task_search`` subgraph
        node with a self-contained ``TaskSearchState``.
        """
        sub_tasks = state.get("sub_tasks", [])
        return [Send("task_search", create_task_search_state(task)) for task in sub_tasks]

    def _build_parallel_graph(self) -> Any:
        """Build graph with parallel sub-task execution via Send API."""
        compiled_subgraph = build_task_search_subgraph(
            config=self.config,
            tools=self.tools,
            debug=self.debug,
            debug_dir=str(self.debug_dir) if self.debug_dir else None,
        )

        def task_search_node(state):
            """Run the search subgraph and return only reducer-annotated keys.

            Filters out transient per-task keys (sub_tasks, current_task_idx,
            etc.) that would cause INVALID_CONCURRENT_GRAPH_UPDATE errors
            when multiple parallel branches write to the parent AgentState.
            """
            result = compiled_subgraph.invoke(state)
            return {
                "task_results": result.get("task_results", {}),
                "working_memory": result.get("working_memory", []),
            }

        wf = StateGraph(AgentState)

        wf.add_node("task_decomposer", self.task_decomposer)
        wf.add_node("task_search", task_search_node)
        wf.add_node("vqa_synthesizer", self.vqa_synthesizer)

        wf.add_edge(START, "task_decomposer")
        wf.add_conditional_edges(
            "task_decomposer",
            self._fan_out_tasks,
            ["task_search"],
        )
        wf.add_edge("task_search", "vqa_synthesizer")
        wf.add_edge("vqa_synthesizer", END)

        return wf.compile()

    # =========================================================================
    # Sequential graph (original loop)
    # =========================================================================

    def _route_after_search_planner(self, state: AgentState) -> str:
        """Route after search_planner node (sequential mode only).

        Search planner either:
        - Plans a new search (has plan) -> go to executor
        - Moved to next task (no plan yet) -> loop back to search_planner
        - All tasks done -> go to vqa_synthesizer
        """
        sub_tasks = state.get("sub_tasks", [])
        task_idx = state.get("current_task_idx", 0)
        search_plan = state.get("current_search_plan", {})

        if task_idx >= len(sub_tasks):
            return "vqa_synthesizer"

        if not search_plan:
            return "search_planner"

        return "executor"

    def _build_sequential_graph(self) -> Any:
        """Build graph with sequential sub-task execution (original behaviour)."""
        search_planner = SearchPlannerNode(**self._node_kwargs)
        executor = ExecutorNode(**self._node_kwargs)
        analyzer = AnalyzerNode(**self._node_kwargs)

        wf = StateGraph(AgentState)

        wf.add_node("task_decomposer", self.task_decomposer)
        wf.add_node("search_planner", search_planner)
        wf.add_node("executor", executor)
        wf.add_node("analyzer", analyzer)
        wf.add_node("vqa_synthesizer", self.vqa_synthesizer)

        wf.add_edge(START, "task_decomposer")
        wf.add_edge("task_decomposer", "search_planner")

        wf.add_conditional_edges(
            "search_planner",
            self._route_after_search_planner,
            {
                "executor": "executor",
                "search_planner": "search_planner",
                "vqa_synthesizer": "vqa_synthesizer",
            },
        )

        wf.add_edge("executor", "analyzer")
        wf.add_edge("analyzer", "search_planner")
        wf.add_edge("vqa_synthesizer", END)

        return wf.compile()

    # =========================================================================
    # Graph Construction
    # =========================================================================

    def build_graph(self) -> Any:
        """Build the LangGraph workflow.

        Selects parallel or sequential mode based on
        ``config.agent.parallel_tasks``.

        Returns:
            Compiled StateGraph
        """
        if self.config.agent.parallel_tasks:
            logger.info("Building parallel retrieval graph (Send API)")
            return self._build_parallel_graph()
        else:
            logger.info("Building sequential retrieval graph")
            return self._build_sequential_graph()

    # =========================================================================
    # Public Interface
    # =========================================================================

    def run(self, query: str, video_path: str = "") -> dict[str, Any]:
        """Run the agent to find clips for robot policy training.

        Args:
            query: High-level task description (e.g., "make me a coffee")
            video_path: Path to source video

        Returns:
            Dict with answer, clips_to_extract, clips_extracted, sub_tasks
        """
        logger.info(f"Query: {query}")

        if self._graph is None:
            self._graph = self.build_graph()

        initial_state = create_initial_state(
            query=query, video_path=video_path, max_sub_tasks=self.max_sub_tasks
        )

        try:
            final_state = self._graph.invoke(initial_state)

            return {
                "success": True,
                "answer": final_state.get("final_answer", ""),
                "clips_to_extract": final_state.get("clips_to_extract", []),
                "clips_extracted": final_state.get("clips_extracted", []),
                "sub_tasks": final_state.get("sub_tasks", []),
                "task_results": final_state.get("task_results", {}),
                "working_memory": final_state.get("working_memory", []),
            }

        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            return {
                "success": False,
                "answer": "",
                "clips_to_extract": [],
                "clips_extracted": [],
                "error": str(e),
            }

    def run_streaming(self, query: str, video_path: str = ""):
        """Run the agent with streaming updates.

        Yields state updates after each node execution.

        Args:
            query: High-level task description
            video_path: Path to source video

        Yields:
            Dict with node_name, state snapshot, and working_memory updates
        """
        logger.info(f"Query (streaming): {query}")

        if self._graph is None:
            self._graph = self.build_graph()

        initial_state = create_initial_state(
            query=query, video_path=video_path, max_sub_tasks=self.max_sub_tasks
        )

        final_state = None

        try:
            for state_update in self._graph.stream(initial_state):
                for node_name, node_output in state_update.items():
                    working_memory = node_output.get("working_memory", [])

                    yield {
                        "node_name": node_name,
                        "status": "completed",
                        "working_memory": working_memory,
                        "current_task_idx": node_output.get("current_task_idx"),
                        "search_type": node_output.get("current_search_type"),
                        "relaxation_level": node_output.get("search_relaxation_level"),
                    }

                    final_state = node_output

            if final_state:
                yield {
                    "node_name": "__final__",
                    "status": "done",
                    "success": True,
                    "answer": final_state.get("final_answer", ""),
                    "clips_to_extract": final_state.get("clips_to_extract", []),
                    "clips_extracted": final_state.get("clips_extracted", []),
                    "sub_tasks": final_state.get("sub_tasks", []),
                    "task_results": final_state.get("task_results", {}),
                    "working_memory": final_state.get("working_memory", []),
                }

        except Exception as e:
            logger.error(f"Agent streaming failed: {e}", exc_info=True)
            yield {
                "node_name": "__error__",
                "status": "error",
                "success": False,
                "error": str(e),
            }
