# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Executor node.

Executes search operations for the current sub-task using available tools.
Uses search plan from SearchPlannerNode directly - no LLM calls.

Search types:
- "segments": Search for action clips in entity graph
- "entities": Search for objects in entity graph
- "relationships": Search for entity interactions in entity graph
- "visual": Search using visual embeddings

The search tool handles relaxation internally based on relaxation_level.
"""

import logging
from collections import defaultdict
from typing import Any

from video_ingestion_agent.retrieval.nodes.base import BaseNode
from video_ingestion_agent.retrieval.state import AgentState

logger = logging.getLogger(__name__)

# Search types that use the entity graph
GRAPH_SEARCH_TYPES = ("segments", "entities", "relationships")


class ExecutorNode(BaseNode):
    """Execute search for current sub-task.

    Simple executor that runs the search plan from SearchPlannerNode.
    No LLM calls - just passes parameters to search tools.

    Handles 4 search types:
    - segments: Action clips from entity graph
    - entities: Objects from entity graph
    - relationships: Entity interactions from entity graph
    - visual: Visual embedding search

    The search tool handles relaxation internally based on relaxation_level.
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Execute search for the current sub-task.

        Args:
            state: Current agent state

        Returns:
            State updates with search results
        """
        logger.info("=== EXECUTOR ===")

        sub_tasks = state.get("sub_tasks", [])
        task_idx = state.get("current_task_idx", 0)
        relaxation_level = state.get("search_relaxation_level", 0)

        if task_idx >= len(sub_tasks):
            logger.info("All sub-tasks completed")
            return {}

        current_task = sub_tasks[task_idx]
        task_id = current_task["task_id"]

        logger.info(f"Executing sub-task {task_id}: {current_task['description']}")
        logger.info(f"Relaxation level: {relaxation_level}")

        # Get search plan from state (set by SearchPlannerNode)
        search_plan = state.get("current_search_plan", {})
        search_type = state.get("current_search_type", "segments")

        logger.info(f"Search type (from planner): {search_type}")

        search_results = self._execute_search(
            search_plan=search_plan,
            description=current_task["description"],
            relaxation_level=relaxation_level,
        )

        logger.info(f"Results: {search_results[:300]}...")

        return {
            "current_search_query": current_task["description"],
            "current_search_results": search_results,
            "working_memory": [
                f"[Sub-task {task_id}: {current_task['description']}]\n"
                f"Search (type={search_type}, relaxation={relaxation_level}):\n{search_results}"
            ],
        }

    def _execute_search(
        self,
        search_plan: dict,
        description: str,
        relaxation_level: int,
    ) -> str:
        """Execute the appropriate search based on search type.

        Args:
            search_plan: Search plan from planner
            description: Search description
            relaxation_level: How relaxed the search should be

        Returns:
            Search results as string
        """
        search_type = search_plan.get("search_type", "segments")

        if search_type in GRAPH_SEARCH_TYPES:
            if "search_graph" not in self.tools:
                return f"Tool 'search_graph' not available for search_type: {search_type}"
            return self._search_graph(search_plan, description, relaxation_level)
        elif search_type == "visual":
            if "search_frames" not in self.tools:
                return "Tool 'search_frames' not available for visual search"
            return self._search_visual(search_plan, description)
        else:
            return f"Unknown search_type: {search_type}"

    def _search_graph(self, search_plan: dict, description: str, relaxation_level: int) -> str:
        """Search the entity graph database.

        Uses search plan directly - the search tool handles relaxation internally.

        Args:
            search_plan: Search plan from planner (contains search_type, action, object_name)
            description: Search description
            relaxation_level: Relaxation level for search (passed to tool)

        Returns:
            Search results as string
        """
        search_type = search_plan.get("search_type", "segments")

        params = {
            "query_type": search_type,  # segments, entities, or relationships
            "action": search_plan.get("action", ""),
            "object_name": search_plan.get("object_name", ""),
            "relaxation_level": relaxation_level,  # Tool handles relaxation internally
            "limit": 20,
        }

        logger.info(f"Search params: {params}")

        # Execute search
        result = self.tools["search_graph"].execute(**params)
        return result.to_string()

    def _search_visual(self, search_plan: dict, description: str) -> str:
        """Search using visual embeddings, resolving frames to action segments.

        When frames carry a ``segment_id`` (set during ingestion), they are
        grouped by segment and cross-referenced against ``action_segments`` in
        the graph DB so the analyzer receives exact clip boundaries instead of
        having to guess from individual frame timestamps.

        Args:
            search_plan: Search plan from planner
            description: Query description

        Returns:
            Search results as string with resolved segment context
        """
        query = search_plan.get("object_name", "") or description
        result = self.tools["search_frames"].execute(query=query, top_k=10)

        if not result.success or not result.data:
            return result.to_string()

        frames = result.data

        # Group frames by segment_id (None → "unlinked")
        seg_groups: dict[str | None, list] = defaultdict(list)
        for frame in frames:
            seg_groups[getattr(frame, "segment_id", None)].append(frame)

        has_segments = any(k is not None for k in seg_groups)

        # If no segment_id on any frame, fall back to plain output
        if not has_segments:
            return result.to_string()

        # Try to resolve segments via the graph DB
        graph_tool = self.tools.get("search_graph")

        lines: list[str] = []
        for seg_id, seg_frames in seg_groups.items():
            seg_frames.sort(key=lambda f: f.timestamp)
            best_sim = max(f.similarity for f in seg_frames)
            frame_strs = [
                f"{f.frame_id} @ {f.timestamp:.1f}s ({f.similarity:.3f})" for f in seg_frames
            ]

            if seg_id is None:
                lines.append(
                    f"Unlinked frames (best similarity: {best_sim:.3f}):\n  {', '.join(frame_strs)}"
                )
                continue

            # Resolve via graph DB
            min_t = seg_frames[0].timestamp
            max_t = seg_frames[-1].timestamp
            video_id = getattr(seg_frames[0], "video_id", None)
            video_path = getattr(seg_frames[0], "video_path", None)
            video_id_filter = (
                video_id
                if isinstance(video_id, int)
                else int(video_id)
                if isinstance(video_id, str) and video_id.isdigit()
                else None
            )

            resolved = []
            if graph_tool is not None and hasattr(graph_tool, "get_segments_overlapping"):
                resolved = graph_tool.get_segments_overlapping(
                    start_t=min_t,
                    end_t=max_t,
                    video_id=video_id_filter,
                    video_path=video_path,
                )

            if resolved:
                seg = self._select_best_overlap_segment(resolved, min_t=min_t, max_t=max_t)
                lines.append(
                    f"Segment {seg_id} [{seg.start_t:.1f}s - {seg.end_t:.1f}s]: "
                    f"{seg.action} {seg.object_name}"
                    f"{' (video: ' + seg.video_path + ')' if seg.video_path else ''}\n"
                    f"  Description: {seg.description or 'N/A'}\n"
                    f"  Matching frames (best similarity: {best_sim:.3f}): "
                    f"{', '.join(frame_strs)}"
                )
            else:
                lines.append(
                    f"Segment {seg_id} [{min_t:.1f}s - {max_t:.1f}s]"
                    f"{' (video: ' + str(video_path or video_id) + ')' if (video_path or video_id) else ''}\n"
                    f"  Matching frames (best similarity: {best_sim:.3f}): "
                    f"{', '.join(frame_strs)}"
                )

        return "\n\n".join(lines)

    @staticmethod
    def _select_best_overlap_segment(candidates: list[Any], min_t: float, max_t: float) -> Any:
        """Choose the best segment match for a frame time window.

        Ranking:
        1) Larger temporal overlap with [min_t, max_t]
        2) Smaller center-point distance as tie-breaker
        """
        query_center = (min_t + max_t) / 2.0

        def score(seg):
            overlap = max(0.0, min(max_t, seg.end_t) - max(min_t, seg.start_t))
            seg_center = (seg.start_t + seg.end_t) / 2.0
            center_dist = abs(seg_center - query_center)
            return (overlap, -center_dist)

        return max(candidates, key=score)
