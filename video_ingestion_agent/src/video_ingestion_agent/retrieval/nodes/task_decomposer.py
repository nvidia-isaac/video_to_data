# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task decomposition node.

Breaks down high-level user queries into concrete sub-tasks for retrieval.
"""

import logging
from typing import Any

from video_ingestion_agent.retrieval.nodes.base import BaseNode
from video_ingestion_agent.retrieval.nodes.prompts import (
    TASK_DECOMPOSITION_SYSTEM,
    TASK_DECOMPOSITION_USER,
)
from video_ingestion_agent.retrieval.state import AgentState

logger = logging.getLogger(__name__)


class TaskDecomposerNode(BaseNode):
    """Decompose high-level task into sub-tasks.

    Takes the user query and breaks it down into specific manipulation
    actions that can be searched in the video database.

    Example:
        "make me a coffee" -> [
            "Find clips of grabbing a mug",
            "Find clips of opening coffee machine",
            "Find clips of pouring coffee"
        ]
    """

    def __init__(self, **kwargs):
        """Initialize the task decomposer.

        Args:
            **kwargs: Passed to BaseNode (config, tools, debug, debug_dir)
        """
        super().__init__(**kwargs)
        self.max_sub_tasks = self.config.agent.max_sub_tasks

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Decompose the query into sub-tasks.

        Args:
            state: Current agent state with query

        Returns:
            State updates with sub_tasks and task_results
        """
        logger.info("=== TASK DECOMPOSER ===")
        logger.info(f"Query: {state['query']}")

        user_prompt = TASK_DECOMPOSITION_USER.format(
            query=state["query"], max_sub_tasks=state.get("max_sub_tasks", self.max_sub_tasks)
        )

        response = self._call_llm(user_prompt, system_prompt=TASK_DECOMPOSITION_SYSTEM)
        result = self._parse_json(response)

        sub_tasks = result.get("sub_tasks", [])

        logger.info(f"Task analysis: {result.get('task_analysis', '')}")
        logger.info(f"Decomposed into {len(sub_tasks)} sub-tasks:")
        for task in sub_tasks:
            logger.info(
                f"  {task.get('task_id')}: {task.get('description')} "
                f"(action: {task.get('target_action')}, object: {task.get('target_object')})"
            )

        # Initialize task results dict
        task_results = {task["task_id"]: {"task": task, "clips": []} for task in sub_tasks}

        return {
            "sub_tasks": sub_tasks,
            "current_task_idx": 0,
            "task_results": task_results,
            "search_relaxation_level": 0,
            "working_memory": [f"[Task Analysis]\n{result.get('task_analysis', '')}"],
        }
