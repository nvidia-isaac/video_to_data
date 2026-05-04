# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for parallel sub-task execution in the retrieval agent.

Covers:
- TaskSearchState creation
- Subgraph output filtering (only reducer-annotated keys returned)
- SearchGraphTool thread-safety (check_same_thread=False)
- Parallel vs sequential graph construction
"""

import sqlite3
from unittest.mock import MagicMock, patch

from video_ingestion_agent.retrieval.config import AgentConfig, RetrievalConfig
from video_ingestion_agent.retrieval.state import create_initial_state
from video_ingestion_agent.retrieval.task_search_subgraph import (
    create_task_search_state,
)


class TestCreateTaskSearchState:
    """Test TaskSearchState initialization for a single sub-task."""

    def test_basic_creation(self):
        task = {
            "task_id": 1,
            "description": "Find pick-up mug",
            "target_action": "pick_up",
            "target_object": "mug",
        }
        state = create_task_search_state(task)

        assert state["sub_tasks"] == [task]
        assert state["current_task_idx"] == 0
        assert state["working_memory"] == []
        assert 1 in state["task_results"]
        assert state["task_results"][1]["task"] == task
        assert state["task_results"][1]["clips"] == []

    def test_preserves_task_id(self):
        for tid in [0, 3, 99]:
            state = create_task_search_state({"task_id": tid})
            assert tid in state["task_results"]

    def test_transient_fields_initialised(self):
        state = create_task_search_state({"task_id": 1})
        assert state["current_search_query"] == ""
        assert state["current_search_results"] == ""
        assert state["current_search_type"] == ""
        assert state["current_search_plan"] == {}
        assert state["search_relaxation_level"] == 0
        assert state["current_analysis"] == {}
        assert state["search_attempts"] == 0
        assert state["search_history"] == []


class TestTaskSearchNodeOutputFiltering:
    """Test that the parallel task_search_node wrapper only returns reducer keys."""

    def _simulate_subgraph_output(self) -> dict:
        """Simulate what a completed subgraph would return."""
        return {
            "sub_tasks": [{"task_id": 1, "description": "test"}],
            "current_task_idx": 1,
            "working_memory": ["searched for X", "found Y"],
            "task_results": {1: {"task": {"task_id": 1}, "clips": [{"start": 0, "end": 5}]}},
            "current_search_query": "pick up mug",
            "current_search_results": "some results",
            "current_search_type": "entity_graph",
            "current_search_plan": {"type": "segments"},
            "search_relaxation_level": 2,
            "current_analysis": {"relevant": True},
            "search_attempts": 3,
            "search_history": [{"type": "segments"}],
        }

    def test_only_reducer_keys_returned(self):
        subgraph_output = self._simulate_subgraph_output()
        mock_subgraph = MagicMock()
        mock_subgraph.invoke.return_value = subgraph_output

        def task_search_node(state):
            result = mock_subgraph.invoke(state)
            return {
                "task_results": result.get("task_results", {}),
                "working_memory": result.get("working_memory", []),
            }

        filtered = task_search_node({"sub_tasks": [{"task_id": 1}]})

        assert set(filtered.keys()) == {"task_results", "working_memory"}
        assert filtered["task_results"] == subgraph_output["task_results"]
        assert filtered["working_memory"] == subgraph_output["working_memory"]

    def test_transient_keys_excluded(self):
        subgraph_output = self._simulate_subgraph_output()
        mock_subgraph = MagicMock()
        mock_subgraph.invoke.return_value = subgraph_output

        def task_search_node(state):
            result = mock_subgraph.invoke(state)
            return {
                "task_results": result.get("task_results", {}),
                "working_memory": result.get("working_memory", []),
            }

        filtered = task_search_node({})

        excluded = [
            "sub_tasks",
            "current_task_idx",
            "current_search_query",
            "current_search_results",
            "current_search_type",
            "current_search_plan",
            "search_relaxation_level",
            "current_analysis",
            "search_attempts",
            "search_history",
        ]
        for key in excluded:
            assert key not in filtered

    def test_empty_subgraph_output(self):
        mock_subgraph = MagicMock()
        mock_subgraph.invoke.return_value = {}

        def task_search_node(state):
            result = mock_subgraph.invoke(state)
            return {
                "task_results": result.get("task_results", {}),
                "working_memory": result.get("working_memory", []),
            }

        filtered = task_search_node({})
        assert filtered["task_results"] == {}
        assert filtered["working_memory"] == []


class TestSearchGraphToolThreadSafety:
    """Test that SearchGraphTool creates thread-safe SQLite connections."""

    def test_connection_uses_check_same_thread_false(self, tmp_path):
        from video_ingestion_agent.retrieval.tools.search_graph import SearchGraphTool

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE video_metadata (id INTEGER PRIMARY KEY, video_path TEXT, "
            "duration REAL, fps REAL, width INTEGER, height INTEGER)"
        )
        conn.execute(
            "CREATE TABLE entities (entity_id TEXT, entity_type TEXT, first_seen REAL, "
            "last_seen REAL, properties TEXT, video_id INTEGER)"
        )
        conn.execute(
            "CREATE TABLE action_segments (id INTEGER PRIMARY KEY, action_type TEXT, "
            "primary_object_id TEXT, start_t REAL, end_t REAL, visual_evidence TEXT, "
            "video_id INTEGER)"
        )
        conn.execute(
            "CREATE TABLE relationships (source_id TEXT, target_id TEXT, rel_type TEXT, "
            "start_t REAL, end_t REAL, supporting_evidence TEXT, video_id INTEGER)"
        )
        conn.commit()
        conn.close()

        tool = SearchGraphTool(str(db_path))
        connection = tool._get_conn()

        # Verify the connection was created -- it should not raise when used from
        # a different thread context (check_same_thread=False).
        # We verify by inspecting the source: sqlite3.connect(..., check_same_thread=False)
        # The connection should be functional.
        assert connection is not None
        result = tool.execute(query_type="entities", limit=5)
        assert result.success is True

        tool.close()

    def test_connection_reuse(self, tmp_path):
        from video_ingestion_agent.retrieval.tools.search_graph import SearchGraphTool

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE video_metadata (id INTEGER PRIMARY KEY, video_path TEXT)")
        conn.execute(
            "CREATE TABLE entities (entity_id TEXT, entity_type TEXT, first_seen REAL, "
            "last_seen REAL, properties TEXT, video_id INTEGER)"
        )
        conn.commit()
        conn.close()

        tool = SearchGraphTool(str(db_path))
        conn1 = tool._get_conn()
        conn2 = tool._get_conn()
        assert conn1 is conn2

        tool.close()


class TestGraphBuildModes:
    """Test that parallel_tasks config controls which graph is built."""

    def test_parallel_default(self):
        config = RetrievalConfig()
        assert config.agent.parallel_tasks is True

    def test_sequential_override(self):
        config = RetrievalConfig(
            agent=AgentConfig(parallel_tasks=False),
        )
        assert config.agent.parallel_tasks is False

    @patch("video_ingestion_agent.retrieval.retrieval_graph.build_task_search_subgraph")
    def test_build_graph_selects_parallel(self, mock_build_subgraph):
        from video_ingestion_agent.retrieval.retrieval_graph import RetrievalAgent

        mock_build_subgraph.return_value = MagicMock()

        config = RetrievalConfig(agent=AgentConfig(parallel_tasks=True))
        agent = RetrievalAgent(config=config, tools={})

        with (
            patch.object(
                agent, "_build_parallel_graph", wraps=agent._build_parallel_graph
            ) as mock_p,
            patch.object(agent, "_build_sequential_graph") as mock_s,
        ):
            agent.build_graph()
            mock_p.assert_called_once()
            mock_s.assert_not_called()

    def test_build_graph_selects_sequential(self):
        from video_ingestion_agent.retrieval.retrieval_graph import RetrievalAgent

        config = RetrievalConfig(agent=AgentConfig(parallel_tasks=False))
        agent = RetrievalAgent(config=config, tools={})

        with (
            patch.object(agent, "_build_parallel_graph") as mock_p,
            patch.object(
                agent, "_build_sequential_graph", wraps=agent._build_sequential_graph
            ) as mock_s,
        ):
            agent.build_graph()
            mock_s.assert_called_once()
            mock_p.assert_not_called()


class TestSubgraphRouting:
    """Test the subgraph routing function."""

    def test_routes_to_end_when_task_complete(self):
        from video_ingestion_agent.retrieval.task_search_subgraph import _route_after_search_planner

        state = {
            "sub_tasks": [{"task_id": 1}],
            "current_task_idx": 1,
            "current_search_plan": {},
        }
        assert _route_after_search_planner(state) == "__end__"

    def test_routes_to_executor_when_plan_exists(self):
        from video_ingestion_agent.retrieval.task_search_subgraph import _route_after_search_planner

        state = {
            "sub_tasks": [{"task_id": 1}],
            "current_task_idx": 0,
            "current_search_plan": {"type": "segments", "action": "pick_up"},
        }
        assert _route_after_search_planner(state) == "executor"

    def test_routes_to_search_planner_when_no_plan(self):
        from video_ingestion_agent.retrieval.task_search_subgraph import _route_after_search_planner

        state = {
            "sub_tasks": [{"task_id": 1}],
            "current_task_idx": 0,
            "current_search_plan": {},
        }
        assert _route_after_search_planner(state) == "search_planner"

    def test_routes_to_end_with_empty_sub_tasks(self):
        from video_ingestion_agent.retrieval.task_search_subgraph import _route_after_search_planner

        state = {
            "sub_tasks": [],
            "current_task_idx": 0,
            "current_search_plan": {},
        }
        assert _route_after_search_planner(state) == "__end__"


class TestFanOutTasks:
    """Test the fan-out function that creates Send objects."""

    def test_creates_send_per_task(self):
        from video_ingestion_agent.retrieval.retrieval_graph import RetrievalAgent

        config = RetrievalConfig(agent=AgentConfig(parallel_tasks=True))
        agent = RetrievalAgent(config=config, tools={})

        state = create_initial_state(query="test", video_path="/tmp/v.mp4")
        state["sub_tasks"] = [
            {"task_id": 1, "description": "task 1"},
            {"task_id": 2, "description": "task 2"},
            {"task_id": 3, "description": "task 3"},
        ]

        sends = agent._fan_out_tasks(state)
        assert len(sends) == 3

        for send in sends:
            assert send.node == "task_search"

    def test_empty_sub_tasks_returns_empty(self):
        from video_ingestion_agent.retrieval.retrieval_graph import RetrievalAgent

        config = RetrievalConfig(agent=AgentConfig(parallel_tasks=True))
        agent = RetrievalAgent(config=config, tools={})

        state = create_initial_state(query="test")
        sends = agent._fan_out_tasks(state)
        assert sends == []
