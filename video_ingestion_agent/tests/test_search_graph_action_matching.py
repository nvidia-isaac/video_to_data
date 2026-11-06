# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Form-tolerant action matching in SearchGraphTool.

The task decomposer emits actions such as ``pick_up`` while ingestion stores
free-form VLM phrases such as ``picking up and placing down``. Segment search
compared them with ``=`` and ``LIKE '%pick_up%'``, so every relaxation level
returned 0 segments and the agent fell through to the visual tool. These tests
use the four segments from the report.
"""

import sqlite3

import pytest

from video_ingestion_agent.retrieval.tools.search_graph import (
    SearchGraphTool,
    action_like_pattern,
    normalize_action,
)

# (id, action_type, primary_object_id, start_t, end_t, visual_evidence)
SEGMENTS = [
    (1, "placing down", "wooden spoon", 0.0, 4.0, "Person places the wooden spoon down"),
    (
        2,
        "picking up and placing down",
        "wooden spoon",
        4.0,
        8.0,
        "Person picks up wooden spoon and places it back down",
    ),
    (3, "picking up and placing down", "wooden bowl", 8.0, 10.0, "Person lifts the wooden bowl"),
    # action_type deliberately differs from the description so the
    # description-only branch of level 3 can be isolated.
    (4, "mixing", "wooden bowl", 10.0, 13.7, "Person stirs the wooden bowl with a spoon"),
]


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Pick_Up", "pick up"),
        ("pick-up", "pick up"),
        ("  stirring  ", "stirring"),
        ("picking   up", "picking up"),
    ],
)
def test_normalize_action_folds_case_and_separators(raw, expected):
    assert normalize_action(raw) == expected


@pytest.mark.parametrize(
    "action, pattern",
    [
        ("pick_up", "%pick%up%"),
        ("pick up", "%pick%up%"),
        ("picking up", "%pick%up%"),
        ("picks up", "%pick%up%"),
        ("picked up", "%pick%up%"),
        ("stirring", "%stir%"),
        ("placing down", "%plac%down%"),
        ("cutting", "%cut%"),
        ("pour", "%pour%"),
        ("open", "%open%"),
        ("press", "%press%"),  # doubled 's' is not an inflection artefact
        ("grasping", "%grasp%"),
    ],
)
def test_action_like_pattern_strips_inflection(action, pattern):
    assert action_like_pattern(action) == pattern


# ---------------------------------------------------------------------------
# Against a real SQLite graph.db
# ---------------------------------------------------------------------------


@pytest.fixture
def tool(tmp_path):
    db_path = tmp_path / "graph.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE video_metadata (id INTEGER PRIMARY KEY, video_path TEXT, "
        "duration REAL, fps REAL, width INTEGER, height INTEGER)"
    )
    conn.execute(
        "CREATE TABLE action_segments (id INTEGER PRIMARY KEY, action_type TEXT, "
        "primary_object_id TEXT, start_t REAL, end_t REAL, visual_evidence TEXT, "
        "video_id INTEGER)"
    )
    conn.execute("INSERT INTO video_metadata VALUES (1, '/v/stir_bowl.mp4', 13.7, 30, 1280, 720)")
    conn.executemany(
        "INSERT INTO action_segments VALUES (?, ?, ?, ?, ?, ?, 1)",
        SEGMENTS,
    )
    conn.commit()
    conn.close()
    t = SearchGraphTool(str(db_path))
    yield t
    t.close()


def hits(tool, **kwargs) -> set[int]:
    result = tool.execute(query_type="segments", **kwargs)
    assert result.success, result.error
    return {seg.segment_id for seg in result.data}


def test_report_scenario_succeeds_at_level_3(tool):
    """The exact failing query: action='pick_up', object='object', levels 0-3.

    Levels 0-2 are legitimately empty -- no object contains 'object' and the
    stored phrase is longer than 'pick up' -- but level 3's OR clause must now
    land on the two picking-up segments instead of returning 0.
    """
    for level in (0, 1, 2):
        assert hits(tool, action="pick_up", object_name="object", relaxation_level=level) == set()
    assert hits(tool, action="pick_up", object_name="object", relaxation_level=3) == {2, 3}


@pytest.mark.parametrize("action", ["pick_up", "pick up", "picking up", "picks up", "PICK_UP"])
def test_every_form_of_the_action_is_equivalent(tool, action):
    assert hits(tool, action=action, object_name="", relaxation_level=2) == {2, 3}


def test_level_2_narrows_by_object(tool):
    assert hits(tool, action="pick_up", object_name="spoon", relaxation_level=2) == {2}


def test_levels_0_and_1_still_require_the_whole_stored_phrase(tool):
    assert hits(tool, action="pick_up", object_name="", relaxation_level=0) == set()
    assert hits(tool, action="pick_up", object_name="", relaxation_level=1) == set()
    assert hits(tool, action="picking up and placing down", object_name="", relaxation_level=0) == {
        2,
        3,
    }


def test_level_0_is_case_and_underscore_insensitive(tool):
    assert hits(tool, action="Placing_Down", object_name="", relaxation_level=0) == {1}


def test_level_3_stem_matches_the_description(tool):
    """'stirring' is only in segment 4's description, under a different action_type."""
    assert hits(tool, action="stirring", object_name="", relaxation_level=2) == set()
    assert hits(tool, action="stirring", object_name="", relaxation_level=3) == {4}


def test_object_only_query_still_searches_description_at_level_3(tool):
    assert hits(tool, action="", object_name="spoon", relaxation_level=3) == {1, 2, 4}
