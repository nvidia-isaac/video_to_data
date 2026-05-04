# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared parsing utilities for LLM responses.

Consolidates duplicate JSON parsing and timestamp parsing logic that was
previously spread across segmenter.py, critic.py, action_segmenter.py,
entity_extractor.py, and agents/nodes/base.py.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from LLM responses.

    Cosmos-Reason2 (and similar models) emit chain-of-thought reasoning
    inside ``<think>`` tags before the actual answer.  The reasoning can
    contain JSON-like fragments that confuse downstream parsing, so we
    strip it before looking for the real answer.

    If ``<answer>`` tags are present the function returns only the content
    between the *first* ``<answer>`` and ``</answer>`` pair.  Otherwise it
    returns the text with ``<think>`` blocks removed.
    """
    # Prefer <answer> block if present
    answer_match = re.search(r"<answer>([\s\S]*?)</answer>", text)
    if answer_match:
        return answer_match.group(1).strip()

    # Fall back: strip <think> blocks
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_llm_json(text: str, expect_array: bool = False) -> list | dict:
    """
    Extract JSON from an LLM response.

    Handles responses wrapped in markdown code blocks (```json ... ```)
    as well as raw JSON. Can extract either a JSON array or a JSON object.

    Automatically strips ``<think>...</think>`` reasoning blocks so that
    chain-of-thought content does not interfere with JSON extraction.
    When ``<answer>`` tags are present only the answer section is searched.

    Args:
        text: Raw LLM response text.
        expect_array: If True, look for a JSON array (``[...]``).
            If False, look for a JSON object (``{...}``).

    Returns:
        Parsed JSON (list if expect_array, dict otherwise).

    Raises:
        ValueError: If no valid JSON is found in the response.
    """
    # Strip chain-of-thought reasoning; focus on <answer> block if present
    text = _strip_think_tags(text)

    # Try to find JSON in markdown code block first
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            result = json.loads(json_str)
            if expect_array and isinstance(result, list):
                return result
            if not expect_array and isinstance(result, dict):
                return result
            # Type mismatch -- fall through to raw search
        except json.JSONDecodeError:
            pass  # Fall through to raw search

    # Try to find raw JSON
    if expect_array:
        json_match = re.search(r"\[[\s\S]*\]", text)
        kind = "array"
    else:
        json_match = re.search(r"\{[\s\S]*\}", text)
        kind = "object"

    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Found JSON {kind} but it is malformed: {e}") from e

    raise ValueError(f"No JSON {kind} found in response")


def parse_timestamp(timestamp) -> float:
    """
    Parse a timestamp in various formats to seconds.

    Supports:
    - float/int (already in seconds)
    - "MM:SS" format
    - "HH:MM:SS" format
    - float string (e.g., "12.5")

    Args:
        timestamp: Timestamp in any supported format.

    Returns:
        Time in seconds as a float.

    Raises:
        ValueError: If the timestamp cannot be parsed.
    """
    if isinstance(timestamp, (int, float)):
        return float(timestamp)

    if isinstance(timestamp, str):
        if ":" in timestamp:
            parts = timestamp.split(":")
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        return float(timestamp)

    raise ValueError(f"Cannot parse timestamp: {timestamp}")
