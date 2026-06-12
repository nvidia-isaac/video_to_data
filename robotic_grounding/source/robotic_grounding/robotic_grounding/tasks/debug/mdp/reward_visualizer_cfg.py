# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the Reward Visualizer GUI."""

from __future__ import annotations

from dataclasses import field

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


@configclass
class RewardVisualizerCfg(ActionTermCfg):
    """Configuration for the reward visualization GUI.

    This visualizer displays real-time reward values for each reward term,
    allowing users to see how rewards change as the robot/object state changes.

    Users can register specific reward terms to visualize, or show all terms.

    Attributes:
        reward_terms: List of reward term names to visualize. Empty list shows all.
        exclude_terms: Terms to exclude from visualization when showing all.
        show_total_reward: Whether to show the total summed reward.
        show_weights: Whether to display the weight multiplier for each term.
        show_episode_sum: Whether to show cumulative episode rewards.
        env_index: Which environment index to visualize.
        enable_history_plot: Whether to show time series plot of total reward.
        history_length: Number of steps to keep in history buffer.
    """

    class_type: type[ActionTerm] = field(
        default_factory=lambda: _get_reward_visualizer_class()
    )
    """The class type for this action term."""

    # Need a dummy asset_name to satisfy ActionTermCfg base class
    asset_name: str = "robot"
    """Dummy asset name (required by base class but not used)."""

    # --- Reward Term Selection ---
    reward_terms: list[str] = field(default_factory=list)
    """List of reward term names to visualize. Empty list shows all terms."""

    exclude_terms: list[str] = field(default_factory=list)
    """Terms to exclude from visualization (useful when showing all)."""

    # --- Display Options ---
    show_total_reward: bool = True
    """Whether to show the total (summed) reward at the top. Defaults to True."""

    show_weights: bool = True
    """Whether to show the weight multiplier for each term. Defaults to True."""

    show_episode_sum: bool = True
    """Whether to show cumulative episode reward for each term. Defaults to True."""

    env_index: int = 0
    """Which environment index to visualize. Defaults to 0."""

    # --- Time Series Plot ---
    enable_history_plot: bool = True
    """Whether to enable time series plot of total reward. Defaults to True."""

    history_length: int = 200
    """Number of steps to keep in history buffer. Defaults to 200."""

    # --- GUI Settings ---
    gui_window_title: str = "Reward Monitor"
    """Title of the DearPyGui window. Defaults to 'Reward Monitor'."""

    gui_window_width: int = 500
    """Width of the GUI window in pixels. Defaults to 500."""

    gui_window_height: int = 450
    """Height of the GUI window in pixels. Defaults to 450."""

    # --- Update Settings ---
    update_interval: int = 1
    """Update GUI every N simulation steps. Defaults to 1 (every step)."""


def _get_reward_visualizer_class() -> type[ActionTerm]:
    """Lazy import to avoid circular dependency."""
    from .reward_visualizer import RewardVisualizer  # noqa: PLC0415

    return RewardVisualizer
