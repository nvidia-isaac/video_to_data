# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Reward Visualizer GUI for real-time reward monitoring."""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .reward_visualizer_cfg import RewardVisualizerCfg


class RewardVisualizer(ActionTerm):
    """Reward visualizer GUI action.

    This action term displays real-time reward values via a DearPyGui window.
    It shows individual reward term values, weights, and cumulative episode sums.
    Optionally displays a time series plot of total reward history.

    Features:
    - Bar chart of individual reward term values with color coding
    - Positive rewards shown in green, negative in red
    - Time series plot of total reward over time
    - Display of weights and episode cumulative sums

    Usage:
        Add this action term to visualize rewards during interactive debugging.
        The visualizer reads from the environment's reward manager.
    """

    cfg: RewardVisualizerCfg

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def __init__(self, cfg: RewardVisualizerCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the reward visualizer.

        Args:
            cfg: Configuration for the visualizer.
            env: The environment instance.
        """
        # Store config and environment (don't call super().__init__)
        self.cfg = cfg
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs

        # Reward manager reference (lazy init after load_managers)
        self._reward_manager = None
        self._initialized = False

        # Visible terms (resolved after reward manager is available)
        self._visible_terms: list[str] = []
        self._term_indices: list[int] = []

        # History buffer for time series plot
        self._reward_history: deque = deque(maxlen=cfg.history_length)
        self._step_count = 0

        # Thread-safe lock
        self._lock = threading.Lock()

        # GUI data buffers (updated from main thread, read by GUI thread)
        self._current_rewards: dict[str, float] = {}
        self._current_weights: dict[str, float] = {}
        self._episode_sums: dict[str, float] = {}
        self._total_reward: float = 0.0

        # Launch GUI in a daemon thread
        self._gui_thread = threading.Thread(
            target=self._launch_gui, name="RewardVisualizerGUI", daemon=True
        )
        self._gui_thread.start()

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Dimension of the action term (not used for visualizer)."""
        return 0

    @property
    def device(self) -> str:
        """Device for tensors."""
        return str(self._device)

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return int(self._num_envs)

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions (unused for visualizer)."""
        return torch.empty(0, device=self._device)

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions (unused for visualizer)."""
        return torch.empty(0, device=self._device)

    # ---------------------------------------------------------------------
    # Initialization Helpers
    # ---------------------------------------------------------------------

    def _lazy_init(self) -> None:
        """Initialize reward manager reference and resolve visible terms."""
        if self._initialized:
            return

        # Get reward manager from environment
        if not hasattr(self._env, "reward_manager") or self._env.reward_manager is None:
            return

        self._reward_manager = self._env.reward_manager
        self._resolve_visible_terms()
        self._initialized = True

    def _resolve_visible_terms(self) -> None:
        """Determine which reward terms to display based on config."""
        assert self._reward_manager is not None, "Reward manager not initialized"

        all_terms = self._reward_manager.active_terms

        if self.cfg.reward_terms:
            # User specified specific terms - filter to valid ones
            self._visible_terms = [t for t in self.cfg.reward_terms if t in all_terms]
        else:
            # Show all terms except excluded ones
            self._visible_terms = [
                t for t in all_terms if t not in self.cfg.exclude_terms
            ]

        # Get indices for efficient lookup
        self._term_indices = [all_terms.index(t) for t in self._visible_terms]

        # Initialize weights from term configs
        for term_name in self._visible_terms:
            term_cfg = self._reward_manager.get_term_cfg(term_name)
            self._current_weights[term_name] = term_cfg.weight

    # ---------------------------------------------------------------------
    # GUI Implementation
    # ---------------------------------------------------------------------

    def _launch_gui(self) -> None:
        """Create the DearPyGui window with reward visualization."""
        import time  # noqa: PLC0415

        import dearpygui.dearpygui as dpg  # noqa: PLC0415

        # Wait for other GUIs to initialize
        time.sleep(1.0)

        # Check if DearPyGui context already exists
        try:
            context_exists = dpg.is_dearpygui_running()
        except Exception:
            context_exists = False

        owns_context = False
        if not context_exists:
            try:
                dpg.create_context()
                dpg.create_viewport(
                    title="Debug Controller",
                    width=600,
                    height=1400,
                )
                owns_context = True
            except Exception:
                pass

        # Create color themes for positive/negative rewards
        with dpg.theme() as positive_theme:
            with dpg.theme_component(dpg.mvProgressBar):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (100, 200, 100, 255))

        with dpg.theme() as negative_theme:
            with dpg.theme_component(dpg.mvProgressBar):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (200, 100, 100, 255))

        # Use unique window tag
        window_tag = "reward_visualizer_window"

        # Store UI element tags for updates
        total_reward_tag = None
        reward_bar_tags: dict[str, int] = {}
        reward_value_tags: dict[str, int] = {}
        episode_sum_tags: dict[str, int] = {}
        history_plot_tag = None
        history_x_data = []
        history_y_data = []

        with dpg.window(
            label=self.cfg.gui_window_title,
            tag=window_tag,
            width=self.cfg.gui_window_width,
            height=self.cfg.gui_window_height,
            pos=(10, 750),
        ):
            dpg.add_text("Real-time reward visualization")
            dpg.add_text(f"Environment index: {self.cfg.env_index}")
            dpg.add_separator()

            # Total reward display
            if self.cfg.show_total_reward:
                with dpg.group(horizontal=True):
                    dpg.add_text("TOTAL REWARD:")
                    total_reward_tag = dpg.add_text("0.0000", color=(255, 255, 100))

            dpg.add_separator()
            dpg.add_text("REWARD TERMS (per step)")

            # Create header row
            if self.cfg.show_weights and self.cfg.show_episode_sum:
                dpg.add_text(
                    "Name                    Weight    Value         Episode Sum"
                )
            elif self.cfg.show_weights:
                dpg.add_text("Name                    Weight    Value")
            elif self.cfg.show_episode_sum:
                dpg.add_text(
                    "Name                              Value         Episode Sum"
                )
            else:
                dpg.add_text("Name                              Value")

            # Placeholder for reward bars (will be populated after init)
            reward_container_tag = dpg.add_group(tag="reward_bars_container")

            dpg.add_separator()

            # History plot
            if self.cfg.enable_history_plot:
                dpg.add_text("REWARD HISTORY")
                with dpg.plot(label="Total Reward Over Time", height=150, width=-1):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                    with dpg.plot_axis(dpg.mvYAxis, label="Reward"):
                        history_plot_tag = dpg.add_line_series([], [], label="Total")

        # Run GUI loop
        if owns_context:
            dpg.setup_dearpygui()
            dpg.show_viewport()

        # Flag to track if reward bars have been created
        bars_created = False

        while True:
            try:
                if owns_context and not dpg.is_dearpygui_running():
                    break
                if not owns_context:
                    try:
                        if not dpg.is_dearpygui_running():
                            break
                    except Exception:
                        break

                # Update reward data from buffers
                with self._lock:
                    current_rewards = self._current_rewards.copy()
                    current_weights = self._current_weights.copy()
                    episode_sums = self._episode_sums.copy()
                    total_reward = self._total_reward
                    history_list = list(self._reward_history)

                # Create reward bars if not yet created and we have terms
                if not bars_created and self._visible_terms:
                    with dpg.group(parent=reward_container_tag):
                        for term_name in self._visible_terms:
                            with dpg.group(horizontal=True):
                                # Term name (truncated)
                                display_name = term_name[:20].ljust(20)
                                dpg.add_text(display_name)

                                # Weight
                                if self.cfg.show_weights:
                                    weight = current_weights.get(term_name, 0.0)
                                    dpg.add_text(f"{weight:7.3f}")

                                # Progress bar for value
                                bar_tag = dpg.add_progress_bar(
                                    default_value=0.5,
                                    width=100,
                                )
                                reward_bar_tags[term_name] = bar_tag

                                # Value text
                                value_tag = dpg.add_text("  0.0000")
                                reward_value_tags[term_name] = value_tag

                                # Episode sum
                                if self.cfg.show_episode_sum:
                                    sum_tag = dpg.add_text("    0.00")
                                    episode_sum_tags[term_name] = sum_tag

                    bars_created = True

                # Update total reward
                if total_reward_tag is not None:
                    color = (100, 255, 100) if total_reward >= 0 else (255, 100, 100)
                    dpg.set_value(total_reward_tag, f"{total_reward:+.4f}")
                    dpg.configure_item(total_reward_tag, color=color)

                # Update reward bars and values
                for term_name in self._visible_terms:
                    if term_name not in reward_bar_tags:
                        continue

                    value = current_rewards.get(term_name, 0.0)
                    ep_sum = episode_sums.get(term_name, 0.0)

                    # Normalize value for progress bar (0-1 range)
                    # Use sigmoid-like scaling for visualization
                    bar_value = (
                        0.5 + 0.5 * (value / (abs(value) + 0.1)) if value != 0 else 0.5
                    )

                    dpg.set_value(reward_bar_tags[term_name], bar_value)
                    dpg.set_value(reward_value_tags[term_name], f"{value:+.4f}")

                    # Apply color theme based on sign
                    if value >= 0:
                        dpg.bind_item_theme(reward_bar_tags[term_name], positive_theme)
                    else:
                        dpg.bind_item_theme(reward_bar_tags[term_name], negative_theme)

                    # Update episode sum
                    if term_name in episode_sum_tags:
                        dpg.set_value(episode_sum_tags[term_name], f"{ep_sum:+8.2f}")

                # Update history plot
                if history_plot_tag is not None and history_list:
                    history_x_data = list(range(len(history_list)))
                    history_y_data = history_list
                    dpg.set_value(history_plot_tag, [history_x_data, history_y_data])

                if owns_context:
                    dpg.render_dearpygui_frame()
                else:
                    time.sleep(0.05)

            except Exception:
                time.sleep(0.1)

        if owns_context:
            dpg.destroy_context()

    # ---------------------------------------------------------------------
    # ActionTerm Interface
    # ---------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        """Process actions (no-op for visualizer)."""
        pass

    def apply_actions(self) -> None:
        """Update reward visualization data."""
        # Lazy initialization
        self._lazy_init()

        if not self._initialized or self._reward_manager is None:
            return

        # Only update at specified interval
        self._step_count += 1
        if self._step_count % self.cfg.update_interval != 0:
            return

        env_idx = self.cfg.env_index

        # Get current step rewards
        step_rewards = self._reward_manager._step_reward[env_idx].cpu()
        total_reward = self._reward_manager._reward_buf[env_idx].item()

        # Update buffers (thread-safe)
        with self._lock:
            self._total_reward = total_reward

            for i, term_name in enumerate(self._visible_terms):
                term_idx = self._term_indices[i]
                self._current_rewards[term_name] = step_rewards[term_idx].item()

                # Get episode sum
                if term_name in self._reward_manager._episode_sums:
                    self._episode_sums[term_name] = self._reward_manager._episode_sums[
                        term_name
                    ][env_idx].item()

            # Update history
            self._reward_history.append(total_reward)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the visualizer (called on environment reset)."""
        # Clear history on reset for the monitored env
        if env_ids is None or self.cfg.env_index in env_ids:
            with self._lock:
                self._reward_history.clear()
                for term_name in self._visible_terms:
                    self._episode_sums[term_name] = 0.0
