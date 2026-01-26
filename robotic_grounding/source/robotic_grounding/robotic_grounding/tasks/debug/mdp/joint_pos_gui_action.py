# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Joint Position GUI Action for interactive control of robot joints."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable

import torch
from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .joint_pos_gui_action_cfg import JointPositionGUIActionCfg


class JointPositionGUIAction(JointPositionAction):
    """Joint position GUI action.

    This action term allows controlling the robot's joint positions interactively via a
    DearPyGui window. A separate thread is spawned for the GUI so that the physics
    simulation can continue to run in the main thread.

    Control modes:
    - "kinematic": Directly write joint state (bypasses PD controller, immediate response)
    - "pd_target": Set PD targets (requires actuator gains, physics-based movement)

    Usage:
        Add this action term in the environment's action configuration instead of a
        regular joint-position action. All environment instances will receive the
        same joint targets defined in the GUI.
    """

    cfg: JointPositionGUIActionCfg

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def __init__(self, cfg: JointPositionGUIActionCfg, env: ManagerBasedEnv) -> None:
        """Initialize the joint position GUI action.

        Args:
            cfg: Configuration for the action term.
            env: The environment instance.
        """
        # Initialize parent class (resolves joints, etc.)
        super().__init__(cfg, env)

        # Use default_joint_pos (from config init_state) instead of current joint_pos
        # At __init__ time, joint_pos might not have been reset to config values yet
        self._desired_pos = self._select_joints(
            self._asset.data.default_joint_pos
        ).clone()

        # Only initialize gains if needed (only relevant for pd_target mode)
        if cfg.control_mode == "pd_target" and cfg.show_gains_sliders:
            # Desired PD gains for the robot actuators
            # Build full gains tensor for all joints, then select the subset we need
            num_all_joints = self._asset.num_joints
            full_stiffness = torch.zeros(
                self.num_envs, num_all_joints, device=self.device
            )
            full_damping = torch.zeros(
                self.num_envs, num_all_joints, device=self.device
            )

            # Populate gains from each actuator in the correct joint positions
            for actuator in self._asset.actuators.values():
                actuator_joint_ids = self._asset.find_joints(actuator.joint_names)[0]
                full_stiffness[:, actuator_joint_ids] = actuator.stiffness
                full_damping[:, actuator_joint_ids] = actuator.damping

            # Select only the joints we care about
            self._desired_stiffness = self._select_joints(full_stiffness).clone()
            self._desired_damping = self._select_joints(full_damping).clone()

            # Store default gains for reset
            self._default_stiffness = self._desired_stiffness.clone()
            self._default_damping = self._desired_damping.clone()
        else:
            self._desired_stiffness = None
            self._desired_damping = None
            self._default_stiffness = None
            self._default_damping = None

        # Thread-safe lock for accessing ``_desired_pos`` from GUI
        self._lock = threading.Lock()
        # Launch GUI in a daemon thread so that it quits automatically with Python
        self._gui_thread = threading.Thread(
            target=self._launch_gui, name="JointGUI", daemon=True
        )
        self._gui_thread.start()

    # ---------------------------------------------------------------------
    # GUI helpers
    # ---------------------------------------------------------------------

    def _launch_gui(self) -> None:
        """Create the DearPyGui window and sliders for each joint."""
        import dearpygui.dearpygui as dpg  # noqa: PLC0415

        # Create context and viewport
        dpg.create_context()
        dpg.create_viewport(title="Debug Controller", width=600, height=1200)

        # Calculate window height based on number of joints and whether gains are shown
        num_joints = len(self._joint_names)
        row_height = 120 if self.cfg.show_gains_sliders else 50
        window_height = min(100 + num_joints * row_height, 1100)

        with dpg.window(
            label="Joint Position Controller",
            tag="joint_window",
            width=580,
            height=window_height,
            pos=(10, 10),
        ):
            # Show control mode info
            mode_text = {
                "kinematic": "Mode: Kinematic (direct state write)",
                "pd_target": "Mode: PD Target (actuator-based)",
            }
            dpg.add_text(mode_text.get(self.cfg.control_mode, "Unknown mode"))
            dpg.add_text(f"Controlling {num_joints} joints")
            dpg.add_separator()

            # Store slider tags for programmatic updates
            pos_slider_tags: list[int] = []
            stiffness_slider_tags: list[int] = []
            damping_slider_tags: list[int] = []
            effort_bar_tags: list[int] = []

            # Create a theme for effort plot
            with dpg.theme() as effort_bar_theme:
                with dpg.theme_component(dpg.mvStemSeries):
                    dpg.add_theme_style(
                        dpg.mvPlotStyleVar_LineWeight, 5, category=dpg.mvThemeCat_Plots
                    )

            def _reset_joints_cb() -> None:
                """Reset all joints to their default positions and gains."""
                with self._lock:
                    default_pos = self._select_joints(
                        self._asset.data.default_joint_pos
                    )
                    self._desired_pos[:] = default_pos.clone()
                    for i in range(len(self._joint_names)):
                        dpg.set_value(
                            pos_slider_tags[i], float(self._desired_pos[0, i].cpu())
                        )
                    if (
                        self._desired_stiffness is not None
                        and self._desired_damping is not None
                        and self._default_stiffness is not None
                        and self._default_damping is not None
                    ):
                        self._desired_stiffness[:] = self._default_stiffness.clone()
                        self._desired_damping[:] = self._default_damping.clone()
                        for i in range(len(self._joint_names)):
                            dpg.set_value(
                                stiffness_slider_tags[i],
                                float(self._desired_stiffness[0, i].cpu()),
                            )
                            dpg.set_value(
                                damping_slider_tags[i],
                                float(self._desired_damping[0, i].cpu()),
                            )

            def _randomize_joints_cb() -> None:
                """Randomize all joints within their limits."""
                with self._lock:
                    limits = self._select_joints(
                        self._asset.data.soft_joint_pos_limits[0].T
                    ).T.cpu()
                    low = limits[:, 0]
                    high = limits[:, 1]
                    random_pos = low + (high - low) * torch.rand_like(low)
                    self._desired_pos[:] = random_pos.unsqueeze(0)
                    for i in range(len(self._joint_names)):
                        dpg.set_value(
                            pos_slider_tags[i], float(self._desired_pos[0, i].cpu())
                        )

            # Add buttons
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset to Default", callback=_reset_joints_cb)
                dpg.add_button(label="Randomize", callback=_randomize_joints_cb)
            dpg.add_separator()

            # Create sliders for each joint
            for local_id, joint_name in enumerate(self._joint_names):
                # Resolve the global joint index
                joint_idx = (
                    local_id
                    if isinstance(self._joint_ids, slice)
                    else self._joint_ids[local_id]
                )

                # Fetch soft limits
                limits = self._asset.data.soft_joint_pos_limits[0, joint_idx].cpu()
                low, high = float(limits[0]), float(limits[1])
                current_val = float(self._desired_pos[0, local_id].cpu())

                # -- Position slider callback
                def _make_slider_cb(
                    idx: int,
                ) -> Callable[[Any, float, Any], None]:
                    def _cb(
                        sender: Any, app_data: float, user_data: Any  # noqa: ARG001
                    ) -> None:
                        with self._lock:
                            self._desired_pos[:, idx] = float(app_data)

                    return _cb

                pos_slider_tag = dpg.add_slider_float(
                    label=f"[{local_id}] {joint_name}",
                    min_value=low,
                    max_value=high,
                    default_value=current_val,
                    callback=_make_slider_cb(local_id),
                    format="%.3f",
                    width=400,
                )
                pos_slider_tags.append(pos_slider_tag)

                # -- P/D Gain sliders (if enabled)
                if (
                    self.cfg.show_gains_sliders
                    and self._desired_stiffness is not None
                    and self._desired_damping is not None
                ):
                    current_stiffness = float(
                        self._desired_stiffness[0, local_id].cpu()
                    )
                    current_damping = float(self._desired_damping[0, local_id].cpu())

                    def _make_stiffness_cb(
                        idx: int,
                    ) -> Callable[[Any, float, Any], None]:
                        def _cb(
                            sender: Any, app_data: float, user_data: Any  # noqa: ARG001
                        ) -> None:
                            with self._lock:
                                if self._desired_stiffness is not None:
                                    self._desired_stiffness[:, idx] = float(app_data)

                        return _cb

                    def _make_damping_cb(
                        idx: int,
                    ) -> Callable[[Any, float, Any], None]:
                        def _cb(
                            sender: Any, app_data: float, user_data: Any  # noqa: ARG001
                        ) -> None:
                            with self._lock:
                                if self._desired_damping is not None:
                                    self._desired_damping[:, idx] = float(app_data)

                        return _cb

                    stiffness_tag = dpg.add_slider_float(
                        label="P-Gain",
                        min_value=0.0,
                        max_value=self.cfg.max_stiffness,
                        default_value=current_stiffness,
                        callback=_make_stiffness_cb(local_id),
                        format="%.1f",
                        indent=20,
                        width=380,
                    )
                    stiffness_slider_tags.append(stiffness_tag)

                    damping_tag = dpg.add_slider_float(
                        label="D-Gain",
                        min_value=0.0,
                        max_value=self.cfg.max_damping,
                        default_value=current_damping,
                        callback=_make_damping_cb(local_id),
                        format="%.1f",
                        indent=20,
                        width=380,
                    )
                    damping_slider_tags.append(damping_tag)

                    # Effort visualization
                    effort_limit = torch.cat(
                        [v.effort_limit for v in self._asset.actuators.values()],
                        dim=-1,
                    )[0, joint_idx].item()
                    with dpg.plot(
                        no_title=True,
                        no_menus=True,
                        no_box_select=True,
                        no_mouse_pos=True,
                        height=60,
                        width=100,
                    ):
                        dpg.add_plot_axis(
                            dpg.mvXAxis,
                            no_gridlines=True,
                            no_tick_marks=True,
                            no_tick_labels=True,
                        )
                        dpg.set_axis_limits(dpg.last_item(), -1, 1)
                        with dpg.plot_axis(
                            dpg.mvYAxis,
                            no_gridlines=True,
                            no_tick_marks=True,
                            no_tick_labels=True,
                        ) as y_axis:
                            dpg.set_axis_limits(y_axis, -effort_limit, effort_limit)
                            bar_tag = dpg.add_stem_series([0.0], [0.0])
                            dpg.bind_item_theme(bar_tag, effort_bar_theme)
                            effort_bar_tags.append(bar_tag)

                dpg.add_separator()

        # Start event loop
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            # Update effort bars if they exist
            if effort_bar_tags:
                with self._lock:
                    applied_effort = self._select_joints(
                        self._asset.data.applied_torque
                    ).clone()
                applied_effort_cpu = applied_effort.cpu()
                for i in range(len(effort_bar_tags)):
                    effort_val = float(applied_effort_cpu[0, i])
                    dpg.set_value(effort_bar_tags[i], [[0.0], [effort_val]])

            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    # ---------------------------------------------------------------------
    # Overridden ActionTerm methods
    # ---------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        """Ignore incoming policy actions; GUI values are authoritative."""
        return None

    def apply_actions(self) -> None:
        """Send joint targets from GUI to the articulation based on control mode."""
        with self._lock:
            target_pos = self._desired_pos.clone()
            if (
                self._desired_stiffness is not None
                and self._desired_damping is not None
            ):
                target_stiffness = self._desired_stiffness.clone()
                target_damping = self._desired_damping.clone()
            else:
                target_stiffness = None
                target_damping = None

        # Ensure tensor is on the correct device
        target_pos = target_pos.to(device=self.device)

        # Apply based on control mode
        if self.cfg.control_mode == "pd_target":
            # Set joint position targets for PD controller
            self._asset.set_joint_position_target(target_pos, joint_ids=self._joint_ids)
        else:  # kinematic
            # Directly write joint state for immediate kinematic control
            self._asset.write_joint_state_to_sim(
                target_pos,
                torch.zeros_like(target_pos),  # Zero velocity
                joint_ids=self._joint_ids,
            )

        # Apply PD gains if enabled (only relevant for pd_target mode)
        if (
            target_stiffness is not None
            and target_damping is not None
            and self.cfg.control_mode == "pd_target"
        ):
            target_stiffness = target_stiffness.to(device=self.device)
            target_damping = target_damping.to(device=self.device)

            if isinstance(self._joint_ids, slice):
                joint_ids_tensor = torch.arange(
                    self._joint_ids.start or 0,
                    self._joint_ids.stop,
                    self._joint_ids.step or 1,
                    device=self.device,
                )
            else:
                joint_ids_tensor = torch.tensor(self._joint_ids, device=self.device)

            # Build full gains tensors
            full_stiffness = torch.cat(
                [
                    actuator.stiffness.clone()
                    for actuator in self._asset.actuators.values()
                ],
                dim=1,
            )
            full_damping = torch.cat(
                [
                    actuator.damping.clone()
                    for actuator in self._asset.actuators.values()
                ],
                dim=1,
            )

            # Update selected joints
            full_stiffness[:, joint_ids_tensor] = target_stiffness
            full_damping[:, joint_ids_tensor] = target_damping

            # Distribute back to actuators
            offset = 0
            for actuator in self._asset.actuators.values():
                num_dof = actuator.stiffness.shape[1]
                actuator.stiffness[:] = full_stiffness.narrow(1, offset, num_dof)
                actuator.damping[:] = full_damping.narrow(1, offset, num_dof)
                offset += num_dof

    # ---------------------------------------------------------------------
    # Misc helpers
    # ---------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term (called on environment reset)."""
        super().reset(env_ids)
        if env_ids is None or 0 in env_ids:
            with self._lock:
                # Use default_joint_pos to reset to configured initial state
                self._desired_pos[...] = self._select_joints(
                    self._asset.data.default_joint_pos
                ).clone()
                if (
                    self._desired_stiffness is not None
                    and self._desired_damping is not None
                    and self._default_stiffness is not None
                    and self._default_damping is not None
                ):
                    self._desired_stiffness[...] = self._default_stiffness.clone()
                    self._desired_damping[...] = self._default_damping.clone()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _select_joints(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely select the joint columns specified by *self._joint_ids* from *tensor*."""
        if isinstance(self._joint_ids, slice):
            slicer = [slice(None)] * tensor.ndim
            slicer[1] = self._joint_ids
            return tensor[tuple(slicer)]

        index_tensor = torch.as_tensor(
            self._joint_ids, dtype=torch.long, device=tensor.device
        )
        return tensor.index_select(1, index_tensor)
