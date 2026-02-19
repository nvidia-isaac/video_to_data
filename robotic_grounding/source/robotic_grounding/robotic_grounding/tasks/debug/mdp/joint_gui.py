# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Joint GUI Action for interactive control of robot joints.

Provides a DearPyGui window with separate collapsible sections for
position-controlled and velocity-controlled joints. The GUI runs in a
daemon thread so the physics simulation can continue on the main thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable

import torch
from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .joint_gui_cfg import JointGUIActionCfg


class JointGUIAction(JointPositionAction):
    """Interactive GUI action term for joint-level control.

    Control modes (position-controlled joints only):

    - ``"kinematic"``: Directly write joint state (bypasses PD controller,
      immediate response).
    - ``"pd_target"``: Set PD targets (requires actuator gains, physics-based
      movement).

    Velocity-controlled joints are always driven via
    ``set_joint_velocity_target`` regardless of the control mode.

    Usage:
        Add this action term in the environment's action configuration instead
        of a regular joint-position action. All environment instances receive
        the same joint targets defined in the GUI.
    """

    cfg: JointGUIActionCfg

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, cfg: JointGUIActionCfg, env: ManagerBasedEnv) -> None:
        """Initialize the joint GUI action.

        Args:
            cfg: Configuration for the action term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # Use default_joint_pos (from config init_state) instead of current
        # joint_pos — at __init__ time the sim may not have reset yet.
        self._desired_pos = self._select_joints(
            self._asset.data.default_joint_pos
        ).clone()

        # -- PD gains (only relevant for pd_target mode) ------------------
        if cfg.control_mode == "pd_target" and cfg.show_gains_sliders:
            num_all_joints = self._asset.num_joints
            full_stiffness = torch.zeros(
                self.num_envs, num_all_joints, device=self.device
            )
            full_damping = torch.zeros(
                self.num_envs, num_all_joints, device=self.device
            )
            for actuator in self._asset.actuators.values():
                ids = self._asset.find_joints(actuator.joint_names)[0]
                full_stiffness[:, ids] = actuator.stiffness
                full_damping[:, ids] = actuator.damping

            self._desired_stiffness = self._select_joints(full_stiffness).clone()
            self._desired_damping = self._select_joints(full_damping).clone()
            self._default_stiffness = self._desired_stiffness.clone()
            self._default_damping = self._desired_damping.clone()
        else:
            self._desired_stiffness = None
            self._desired_damping = None
            self._default_stiffness = None
            self._default_damping = None

        # -- Velocity-controlled joints (optional) ------------------------
        vel_names = getattr(cfg, "velocity_joint_names", None)
        if vel_names:
            vel_ids, vel_names_resolved = self._asset.find_joints(vel_names)
            self._velocity_joint_ids = torch.as_tensor(
                vel_ids, device=self.device, dtype=torch.long
            )
            self._velocity_names = list(vel_names_resolved)
            num_vel = len(self._velocity_joint_ids)
            self._desired_vel = torch.zeros(self.num_envs, num_vel, device=self.device)
            self._vel_limits = self._asset.data.soft_joint_vel_limits[
                0, self._velocity_joint_ids
            ].cpu()
        else:
            self._velocity_joint_ids = None
            self._velocity_names = []
            self._desired_vel = None
            self._vel_limits = None

        # -- Position-only joint mask (exclude velocity joints) -----------
        self._pos_col_mask: list[int] | None = None
        self._pos_joint_ids: list[int] | None = None
        if self._velocity_joint_ids is not None:
            vel_set = set(self._velocity_joint_ids.cpu().tolist())
            if isinstance(self._joint_ids, slice):
                all_ids = list(
                    range(
                        self._joint_ids.start or 0,
                        (
                            self._joint_ids.stop
                            if self._joint_ids.stop is not None
                            else self._asset.num_joints
                        ),
                        self._joint_ids.step or 1,
                    )
                )
            else:
                all_ids = list(self._joint_ids)
            self._pos_col_mask = [
                i for i, jid in enumerate(all_ids) if jid not in vel_set
            ]
            self._pos_joint_ids = [all_ids[i] for i in self._pos_col_mask]

        # -- Threading / GUI ----------------------------------------------
        self._lock = threading.Lock()
        self._request_base_reset = False
        self._gui_thread = threading.Thread(
            target=self._launch_gui, name="JointGUI", daemon=True
        )
        self._gui_thread.start()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _launch_gui(self) -> None:
        """Create the DearPyGui window with position and velocity sections."""
        import dearpygui.dearpygui as dpg  # noqa: PLC0415

        dpg.create_context()
        dpg.create_viewport(title="Debug Controller", width=600, height=1200)

        # Estimate window height
        num_joints = len(self._joint_names)
        row_height = 120 if self.cfg.show_gains_sliders else 50
        window_height = min(100 + num_joints * row_height, 1100)

        with dpg.window(
            label="Joint Controller",
            tag="joint_window",
            width=580,
            height=window_height,
            pos=(10, 10),
        ):
            # -- Header ---------------------------------------------------
            mode_text = {
                "kinematic": "Mode: Kinematic (direct state write)",
                "pd_target": "Mode: PD Target (actuator-based)",
            }
            dpg.add_text(mode_text.get(self.cfg.control_mode, "Unknown mode"))
            dpg.add_text(f"Controlling {num_joints} joints")
            dpg.add_separator()

            # Slider tag lists for programmatic updates
            pos_slider_tags: list[int] = []
            stiffness_slider_tags: list[int] = []
            damping_slider_tags: list[int] = []
            effort_bar_tags: list[int] = []
            vel_slider_tags: list[int] = []

            # Effort bar theme
            with dpg.theme() as effort_bar_theme:
                with dpg.theme_component(dpg.mvStemSeries):
                    dpg.add_theme_style(
                        dpg.mvPlotStyleVar_LineWeight,
                        5,
                        category=dpg.mvThemeCat_Plots,
                    )

            # -- Callbacks -------------------------------------------------

            def _reset_joints_cb() -> None:
                """Reset all joints to defaults and request base reset."""
                with self._lock:
                    default_pos = self._select_joints(
                        self._asset.data.default_joint_pos
                    )
                    self._desired_pos[:] = default_pos.clone()
                    for i in range(len(self._joint_names)):
                        dpg.set_value(
                            pos_slider_tags[i],
                            float(self._desired_pos[0, i].cpu()),
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
                    if self._desired_vel is not None:
                        self._desired_vel.zero_()
                        for tag in vel_slider_tags:
                            dpg.set_value(tag, 0.0)
                    self._request_base_reset = True

            def _randomize_joints_cb() -> None:
                """Randomize position-controlled joints within limits."""
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
                            pos_slider_tags[i],
                            float(self._desired_pos[0, i].cpu()),
                        )

            # -- Buttons ---------------------------------------------------
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset to Default", callback=_reset_joints_cb)
                dpg.add_button(label="Randomize", callback=_randomize_joints_cb)
            dpg.add_separator()

            # =============================================================
            # Velocity Control section
            # =============================================================
            if (
                getattr(self.cfg, "velocity_joint_names", None)
                and self._velocity_joint_ids is not None
            ):
                with dpg.collapsing_header(label="Velocity Control", default_open=True):
                    for local_id, joint_name in enumerate(self._velocity_names):
                        limit = float(self._vel_limits[local_id].item())
                        if limit <= 0:
                            limit = 12.0

                        def _make_vel_cb(
                            idx: int,
                        ) -> Callable[[Any, float, Any], None]:
                            def _cb(
                                sender: Any,
                                app_data: float,
                                user_data: Any,  # noqa: ARG001
                            ) -> None:
                                with self._lock:
                                    if self._desired_vel is not None:
                                        self._desired_vel[:, idx] = float(app_data)

                            return _cb

                        tag = dpg.add_slider_float(
                            label=joint_name,
                            min_value=-limit,
                            max_value=limit,
                            default_value=0.0,
                            callback=_make_vel_cb(local_id),
                            format="%.2f",
                            width=400,
                        )
                        vel_slider_tags.append(tag)

            # =============================================================
            # Position Control section
            # =============================================================
            with dpg.collapsing_header(label="Position Control", default_open=True):
                for local_id, joint_name in enumerate(self._joint_names):
                    joint_idx = (
                        local_id
                        if isinstance(self._joint_ids, slice)
                        else self._joint_ids[local_id]
                    )

                    limits = self._asset.data.soft_joint_pos_limits[0, joint_idx].cpu()
                    low, high = float(limits[0]), float(limits[1])
                    current_val = float(self._desired_pos[0, local_id].cpu())

                    def _make_pos_cb(
                        idx: int,
                    ) -> Callable[[Any, float, Any], None]:
                        def _cb(
                            sender: Any,
                            app_data: float,
                            user_data: Any,  # noqa: ARG001
                        ) -> None:
                            with self._lock:
                                self._desired_pos[:, idx] = float(app_data)

                        return _cb

                    pos_slider_tag = dpg.add_slider_float(
                        label=f"[{local_id}] {joint_name}",
                        min_value=low,
                        max_value=high,
                        default_value=current_val,
                        callback=_make_pos_cb(local_id),
                        format="%.3f",
                        width=400,
                    )
                    pos_slider_tags.append(pos_slider_tag)

                    # P/D gain sliders (pd_target mode only)
                    if (
                        self.cfg.show_gains_sliders
                        and self._desired_stiffness is not None
                        and self._desired_damping is not None
                    ):
                        current_stiffness = float(
                            self._desired_stiffness[0, local_id].cpu()
                        )
                        current_damping = float(
                            self._desired_damping[0, local_id].cpu()
                        )

                        def _make_stiffness_cb(
                            idx: int,
                        ) -> Callable[[Any, float, Any], None]:
                            def _cb(
                                sender: Any,
                                app_data: float,
                                user_data: Any,  # noqa: ARG001
                            ) -> None:
                                with self._lock:
                                    if self._desired_stiffness is not None:
                                        self._desired_stiffness[:, idx] = float(
                                            app_data
                                        )

                            return _cb

                        def _make_damping_cb(
                            idx: int,
                        ) -> Callable[[Any, float, Any], None]:
                            def _cb(
                                sender: Any,
                                app_data: float,
                                user_data: Any,  # noqa: ARG001
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

        # -- Event loop ----------------------------------------------------
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
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

    # ------------------------------------------------------------------
    # Overridden ActionTerm methods
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        """Ignore incoming policy actions; GUI values are authoritative."""
        return None

    def apply_actions(self) -> None:
        """Send joint targets from GUI to the articulation."""
        # Handle pending base reset from the GUI thread
        with self._lock:
            do_base_reset = self._request_base_reset
            if do_base_reset:
                self._request_base_reset = False
        if do_base_reset:
            self._reset_robot_base_and_joints()

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

        target_pos = target_pos.to(device=self.device)

        # -- Velocity targets ----------------------------------------------
        if self._velocity_joint_ids is not None and self._desired_vel is not None:
            with self._lock:
                vel = self._desired_vel.clone().to(device=self.device)
            self._asset.set_joint_velocity_target(
                vel, joint_ids=self._velocity_joint_ids
            )

        # -- Position targets (exclude velocity-controlled joints) ---------
        if self._pos_col_mask is not None:
            pos_target = target_pos[:, self._pos_col_mask]
            pos_ids = self._pos_joint_ids
        else:
            pos_target = target_pos
            pos_ids = self._joint_ids

        if self.cfg.control_mode == "pd_target":
            self._asset.set_joint_position_target(pos_target, joint_ids=pos_ids)
        else:  # kinematic
            self._asset.write_joint_state_to_sim(
                pos_target,
                torch.zeros_like(pos_target),
                joint_ids=pos_ids,
            )

        # -- PD gains (pd_target mode only) --------------------------------
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

            full_stiffness = torch.cat(
                [a.stiffness.clone() for a in self._asset.actuators.values()],
                dim=1,
            )
            full_damping = torch.cat(
                [a.damping.clone() for a in self._asset.actuators.values()],
                dim=1,
            )

            full_stiffness[:, joint_ids_tensor] = target_stiffness
            full_damping[:, joint_ids_tensor] = target_damping

            offset = 0
            for actuator in self._asset.actuators.values():
                num_dof = actuator.stiffness.shape[1]
                actuator.stiffness[:] = full_stiffness.narrow(1, offset, num_dof)
                actuator.damping[:] = full_damping.narrow(1, offset, num_dof)
                offset += num_dof

    # ------------------------------------------------------------------
    # Base + joint reset (for GUI "Reset to Default")
    # ------------------------------------------------------------------

    def _reset_robot_base_and_joints(self) -> None:
        """Reset robot root pose/velocity and joints to config defaults."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        init_state = getattr(self._asset.cfg, "init_state", None)
        if init_state is not None:
            init_pos = getattr(init_state, "pos", (0.0, 0.0, 0.0))
            init_rot = getattr(init_state, "rot", (1.0, 0.0, 0.0, 0.0))
            root_pos = self._env.scene.env_origins.clone().to(device=self.device)
            root_pos += torch.tensor(init_pos, dtype=root_pos.dtype, device=self.device)
            root_quat = (
                torch.tensor(init_rot, dtype=root_pos.dtype, device=self.device)
                .unsqueeze(0)
                .expand(self.num_envs, 4)
            )
            self._asset.write_root_pose_to_sim(
                torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
            )
        self._asset.write_root_velocity_to_sim(
            torch.zeros_like(self._asset.data.root_vel_w[env_ids]),
            env_ids=env_ids,
        )
        joint_pos = self._asset.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self._asset.data.joint_vel[env_ids])
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term (called on environment reset)."""
        super().reset(env_ids)
        if env_ids is None or 0 in env_ids:
            with self._lock:
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

    def _select_joints(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select columns for *self._joint_ids* from *tensor*."""
        if isinstance(self._joint_ids, slice):
            slicer = [slice(None)] * tensor.ndim
            slicer[1] = self._joint_ids
            return tensor[tuple(slicer)]

        index_tensor = torch.as_tensor(
            self._joint_ids, dtype=torch.long, device=tensor.device
        )
        return tensor.index_select(1, index_tensor)
