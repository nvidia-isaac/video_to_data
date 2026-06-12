# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified Object Pose GUI Action for interactive control of object base pose."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .object_pose_gui_action_cfg import ObjectPoseGUIActionCfg


class ObjectPoseGUIAction(ActionTerm):
    """Unified object pose GUI action.

    This action term allows controlling an object's 6DoF base pose interactively
    via a DearPyGui window. It supports:

    1. **RigidObject**: Direct pose via `write_root_link_pose_to_sim()`
    2. **Articulation with floating base joints**: Pose via floating base joints
    3. **Articulation without floating base joints**: Direct root pose write

    A separate thread is spawned for the GUI so that the physics simulation can continue
    to run in the main thread.

    The pose is specified as:
    - Position: x, y, z in meters (world frame)
    - Orientation: roll, pitch, yaw in radians (Euler XYZ convention)

    Usage:
        Add this action term in the environment's action configuration. The GUI
        values are authoritative - any RL actions sent to this term are ignored.
    """

    cfg: ObjectPoseGUIActionCfg

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def __init__(self, cfg: ObjectPoseGUIActionCfg, env: ManagerBasedEnv) -> None:
        """Initialize the object pose GUI action.

        Args:
            cfg: Configuration for the action term.
            env: The environment instance.
        """
        # Store config and environment
        self.cfg = cfg
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs

        # Get the object from the scene
        self._object = env.scene[cfg.asset_name]

        # Determine object type and control mode
        self._is_articulated = isinstance(self._object, Articulation)
        self._use_floating_base_joints = (
            self._is_articulated and cfg.position_joint_names is not None
        )

        # Initialize joint ID mappings (None when not using floating base joints)
        self._pos_joint_ids: dict[str, int] | None = None
        self._rot_joint_ids: dict[str, int] | None = None

        # Validate configuration and setup floating base joints if configured
        if self._use_floating_base_joints:
            if cfg.rotation_joint_names is None:
                raise ValueError(
                    "rotation_joint_names must be provided when position_joint_names is set"
                )
            self._setup_floating_base_joints()

        # Initialize desired pose
        self._init_pose_from_object()

        # Store initial/default pose for reset
        self._default_pos = self._desired_pos.clone()
        self._default_euler = self._desired_euler.clone()

        # Thread-safe lock for accessing pose from GUI
        self._lock = threading.Lock()

        # Launch GUI in a daemon thread
        self._gui_thread = threading.Thread(
            target=self._launch_gui,
            name=f"ObjectPoseGUI_{cfg.asset_name}",
            daemon=True,
        )
        self._gui_thread.start()

    def _setup_floating_base_joints(self) -> None:
        """Resolve joint indices for floating base joints."""
        cfg = self.cfg
        # Type narrowing - these are guaranteed non-None when this method is called
        assert cfg.position_joint_names is not None
        assert cfg.rotation_joint_names is not None

        pos_joint_ids: dict[str, int] = {}
        rot_joint_ids: dict[str, int] = {}

        for axis, joint_name in cfg.position_joint_names.items():
            joint_ids, _ = self._object.find_joints(joint_name)
            if len(joint_ids) > 0:
                pos_joint_ids[axis] = joint_ids[0]
            else:
                raise ValueError(
                    f"Joint '{joint_name}' for position axis '{axis}' not found "
                    f"in articulation '{cfg.asset_name}'"
                )

        for axis, joint_name in cfg.rotation_joint_names.items():
            joint_ids, _ = self._object.find_joints(joint_name)
            if len(joint_ids) > 0:
                rot_joint_ids[axis] = joint_ids[0]
            else:
                raise ValueError(
                    f"Joint '{joint_name}' for rotation axis '{axis}' not found "
                    f"in articulation '{cfg.asset_name}'"
                )

        self._pos_joint_ids = pos_joint_ids
        self._rot_joint_ids = rot_joint_ids

    def _init_pose_from_object(self) -> None:
        """Initialize desired pose from object's current/default state."""
        if self._use_floating_base_joints:
            # Type narrowing - these are guaranteed non-None when using floating base
            assert self._pos_joint_ids is not None
            assert self._rot_joint_ids is not None
            # Initialize from default joint positions
            joint_pos = self._object.data.default_joint_pos[0].cpu()
            self._desired_pos = torch.tensor(
                [
                    joint_pos[self._pos_joint_ids["x"]].item(),
                    joint_pos[self._pos_joint_ids["y"]].item(),
                    joint_pos[self._pos_joint_ids["z"]].item(),
                ]
            )
            self._desired_euler = torch.tensor(
                [
                    joint_pos[self._rot_joint_ids["roll"]].item(),
                    joint_pos[self._rot_joint_ids["pitch"]].item(),
                    joint_pos[self._rot_joint_ids["yaw"]].item(),
                ]
            )
        else:
            # Initialize from root link pose (works for both Rigid and Articulation)
            current_pos = self._object.data.root_link_pos_w[0].clone()
            current_quat = self._object.data.root_link_quat_w[0].clone()

            # Convert quaternion to Euler angles for GUI
            roll, pitch, yaw = euler_xyz_from_quat(current_quat.unsqueeze(0))

            self._desired_pos = current_pos.cpu()
            self._desired_euler = torch.tensor([roll.item(), pitch.item(), yaw.item()])

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Dimension of the action term (not used for GUI action)."""
        return 0  # No RL action dimension - GUI is authoritative

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
        """The input/raw actions sent to the term (unused for GUI)."""
        return torch.empty(0, device=self._device)

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions (returns current desired pose)."""
        with self._lock:
            pos = self._desired_pos.clone()
            euler = self._desired_euler.clone()
        return torch.cat([pos, euler])

    # ---------------------------------------------------------------------
    # GUI Implementation
    # ---------------------------------------------------------------------

    def _launch_gui(self) -> None:
        """Create the DearPyGui window with pose control sliders."""
        import dearpygui.dearpygui as dpg  # noqa: PLC0415

        # Wait a bit to let joint GUI initialize first if it exists
        time.sleep(0.5)

        # Check if DearPyGui context already exists (from joint GUI)
        try:
            context_exists = dpg.is_dearpygui_running()
        except Exception:
            context_exists = False

        # Only create context if it doesn't exist
        owns_context = False
        if not context_exists:
            try:
                dpg.create_context()
                dpg.create_viewport(
                    title="Debug Controller",
                    width=600,
                    height=800,
                )
                owns_context = True
            except Exception:
                # Context might have been created by another thread in the meantime
                pass

        # Get limits from config
        pos_limits = self.cfg.position_limits
        rot_limits = self.cfg.rotation_limits

        # Store slider tags for programmatic updates
        pos_slider_tags: dict[str, int] = {}
        euler_slider_tags: dict[str, int] = {}

        # Use unique window tag to avoid conflicts
        window_tag = f"object_pose_window_{self.cfg.asset_name}"

        # Determine control description for GUI
        if self._use_floating_base_joints:
            control_desc = "(floating base joints)"
        elif self._is_articulated:
            control_desc = "(direct root pose)"
        else:
            control_desc = "(rigid object)"

        with dpg.window(
            label=self.cfg.gui_window_title,
            tag=window_tag,
            width=480,
            height=350,
            pos=(10, 450),
        ):
            dpg.add_text(f"Control pose of '{self.cfg.asset_name}' {control_desc}")
            dpg.add_separator()

            # --- Buttons ---
            def _reset_pose_cb() -> None:
                """Reset object to default pose."""
                with self._lock:
                    self._desired_pos[:] = self._default_pos.clone()
                    self._desired_euler[:] = self._default_euler.clone()
                    # Update GUI sliders
                    for i, axis in enumerate(["x", "y", "z"]):
                        dpg.set_value(
                            pos_slider_tags[axis], float(self._desired_pos[i])
                        )
                    for i, axis in enumerate(["roll", "pitch", "yaw"]):
                        dpg.set_value(
                            euler_slider_tags[axis], float(self._desired_euler[i])
                        )

            def _randomize_pose_cb() -> None:
                """Randomize object pose within limits."""
                with self._lock:
                    # Random position
                    for i, axis in enumerate(["x", "y", "z"]):
                        low, high = pos_limits[axis]
                        self._desired_pos[i] = low + (high - low) * torch.rand(1).item()
                        dpg.set_value(
                            pos_slider_tags[axis], float(self._desired_pos[i])
                        )
                    # Random orientation
                    for i, axis in enumerate(["roll", "pitch", "yaw"]):
                        low, high = rot_limits[axis]
                        self._desired_euler[i] = (
                            low + (high - low) * torch.rand(1).item()
                        )
                        dpg.set_value(
                            euler_slider_tags[axis], float(self._desired_euler[i])
                        )

            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset to Default", callback=_reset_pose_cb)
                dpg.add_button(label="Randomize Pose", callback=_randomize_pose_cb)

            dpg.add_separator()

            # --- Position Sliders ---
            dpg.add_text("POSITION (meters)")

            for i, axis in enumerate(["x", "y", "z"]):
                low, high = pos_limits[axis]
                current_val = float(self._desired_pos[i])

                def _pos_slider_cb(
                    sender: int, app_data: float, user_data: int  # noqa: ARG001
                ) -> None:
                    idx = user_data
                    with self._lock:
                        self._desired_pos[idx] = float(app_data)

                slider_tag = dpg.add_slider_float(
                    label=f"{axis.upper()}",
                    min_value=low,
                    max_value=high,
                    default_value=current_val,
                    callback=_pos_slider_cb,
                    user_data=i,
                    format="%.3f m",
                    width=350,
                )
                pos_slider_tags[axis] = slider_tag

            dpg.add_separator()

            # --- Orientation Sliders ---
            dpg.add_text("ORIENTATION (Euler XYZ, radians)")

            for i, axis in enumerate(["roll", "pitch", "yaw"]):
                low, high = rot_limits[axis]
                current_val = float(self._desired_euler[i])

                def _euler_slider_cb(
                    sender: int, app_data: float, user_data: int  # noqa: ARG001
                ) -> None:
                    idx = user_data
                    with self._lock:
                        self._desired_euler[idx] = float(app_data)

                slider_tag = dpg.add_slider_float(
                    label=f"{axis.capitalize()}",
                    min_value=low,
                    max_value=high,
                    default_value=current_val,
                    callback=_euler_slider_cb,
                    user_data=i,
                    format="%.3f rad",
                    width=350,
                )
                euler_slider_tags[axis] = slider_tag

            dpg.add_separator()

            # --- Current Pose Display (read-only) ---
            dpg.add_text("CURRENT POSE (read-only)")
            pos_text_tag = dpg.add_text("Pos: [0.000, 0.000, 0.000]")
            rot_text_tag = dpg.add_text("Rot: [0.000, 0.000, 0.000]")

        # Only run our own event loop if we own the context
        if owns_context:
            # Setup and show
            dpg.setup_dearpygui()
            dpg.show_viewport()

            # Main GUI loop
            while dpg.is_dearpygui_running():
                self._update_pose_display(pos_text_tag, rot_text_tag)
                dpg.render_dearpygui_frame()

            dpg.destroy_context()
        else:
            # Another GUI owns the context - just keep updating our display in a loop
            while True:
                try:
                    if not dpg.is_dearpygui_running():
                        break
                    self._update_pose_display(pos_text_tag, rot_text_tag)
                    time.sleep(0.05)  # Small delay to avoid busy loop
                except Exception:
                    time.sleep(0.1)  # Wait if not ready

    def _update_pose_display(self, pos_text_tag: int, rot_text_tag: int) -> None:
        """Update the current pose display in the GUI."""
        import dearpygui.dearpygui as dpg  # noqa: PLC0415

        try:
            if self._use_floating_base_joints:
                # Type narrowing - these are guaranteed non-None when using floating base
                assert self._pos_joint_ids is not None
                assert self._rot_joint_ids is not None
                # Read from joint positions
                joint_pos = self._object.data.joint_pos[0].cpu()
                current_pos = [
                    joint_pos[self._pos_joint_ids["x"]].item(),
                    joint_pos[self._pos_joint_ids["y"]].item(),
                    joint_pos[self._pos_joint_ids["z"]].item(),
                ]
                current_euler = [
                    joint_pos[self._rot_joint_ids["roll"]].item(),
                    joint_pos[self._rot_joint_ids["pitch"]].item(),
                    joint_pos[self._rot_joint_ids["yaw"]].item(),
                ]
                dpg.set_value(
                    pos_text_tag,
                    f"Pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]",
                )
                dpg.set_value(
                    rot_text_tag,
                    f"Euler: [{current_euler[0]:.3f}, {current_euler[1]:.3f}, "
                    f"{current_euler[2]:.3f}]",
                )
            else:
                # Read from root link pose
                current_pos = self._object.data.root_link_pos_w[0].cpu()
                current_quat = self._object.data.root_link_quat_w[0].cpu()
                dpg.set_value(
                    pos_text_tag,
                    f"Pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]",
                )
                dpg.set_value(
                    rot_text_tag,
                    f"Quat: [{current_quat[0]:.3f}, {current_quat[1]:.3f}, "
                    f"{current_quat[2]:.3f}, {current_quat[3]:.3f}]",
                )
        except Exception:
            pass  # Object may not be ready yet

    # ---------------------------------------------------------------------
    # ActionTerm Interface
    # ---------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        """Ignore incoming RL actions; GUI values are authoritative."""
        pass

    def apply_actions(self) -> None:
        """Apply the GUI-specified pose to the object."""
        if self._use_floating_base_joints:
            self._apply_via_floating_base_joints()
        else:
            self._apply_via_root_pose()

    def _apply_via_floating_base_joints(self) -> None:
        """Apply pose via floating base joints (for Articulation with floating base)."""
        # Type narrowing - these are guaranteed non-None when this method is called
        assert self._pos_joint_ids is not None
        assert self._rot_joint_ids is not None

        with self._lock:
            pos = self._desired_pos.clone()
            euler = self._desired_euler.clone()

        # Convert to device
        pos = pos.to(self._device)
        euler = euler.to(self._device)

        # Build joint position target tensor
        joint_target = self._object.data.joint_pos.clone()

        # Set position joints for all environments
        joint_target[:, self._pos_joint_ids["x"]] = pos[0]
        joint_target[:, self._pos_joint_ids["y"]] = pos[1]
        joint_target[:, self._pos_joint_ids["z"]] = pos[2]

        # Set rotation joints for all environments
        joint_target[:, self._rot_joint_ids["roll"]] = euler[0]
        joint_target[:, self._rot_joint_ids["pitch"]] = euler[1]
        joint_target[:, self._rot_joint_ids["yaw"]] = euler[2]

        # Get the joint IDs for the floating base
        floating_base_joint_ids = list(self._pos_joint_ids.values()) + list(
            self._rot_joint_ids.values()
        )

        if self.cfg.control_mode == "kinematic":
            # Directly write joint positions (immediate effect)
            self._object.write_joint_state_to_sim(
                joint_target[:, floating_base_joint_ids],
                torch.zeros_like(joint_target[:, floating_base_joint_ids]),
                joint_ids=floating_base_joint_ids,
            )
        else:  # pd_target
            # Set joint position targets (follows PD controller)
            self._object.set_joint_position_target(
                joint_target[:, floating_base_joint_ids],
                joint_ids=floating_base_joint_ids,
            )

    def _apply_via_root_pose(self) -> None:
        """Apply pose directly via root link (for Rigid or Articulation without floating base)."""
        with self._lock:
            pos = self._desired_pos.clone()
            euler = self._desired_euler.clone()

        # Convert to device
        pos = pos.to(self._device)
        euler = euler.to(self._device)

        # Convert Euler to quaternion (wxyz format)
        quat = quat_from_euler_xyz(
            euler[0:1],  # roll
            euler[1:2],  # pitch
            euler[2:3],  # yaw
        ).squeeze(0)

        # Build pose tensor [x, y, z, w, qx, qy, qz]
        pose = torch.cat([pos, quat])

        # Expand for all environments
        pose_batch = pose.unsqueeze(0).expand(self._num_envs, -1)

        # Write pose to simulation
        self._object.write_root_link_pose_to_sim(pose_batch)

        # Zero out velocities to prevent drift
        zero_vel = torch.zeros(self._num_envs, 6, device=self._device)
        self._object.write_root_com_velocity_to_sim(zero_vel)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term (called on environment reset).

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        # Sync GUI with current object pose on reset
        if env_ids is None or 0 in env_ids:
            with self._lock:
                if self._use_floating_base_joints:
                    # Type narrowing - these are guaranteed non-None when using floating base
                    assert self._pos_joint_ids is not None
                    assert self._rot_joint_ids is not None
                    # Reset from default joint positions
                    joint_pos = self._object.data.default_joint_pos[0].cpu()

                    self._desired_pos[0] = joint_pos[self._pos_joint_ids["x"]].item()
                    self._desired_pos[1] = joint_pos[self._pos_joint_ids["y"]].item()
                    self._desired_pos[2] = joint_pos[self._pos_joint_ids["z"]].item()

                    self._desired_euler[0] = joint_pos[
                        self._rot_joint_ids["roll"]
                    ].item()
                    self._desired_euler[1] = joint_pos[
                        self._rot_joint_ids["pitch"]
                    ].item()
                    self._desired_euler[2] = joint_pos[
                        self._rot_joint_ids["yaw"]
                    ].item()
                else:
                    # Reset from current root link pose
                    current_pos = self._object.data.root_link_pos_w[0].cpu()
                    current_quat = self._object.data.root_link_quat_w[0].cpu()
                    roll, pitch, yaw = euler_xyz_from_quat(current_quat.unsqueeze(0))

                    self._desired_pos[:] = current_pos
                    self._desired_euler[0] = roll.item()
                    self._desired_euler[1] = pitch.item()
                    self._desired_euler[2] = yaw.item()

                # Update defaults
                self._default_pos[:] = self._desired_pos.clone()
                self._default_euler[:] = self._desired_euler.clone()


# Backwards compatibility alias
ArticulatedObjectPoseGUIAction = ObjectPoseGUIAction
RigidObjectPoseGUIAction = ObjectPoseGUIAction
