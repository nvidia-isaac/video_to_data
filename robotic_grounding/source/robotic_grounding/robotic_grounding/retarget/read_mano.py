# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Any, Dict

import numpy as np
import torch
from judo.visualizers.model import add_mesh
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer

from robotic_grounding.retarget import BODY_MODELS_DIR
from robotic_grounding.retarget.params import (
    MANO_JOINTS_ORDER,
    TRANSFORMS_TO_JOINTS,
)
from robotic_grounding.retarget.utils import quat_from_matrix

"""Reference:
https://github.com/lixiny/manotorch/blob/2f6a701e76ee544bbeac1d3e72c628061f585a6c/scripts/simple_app.py
"""


class MANO:
    """MANO model class to process MANO model and visualize."""

    def __init__(
        self,
        gender: str = "neutral",
        device: torch.device | None = None,
        flat_hand_mean: bool = True,
        center_idx: int | None = None,
    ) -> None:
        """
        Initialize the MANO model.

        Args:
            gender: str, "neutral" or "male" or "female"
            device: torch.device | None, the device to use for the model
            flat_hand_mean: bool, whether to use the flat hand mean for MANO
            center_idx: int | None, index of center joint for MANO (e.g. 0 for wrist).
                If None, no joint centering is applied.
        """
        self.device = device if device is not None else torch.device("cpu")
        mano_assets_root = str(BODY_MODELS_DIR / "mano")

        # Right hand
        self.right_mano_layer = ManoLayer(
            use_pca=False,
            side="right",
            gender=gender,
            center_idx=center_idx,
            mano_assets_root=mano_assets_root,
            flat_hand_mean=flat_hand_mean,
        ).to(self.device)
        self.right_axis_layer = AxisLayerFK(
            side=self.right_mano_layer.side,
            mano_assets_root=mano_assets_root,
        ).to(self.device)

        # Left hand
        self.left_mano_layer = ManoLayer(
            use_pca=False,
            side="left",
            gender=gender,
            center_idx=center_idx,
            mano_assets_root=mano_assets_root,
            flat_hand_mean=flat_hand_mean,
        ).to(self.device)
        self.left_axis_layer = AxisLayerFK(
            side=self.left_mano_layer.side,
            mano_assets_root=str(BODY_MODELS_DIR / "mano"),
        ).to(self.device)

        # Faces
        self.right_faces: (
            torch.Tensor
        ) = self.right_mano_layer.get_mano_closed_faces().to(
            self.device
        )  # (1552, 3)
        self.left_faces: torch.Tensor = self.left_mano_layer.get_mano_closed_faces().to(
            self.device
        )  # (1552, 3)

    def forward(
        self,
        side: str,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
        finger_pose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MANO model.

        Args:
            side: str, "right" or "left"
            betas: torch.Tensor, (B, 10) or (10)
            global_orient: torch.Tensor, (B, 3)
            transl: torch.Tensor, (B, 3)
            finger_pose: torch.Tensor, (B, 45) or (B, 15, 3)
        """
        # 1. Check input shapes
        assert betas.ndim in {1, 2}, "Betas must be of shape (B, 10) or (10)"
        assert (
            global_orient.ndim == 2 and global_orient.shape[1] == 3
        ), "Global orient must be of shape (B, 3)"
        assert (
            transl.ndim == 2 and transl.shape[1] == 3
        ), "Transl must be of shape (B, 3)"
        assert finger_pose.ndim in {
            2,
            3,
        }, "Hand pose must be of shape (B, 45) or (B, 15, 3)"

        betas = betas.to(self.device)
        global_orient = global_orient.to(self.device)
        transl = transl.to(self.device)
        finger_pose = finger_pose.to(self.device)

        # 2. Expand betas and reshape hand pose
        B = len(global_orient)
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).expand(B, -1)
        finger_pose = finger_pose.reshape(B, 45)
        hand_pose = torch.cat((global_orient, finger_pose), dim=-1)
        transl = transl.reshape(B, 1, 3)

        # 3. Forward pass of the MANO model
        mano_layer = self.right_mano_layer if side == "right" else self.left_mano_layer
        axis_layer = self.right_axis_layer if side == "right" else self.left_axis_layer

        mano_results = mano_layer(
            pose_coeffs=hand_pose,  # (B, 48)
            betas=betas,  # (B, 10)
        )

        # 4. Apply translation
        joints = mano_results.joints + transl
        vertices = mano_results.verts + transl
        T_g_p = mano_results.transforms_abs
        T_g_p[:, :, :3, 3] += transl

        # 5. Compute joint rotations
        T_g_a, _, _ = axis_layer(T_g_p)  # (B, 16, 4, 4)
        axes_loc = (
            torch.eye(3).reshape(1, 1, 3, 3).repeat(B, 16, 1, 1).to(vertices.device)
        )  # (B, 16, 3, 3)
        transforms_rotation_wo_tips = torch.matmul(
            T_g_a[:, :, :3, :3], axes_loc
        )  # (B, 16, 3, 3)

        # 6. Add rotations to joints and convert to quaternions
        joints_rotation_matrices = transforms_rotation_wo_tips[
            :, TRANSFORMS_TO_JOINTS
        ]  # (B, 21, 3, 3)
        joints_wxyz = quat_from_matrix(joints_rotation_matrices)  # (B, 21, 4)

        faces = self.right_faces if side == "right" else self.left_faces  # (1538, 3)
        return {
            "joints": joints,  # (B, 21, 3)
            "joints_wxyz": joints_wxyz,  # (B, 16, 3)
            "vertices": vertices,  # (B, 778, 3)
            "faces": faces,  # (1538, 3)
        }

    def visualize(
        self,
        viser_server: Any,
        side: str,
        vertices: torch.Tensor | np.ndarray | None = None,
        faces: torch.Tensor | np.ndarray | None = None,
        joints: torch.Tensor | np.ndarray | None = None,
        joints_wxyz: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """
        Visualize the MANO model.

        Args:
            viser_server: Any
            side: str, "right" or "left"
            vertices: torch.Tensor | np.ndarray, (778, 3)
            faces: torch.Tensor | np.ndarray, (1538, 3)
            joints: torch.Tensor | np.ndarray, (21, 3)
            joints_wxyz: torch.Tensor | np.ndarray, (21, 4)
        """
        if vertices is not None and faces is not None:
            vertices = (
                vertices.cpu().numpy()
                if isinstance(vertices, torch.Tensor)
                else vertices
            )
            faces = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces

            add_mesh(
                viser_server,
                f"/mano/{side}/hand",
                vertices=vertices,
                faces=faces,
                pos=np.array([0, 0, 0]),
                quat=np.array([1, 0, 0, 0]),
                rgba=np.array([255, 219, 172, 180]),
            )

        if joints is not None and joints_wxyz is not None:
            joints = (
                joints.cpu().numpy() if isinstance(joints, torch.Tensor) else joints
            )
            joints_wxyz = (
                joints_wxyz.cpu().numpy()
                if isinstance(joints_wxyz, torch.Tensor)
                else joints_wxyz
            )
            if not hasattr(self, f"viser_mano_{side}_joints_handles"):
                setattr(self, f"viser_mano_{side}_joints_handles", {})
                handles = getattr(self, f"viser_mano_{side}_joints_handles")
                for joint_idx, joint_name in enumerate(MANO_JOINTS_ORDER):
                    handles[joint_name] = viser_server.scene.add_frame(
                        f"/mano/{side}/joints/{joint_name}",
                        position=joints[joint_idx],
                        wxyz=joints_wxyz[joint_idx],
                        axes_length=0.015,
                        axes_radius=0.0005,
                    )
            else:
                handles = getattr(self, f"viser_mano_{side}_joints_handles")
                for joint_idx, joint_name in enumerate(MANO_JOINTS_ORDER):
                    handles[joint_name].position = joints[joint_idx]
                    handles[joint_name].wxyz = joints_wxyz[joint_idx]
