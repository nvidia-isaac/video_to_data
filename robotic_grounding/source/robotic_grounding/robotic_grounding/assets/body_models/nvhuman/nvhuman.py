"""NVHuman body model layer."""

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from .lbs import lbs


def to_tensor(
    array: Any,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Convert array to torch tensor.

    Args:
        array: Input array to convert.
        dtype: Target dtype for the tensor.

    Returns:
        Torch tensor or None if already a tensor.
    """
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
    return None


def to_np(array: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert array to numpy array.

    Args:
        array: Input array to convert.
        dtype: Target dtype for the array.

    Returns:
        Numpy array.
    """
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


J93_to_12 = [
    12,  # left arm (actually left shoulder)
    44,  # right arm (actually right shoulder)
    13,  # left elbow
    45,  # right elbow
    14,  # left wrist
    46,  # right wrist
    75,  # left leg (actually left hip)
    84,  # right leg (actually right hip)
    76,  # left knee
    85,  # right knee
    77,  # left ankle
    86,  # right ankle
]


class NVHumanLayer(nn.Module):
    """NVHuman body model layer for Linear Blend Skinning."""

    def __init__(
        self,
        model_path: str,
        rest_type: str = "T",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the NVHuman layer.

        Args:
            model_path: Path to the NVHuman model file (.npz).
            rest_type: Rest pose type ("T" or "A").
            dtype: Data type for tensors.
        """
        super().__init__()

        self.data = np.load(model_path)

        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(self.data["faces"], dtype=np.int64), dtype=torch.long),
            False,
        )
        # The vertices of the template model, (18056, 3)
        if "v_template" in self.data:
            self.register_buffer(
                "v_template",
                to_tensor(to_np(self.data["v_template"]), dtype=dtype),
                False,
            )

            # The shape components
            # Shape blend shapes basis, (18056, 3, 10)
            self.register_buffer(
                "shapedirs",
                to_tensor(to_np(self.data["shapedirs"]), dtype=dtype),
                False,
            )
        else:
            raise ValueError("v_template not found in the model")

        self.num_joints = self.data["num_joints"]
        self.register_buffer(
            "parents", to_tensor(to_np(self.data["parents"]), dtype=torch.long), False
        )
        # Vertices to Joints location (23 + 1, 6890)
        self.register_buffer(
            "J_regressor",
            to_tensor(to_np(self.data["J_regressor"]), dtype=dtype),
            False,
        )

        self.register_buffer(
            "lbs_weights",
            to_tensor(to_np(self.data["lbs_weights"]), dtype=dtype),
            False,
        )

        self.rig_joint_names = self.data["rig_joint_names"]

        self.dtype = dtype

        # additional vertex keypoints
        self.vertex_ids = [
            4512,  # nose-tip
            191,  # left_lip_corner
            2395,  # right_lip_corner
            4514,  # lip_top_center,
            4547,  # lip_bottom_center,
            5132,  # left_eye_left_corner
            4682,  # left_eye_right_corner
            9597,  # right_eye_right_corner
            9147,  # right_eye_left_corner
            648,  # left_ear_tragus
            2852,  # right_ear_tragus
            8916,  # left_big_toe
            17966,  # right_big_toe
            1965,  # left_small_toe
            13510,  # right_small_toe
            7797,  # left_heel
            15134,  # right_heel
            1265,  # right_thumb_tip
            85,  # left_thumb_tip
            1825,  # right_index_tip
            341,  # left_index_tip
            1930,  # right_middlefinger_tip
            1688,  # left_middlefinger_tip
            1921,  # right_ringfinger_tip
            1197,  # left_ringfinger_tip
            1909,  # right_pinky_tip
            1185,  # left_pinky_tip
        ]

    def get_skeleton(self, betas: torch.Tensor) -> torch.Tensor:
        """Get the skeleton joint positions from shape parameters.

        Args:
            betas: Shape parameters tensor.

        Returns:
            Joint positions tensor.
        """
        v_shaped = self.v_template + torch.einsum(
            "...l,lmk->...mk", [betas, self.shapedirs]
        )
        return torch.einsum("...ik,ji->...jk", [v_shaped, self.J_regressor])

    def forward(
        self,
        body_pose: torch.Tensor,
        betas: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        pose2rot: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the NVHuman model.

        Args:
            body_pose: Body pose parameters.
            betas: Shape parameters.
            global_orient: Global orientation.
            transl: Translation.
            pose2rot: Whether to convert pose to rotation matrices.

        Returns:
            Dictionary containing vertices, joints, joints_12, and global_rot_mats.
        """
        # concate root orientation with thetas
        if global_orient is not None:
            if global_orient.ndim == body_pose.ndim - 1:
                global_orient = global_orient.unsqueeze(-2)
            if body_pose.shape[-1] == 3:
                full_pose = torch.cat([global_orient, body_pose], dim=-2)
            else:
                full_pose = torch.cat([global_orient, body_pose], dim=-1)
        else:
            full_pose = body_pose

        if full_pose.ndim == 3 and full_pose.shape[-1] == 3:
            B = full_pose.shape[0]
            L = 1
        elif full_pose.ndim == 3:
            # full_pose: (B, L, J * 3)
            B, L = full_pose.shape[:2]
            full_pose = full_pose.reshape(B * L, full_pose.shape[-1])
            if betas is not None:
                betas = betas.reshape(B * L, betas.shape[-1])
            if transl is not None:
                transl = transl.reshape(B * L, 3)
        elif full_pose.ndim == 2:
            B = full_pose.shape[0]
            L = 1

        vertices, joints, global_rot_mats = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
            dtype=self.dtype,
        )

        if transl is not None:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        # # additional vertex keypoints
        # extra_joints = vertices[:, self.vertex_ids, :]
        # joints = torch.cat([joints, extra_joints], dim=1)

        if L > 1:
            vertices = vertices.reshape(B, L, -1, 3)
            joints = joints.reshape(B, L, -1, 3)
            global_rot_mats = global_rot_mats.reshape(B, L, -1, 3, 3)

        # select joints 12 (no face joints)
        joints_12 = joints[..., J93_to_12, :]

        output = {
            "vertices": vertices,
            "joints": joints,
            "joints_12": joints_12,
            "global_rot_mats": global_rot_mats,  # [B, (L,) num_joints, 3, 3]
        }

        return output
