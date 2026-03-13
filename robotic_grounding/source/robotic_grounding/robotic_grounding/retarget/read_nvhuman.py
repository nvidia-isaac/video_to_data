"""NVHuman model wrapper for motion processing and visualization."""

from typing import Any, Dict, Optional

import numpy as np
import torch
from judo.visualizers.model import add_mesh
from scipy.spatial.transform import Rotation as R

from robotic_grounding.assets.body_models.nvhuman.nvhuman import NVHumanLayer
from robotic_grounding.retarget import BODY_MODELS_DIR


class NVHuman:
    """NVHuman model class to process NVHuman motion data and visualize."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the NVHuman model.

        Args:
            model_path: Path to the NVHuman model file. If None, uses default path.
            device: Torch device to use. Defaults to CUDA if available.
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if model_path is None:
            model_path = str(
                BODY_MODELS_DIR / "nvhuman" / "models" / "nvHuman_shape_TPose.npz"
            )

        self.model = NVHumanLayer(model_path=model_path).to(self.device)
        self.num_joints = self.model.num_joints
        self.faces = self.model.faces_tensor.cpu().numpy()

    def forward(
        self,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the NVHuman model.

        Args:
            body_pose: Body pose parameters, shape (B, num_pose_params).
            betas: Shape parameters, shape (B, num_betas) or (num_betas,).
            global_orient: Global orientation, shape (B, 3).
            transl: Translation, shape (B, 3).

        Returns:
            Dictionary containing:
                - joints: Joint positions, shape (B, num_joints, 3).
                - joints_wxyz: Joint orientations as wxyz quaternions, shape (B, num_joints, 4).
                - global_rot_mats: Global rotation matrices, shape (B, num_joints, 3, 3).
        """
        body_pose = body_pose.to(self.device)
        betas = betas.to(self.device)
        global_orient = global_orient.to(self.device)
        transl = transl.to(self.device)

        # Forward pass
        output = self.model.forward(
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
        )

        result = {
            "joints": output["joints"],  # (B, num_joints, 3)
            "vertices": output["vertices"],  # (B, num_vertices, 3)
        }

        # Convert rotation matrices to quaternions if available
        if "global_rot_mats" in output:
            rot_mats = output["global_rot_mats"]  # (B, num_joints, 3, 3)
            result["global_rot_mats"] = rot_mats
            result["joints_wxyz"] = self._rotation_matrices_to_wxyz(rot_mats)
        else:
            raise ValueError("Global rotation matrices not available")

        return result

    def _rotation_matrices_to_wxyz(self, rot_mats: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to wxyz quaternions.

        Args:
            rot_mats: Rotation matrices, shape (B, num_joints, 3, 3).

        Returns:
            Quaternions in wxyz format, shape (B, num_joints, 4).
        """
        B, num_joints = rot_mats.shape[:2]
        rot_mats_np = rot_mats.detach().cpu().numpy()

        joints_wxyz = np.zeros((B, num_joints, 4), dtype=np.float64)
        for b in range(B):
            for j in range(num_joints):
                quat = R.from_matrix(rot_mats_np[b, j]).as_quat(scalar_first=True)
                joints_wxyz[b, j] = quat

        return torch.from_numpy(joints_wxyz).to(self.device)

    def load_motion(
        self,
        params_path: str,
    ) -> Dict[str, np.ndarray]:
        """Load motion parameters and compute joint positions/orientations.

        Args:
            params_path: Path to the motion parameters file (.pt).
            normalize_to_origin: If True, normalize the motion so the first frame
                starts at the origin with canonical orientation (identity global_orient).
                This makes the coordinate transforms work correctly.

        Returns:
            Dictionary containing:
                - joints: Joint positions, shape (num_frames, num_joints, 3).
                - joints_wxyz: Joint orientations as wxyz quaternions.
                - vertices: Mesh vertices, shape (num_frames, num_vertices, 3).
                - num_frames: Number of frames in the motion.
        """
        params = torch.load(params_path)

        output = self.forward(
            body_pose=params["body_pose"].to(self.device),
            betas=params["betas"].to(self.device),
            global_orient=params["global_orient"].to(self.device),
            transl=params["transl"].to(self.device),
        )

        joints = output["joints"].cpu().numpy()
        joints_wxyz = output["joints_wxyz"].cpu().numpy()
        vertices = output["vertices"].cpu().numpy()
        num_frames = joints.shape[0]

        # Get first frame's global orientation and translation
        global_orient_first = params["global_orient"][0].cpu().numpy()
        transl_first = params["transl"][0].cpu().numpy()

        # Compute inverse rotation
        R_first = R.from_rotvec(global_orient_first).as_matrix()
        R_first_inv = R_first.T

        # Apply inverse rotation and translation to all frames
        for i in range(num_frames):
            # Remove first frame translation, then undo rotation
            joints[i] = (joints[i] - transl_first) @ R_first_inv.T
            vertices[i] = (vertices[i] - transl_first) @ R_first_inv.T

            # Correct the joint rotations
            for j in range(joints_wxyz.shape[1]):
                rot_mat = R.from_quat(joints_wxyz[i, j], scalar_first=True).as_matrix()
                rot_mat_corrected = R_first_inv @ rot_mat
                joints_wxyz[i, j] = R.from_matrix(rot_mat_corrected).as_quat(
                    scalar_first=True
                )

        return {
            "joints": joints,
            "joints_wxyz": joints_wxyz,
            "vertices": vertices,
            "num_frames": num_frames,
        }

    def visualize(
        self,
        viser_server: Any,
        vertices: torch.Tensor | np.ndarray,
        root_path: str = "/nvhuman",
        rgba: np.ndarray | None = None,
    ) -> None:
        """Visualize NVHuman mesh in viser.

        Args:
            viser_server: Viser server instance.
            vertices: Mesh vertices, shape (num_vertices, 3).
            root_path: Root path in viser scene tree.
            rgba: RGBA color array, shape (4,). Defaults to skin tone.
        """
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if rgba is None:
            rgba = np.array([255, 219, 172, 180])

        add_mesh(
            viser_server,
            f"{root_path}/mesh",
            vertices=vertices,
            faces=self.faces,
            pos=np.array([0, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            rgba=rgba,
        )
