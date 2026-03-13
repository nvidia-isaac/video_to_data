"""Motion dataset utility for loading and managing motion data for MMD reward computation."""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from isaaclab.assets import Articulation


class MotionDataset:
    """
    Utility class for loading and managing motion datasets.

    This class handles loading motion data from various formats (CSV, H5, YAML),
    aligning joint ordering between file format and robot, and sampling states
    for MMD reward computation.
    """

    def __init__(
        self,
        dataset_path: str,
        robot: Articulation,
        file_joint_names: list[str] | None = None,
        joint_order_file: str | None = None,
        include_root: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize the motion dataset.

        Args:
            dataset_path: Path to the motion dataset (directory with CSV files or single H5/YAML file).
            robot: The robot articulation for joint ordering reference.
            file_joint_names: List of joint names in the order they appear in the file.
                If None, will try to load from joint_order_file or assume IsaacLab order.
            joint_order_file: Path to a file containing joint ordering (one joint per line).
            include_root: Whether to include root position/orientation in the state.
            device: Device to store the data on.
        """
        self.dataset_path = Path(dataset_path)
        self.robot = robot
        self.include_root = include_root
        self.device = device

        # Get robot joint names (excluding root)
        self.robot_joint_names = robot.joint_names

        # Load file joint order
        if file_joint_names is not None:
            self.file_joint_names = file_joint_names
        elif joint_order_file is not None:
            self.file_joint_names = self._load_joint_order_file(joint_order_file)
        else:
            # Assume file is in robot order
            self.file_joint_names = self.robot_joint_names

        # Compute reordering indices (file order -> robot order)
        self.reorder_indices = self._compute_reorder_indices()

        # Load all motion data
        self.motion_data = self._load_dataset()

        # Extract joint positions as the feature for MMD
        self.joint_states = self._extract_joint_states()

        print(
            f"[MotionDataset] Loaded {len(self.joint_states)} states from {self.dataset_path}"
        )
        print(f"[MotionDataset] State dimension: {self.joint_states.shape[1]}")

    def _load_joint_order_file(self, filepath: str) -> list[str]:
        """Load joint order from a text file (one joint per line)."""
        joint_names = []
        with open(filepath, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line and not line.startswith("root"):  # Skip root line
                    joint_names.append(line)
        return joint_names

    def _compute_reorder_indices(self) -> torch.Tensor | None:
        """Compute indices to reorder from file joint order to robot joint order.

        Returns:
            Tensor of indices or None if orders match.
        """
        # Find common joints between file and robot
        common_joints = []
        file_indices = []
        robot_indices = []

        for i, joint_name in enumerate(self.robot_joint_names):
            if joint_name in self.file_joint_names:
                file_idx = self.file_joint_names.index(joint_name)
                common_joints.append(joint_name)
                file_indices.append(file_idx)
                robot_indices.append(i)

        if len(common_joints) == 0:
            raise ValueError(
                f"No common joints found between file ({self.file_joint_names[:5]}...) "
                f"and robot ({self.robot_joint_names[:5]}...)"
            )

        print(f"[MotionDataset] Found {len(common_joints)} common joints")

        # Store mapping info
        self.common_joints = common_joints
        self.file_indices = file_indices
        self.robot_indices = robot_indices

        return torch.tensor(file_indices, dtype=torch.long, device=self.device)

    def _load_dataset(self) -> list[np.ndarray]:
        """Load motion data from the dataset path.

        Returns:
            List of motion arrays, each with shape (T, state_dim).
        """
        motion_data = []

        if self.dataset_path.is_dir():
            # Load all CSV files in directory
            csv_files = sorted(self.dataset_path.glob("*.csv"))
            for csv_file in csv_files:
                data = self._load_csv(csv_file)
                if data is not None and len(data) > 0:
                    motion_data.append(data)
        elif self.dataset_path.suffix == ".h5":
            data = self._load_h5(self.dataset_path)
            if data is not None and len(data) > 0:
                motion_data.append(data)
        elif self.dataset_path.suffix in [".yaml", ".yml"]:
            data = self._load_yaml(self.dataset_path)
            if data is not None and len(data) > 0:
                motion_data.append(data)
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_path}")

        if len(motion_data) == 0:
            raise ValueError(f"No motion data loaded from {self.dataset_path}")

        return motion_data

    def _load_csv(self, filepath: Path) -> np.ndarray | None:
        """Load motion data from a CSV file.

        CSV format: root_pos(3), root_quat(4), joints(N) per row.
        """
        try:
            data = []
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:  # Skip empty rows
                        values = [float(v) for v in row]
                        data.append(values)
            if data:
                return np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"[MotionDataset] Warning: Failed to load {filepath}: {e}")
        return None

    def _load_h5(self, filepath: Path) -> np.ndarray | None:
        """Load motion data from an H5 file."""
        try:
            with h5py.File(filepath, "r") as f:
                if "qpos" in f:
                    return np.array(f["qpos"], dtype=np.float32)
        except Exception as e:
            print(f"[MotionDataset] Warning: Failed to load {filepath}: {e}")
        return None

    def _load_yaml(self, filepath: Path) -> np.ndarray | None:
        """Load motion data from a YAML file."""
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            if "qpos" in data:
                return np.array(data["qpos"], dtype=np.float32)
        except Exception as e:
            print(f"[MotionDataset] Warning: Failed to load {filepath}: {e}")
        return None

    def _extract_joint_states(self) -> torch.Tensor:
        """Extract joint position states from all motion data.

        Returns:
            Tensor of shape (N, joint_dim) containing all joint states.
        """
        all_states = []

        for motion in self.motion_data:
            # Motion format: [root_pos(3), root_quat(4), joints(...)]
            # Extract joint positions (skip root: 7 values)
            if motion.shape[1] > 7:
                joint_pos = motion[:, 7:]  # (T, num_file_joints)

                # Reorder to robot joint order using common joints
                if self.reorder_indices is not None:
                    # Extract only the joints that exist in file
                    joint_pos_reordered = joint_pos[:, self.file_indices]
                    all_states.append(joint_pos_reordered)
                else:
                    all_states.append(joint_pos)

        if len(all_states) == 0:
            raise ValueError("No joint states extracted from motion data")

        # Concatenate all states
        all_states = np.concatenate(all_states, axis=0)
        return torch.tensor(all_states, dtype=torch.float32, device=self.device)

    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample a subset of motion states.

        Args:
            num_samples: Number of samples to return.

        Returns:
            Tensor of shape (num_samples, joint_dim).
        """
        total_samples = len(self.joint_states)
        if num_samples >= total_samples:
            return self.joint_states

        indices = torch.randperm(total_samples, device=self.device)[:num_samples]
        return self.joint_states[indices]

    def get_all_states(self) -> torch.Tensor:
        """Get all motion states.

        Returns:
            Tensor of shape (N, joint_dim).
        """
        return self.joint_states

    def get_joint_indices_for_robot(self) -> list[int]:
        """Get the robot joint indices that are available in the dataset.

        Returns:
            List of robot joint indices that have corresponding data in the dataset.
        """
        return self.robot_indices

    @property
    def num_states(self) -> int:
        """Return the total number of states in the dataset."""
        return len(self.joint_states)

    @property
    def state_dim(self) -> int:
        """Return the dimension of each state."""
        return self.joint_states.shape[1]
