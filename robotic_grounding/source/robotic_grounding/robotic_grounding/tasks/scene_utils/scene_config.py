from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from robotic_grounding.assets.object_registry import get_object_spec


@dataclass
class ObjectConfig:
    """Configuration for a scene object (target or fixed)."""

    name: str
    usd_path: str
    position_key: str
    quaternion_key: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pos_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    init_pos: list[float] | None = None
    init_rot: list[float] | None = None


def _resolve_object(
    obj_name: str, obj_cfg: dict | None = None
) -> tuple[str | None, tuple[float, float, float] | None]:
    """Resolve USD path and scale from config or registry.

    Args:
        obj_name: Name of the object.
        obj_cfg: Optional config dict with usd_path and scale overrides.

    Returns:
        Tuple of (usd_path, scale).
    """
    obj_cfg = obj_cfg or {}
    if "usd_path" in obj_cfg:
        usd_path = obj_cfg["usd_path"]
        scale = tuple(obj_cfg.get("scale", [1.0, 1.0, 1.0]))
    else:
        spec = get_object_spec(obj_name)
        if spec is None:
            print(f"Object '{obj_name}' not in registry and no usd_path provided")
            return None, None
        usd_path = spec.usd_path
        scale = obj_cfg.get("scale", spec.scale)
    return usd_path, scale


def _parse_object_config(obj_cfg: dict) -> ObjectConfig | None:
    """Parse an object configuration dict into an ObjectConfig.

    Args:
        obj_cfg: Config dict with name, optional usd_path, scale, position_key, quaternion_key, pos_offset.
                 Can also include init_pos and init_rot for fixed objects not from motion file.

    Returns:
        Parsed ObjectConfig.
    """
    obj_name = obj_cfg["name"]
    position_key = obj_cfg.get("position_key", f"{obj_name}_position")
    quaternion_key = obj_cfg.get("quaternion_key", f"{obj_name}_wxyz")
    usd_path, scale = _resolve_object(obj_name, obj_cfg)
    if usd_path is None or scale is None:
        return None

    return ObjectConfig(
        name=obj_name,
        usd_path=usd_path,
        scale=scale,
        position_key=position_key,
        quaternion_key=quaternion_key,
        pos_offset=obj_cfg.get("pos_offset", [0.0, 0.0, 0.0]),
        init_pos=obj_cfg.get("init_pos"),
        init_rot=obj_cfg.get("init_rot"),
    )


def _discover_objects_from_file(motion_file: str) -> list[str]:
    """Discover object names from file by finding *_position keys.

    Args:
        motion_file: Path to file.

    Returns:
        List of object names found in the file.
    """
    objects = []
    if motion_file.endswith(".h5"):
        with h5py.File(motion_file, "r") as f:
            for key in f.keys():
                if key.endswith("_position"):
                    obj_name = key[:-9]
                    objects.append(obj_name)
    elif motion_file.endswith(".yaml"):
        with open(motion_file, "r") as f:
            data = yaml.safe_load(f)
            print(data.keys())
            for obj_name in data.keys():
                if obj_name.endswith("_position"):
                    base_name = obj_name[:-9]
                    objects.append(base_name)
    elif motion_file.endswith(".parquet"):
        table = pq.read_table(motion_file)
        if "object_name" in table.column_names:
            obj_name = table.column("object_name")[0].as_py()
            if obj_name:
                objects.append(obj_name)
    else:
        raise ValueError(f"Unsupported file type: {motion_file}")
    return objects


@dataclass
class SceneConfig:
    """Scene configuration loaded from YAML, HDF5, or Parquet.

    Defines the target object, fixed objects, and robot configuration for a scene.
    Objects can be specified explicitly in YAML or auto-discovered from data files.

    For end-effector motion files, the motion data contains EE poses (6dof) + hand joints (7dof)
    rather than full body joint positions. The robot_init_qpos will be set to defaults
    and EE-specific fields will be populated instead.
    """

    motion_file: str
    robot_qpos_key: str
    robot_anchor_offset: list[float]
    target_object: ObjectConfig
    fixed_objects: list[ObjectConfig]
    robot_type: str = "g1"
    file_joint_order: str | list[str] | None = None
    ee_links: list[str] | None = None

    # Full robot joint positions (for h5/yaml files)
    robot_init_qpos: torch.Tensor | None = None

    # EE-based
    left_hand_init_qpos: list[float] | None = None
    right_hand_init_qpos: list[float] | None = None
    head_init_translation: list[float] | None = None
    head_init_wxyz: list[float] | None = None
    root_init_translation: list[float] | None = None
    root_init_wxyz: list[float] | None = None

    # Flag indicating if motion data is EE-based (parquet) vs full joints
    is_ee_motion: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> SceneConfig:
        """Load scene configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Populated SceneConfig with initial poses loaded from HDF5.
        """
        path = Path(yaml_path)

        if not path.exists():
            raise FileNotFoundError(f"Scene config file not found: {path.absolute()}")

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        motion_file: str = cfg["motion_file"]
        if not Path(motion_file).is_absolute():
            motion_file = str(Path.cwd() / motion_file)

        robot_cfg = cfg.get("robot", {})
        target_object = _parse_object_config(cfg["target_object"])
        if target_object is None:
            raise ValueError("Could not resolve target_object from config")

        fixed_objects: list[ObjectConfig] = []
        if "fixed_objects" in cfg:
            for obj_cfg in cfg["fixed_objects"]:
                parsed = _parse_object_config(obj_cfg)
                if parsed is not None:
                    fixed_objects.append(parsed)
        else:
            discovered = _discover_objects_from_file(motion_file)
            for obj_name in discovered:
                if obj_name == target_object.name or obj_name in robot_cfg.get(
                    "ee_links", []
                ):
                    continue
                spec = get_object_spec(obj_name)
                if spec is None:
                    continue
                fixed_objects.append(
                    ObjectConfig(
                        name=obj_name,
                        usd_path=spec.usd_path,
                        scale=spec.scale,
                        position_key=f"{obj_name}_position",
                        quaternion_key=f"{obj_name}_wxyz",
                    )
                )

        ee_links = robot_cfg.get("ee_links")

        config = cls(
            motion_file=motion_file,
            robot_qpos_key=robot_cfg.get("qpos_key", "qpos"),
            robot_anchor_offset=robot_cfg.get("anchor_offset", [0.0, 0.0, 0.0]),
            target_object=target_object,
            fixed_objects=fixed_objects,
            robot_type=robot_cfg.get("type", "g1"),
            file_joint_order=robot_cfg.get("file_joint_order"),
            ee_links=ee_links,
            is_ee_motion=ee_links is not None and len(ee_links) > 0,
        )

        config._load_initial_poses()
        return config

    def _load_initial_poses(self) -> None:
        """Load initial poses from file frame 0 for all objects."""
        if self.motion_file.endswith(".h5"):
            with h5py.File(self.motion_file, "r") as f:
                self.robot_init_qpos = f[self.robot_qpos_key][0]

                pos = f[self.target_object.position_key][0].tolist()
                rot = f[self.target_object.quaternion_key][0].tolist()
                self.target_object.init_pos = [
                    p + o
                    for p, o in zip(pos, self.target_object.pos_offset, strict=True)
                ]
                self.target_object.init_rot = rot

                for fixed_obj in self.fixed_objects:
                    # Skip if init_pos/init_rot already set from config
                    if (
                        fixed_obj.init_pos is not None
                        and fixed_obj.init_rot is not None
                    ):
                        continue
                    pos = f[fixed_obj.position_key][0].tolist()
                    rot = f[fixed_obj.quaternion_key][0].tolist()
                    fixed_obj.init_pos = [
                        p + o for p, o in zip(pos, fixed_obj.pos_offset, strict=True)
                    ]
                    fixed_obj.init_rot = rot

        elif self.motion_file.endswith(".yaml"):
            with open(self.motion_file, "r") as f:
                data = yaml.safe_load(f)
                self.robot_init_qpos = [float(v) for v in data[self.robot_qpos_key][0]]

                pos = data[self.target_object.position_key][0]
                rot = data[self.target_object.quaternion_key][0]
                self.target_object.init_pos = [
                    float(p) + float(o)
                    for p, o in zip(pos, self.target_object.pos_offset, strict=True)
                ]
                self.target_object.init_rot = [float(r) for r in rot]

                for fixed_obj in self.fixed_objects:
                    # Skip if init_pos/init_rot already set from config
                    if (
                        fixed_obj.init_pos is not None
                        and fixed_obj.init_rot is not None
                    ):
                        continue
                    pos = data[fixed_obj.position_key][0]
                    rot = data[fixed_obj.quaternion_key][0]
                    fixed_obj.init_pos = [
                        float(p) + float(o)
                        for p, o in zip(pos, fixed_obj.pos_offset, strict=True)
                    ]
                    fixed_obj.init_rot = [float(r) for r in rot]

        elif self.motion_file.endswith(".parquet"):
            if self.is_ee_motion:
                self._load_ee_parquet_poses()
            else:
                raise ValueError(
                    "Parquet files are only supported for end-effector motion right now."
                )

    def _load_ee_parquet_poses(self) -> None:
        """Load initial poses from end-effector motion file frame 0."""
        table = pq.read_table(self.motion_file)
        data = table.to_pydict()

        # Load EE-based hand data (first timestep of first trajectory)
        if "robot_left_qpos" in data and data["robot_left_qpos"]:
            left_qpos_trajectory = data["robot_left_qpos"][0]  # First trajectory
            if left_qpos_trajectory:
                self.left_hand_init_qpos = list(
                    left_qpos_trajectory[0]
                )  # First timestep

        if "robot_right_qpos" in data and data["robot_right_qpos"]:
            right_qpos_trajectory = data["robot_right_qpos"][0]
            if right_qpos_trajectory:
                self.right_hand_init_qpos = list(right_qpos_trajectory[0])

        # Load head pose
        if "nvhuman_head_translation" in data and data["nvhuman_head_translation"]:
            head_trans_trajectory = data["nvhuman_head_translation"][0]
            if head_trans_trajectory:
                self.head_init_translation = list(head_trans_trajectory[0])

        if "nvhuman_head_wxyz" in data and data["nvhuman_head_wxyz"]:
            head_wxyz_trajectory = data["nvhuman_head_wxyz"][0]
            if head_wxyz_trajectory:
                self.head_init_wxyz = list(head_wxyz_trajectory[0])

        # Load root pose
        if "nvhuman_root_translation" in data and data["nvhuman_root_translation"]:
            root_trans_trajectory = data["nvhuman_root_translation"][0]
            if root_trans_trajectory:
                self.root_init_translation = list(root_trans_trajectory[0])

        if "nvhuman_root_wxyz" in data and data["nvhuman_root_wxyz"]:
            root_wxyz_trajectory = data["nvhuman_root_wxyz"][0]
            if root_wxyz_trajectory:
                self.root_init_wxyz = list(root_wxyz_trajectory[0])

        # Load object pose - convert axis-angle to quaternion wxyz
        if "object_translation" in data and data["object_translation"]:
            obj_trans_trajectory = data["object_translation"][0]
            if obj_trans_trajectory:
                pos = list(obj_trans_trajectory[0])
                self.target_object.init_pos = [
                    p + o
                    for p, o in zip(pos, self.target_object.pos_offset, strict=True)
                ]

        if "object_axis_angle" in data and data["object_axis_angle"]:
            obj_aa_trajectory = data["object_axis_angle"][0]
            if obj_aa_trajectory:
                axis_angle = np.array(obj_aa_trajectory[0])
                # Convert axis-angle to quaternion (wxyz format)
                rot = R.from_rotvec(axis_angle)
                wxyz = rot.as_quat(scalar_first=True)
                self.target_object.init_rot = wxyz.tolist()

        self.robot_init_qpos = None
