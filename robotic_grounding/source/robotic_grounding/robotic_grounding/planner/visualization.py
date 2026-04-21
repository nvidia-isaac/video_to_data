"""MuJoCo viewer for planner output visualization.

Shows planned body motion with target/achieved EE axes, object trajectory,
and optional finger animation.
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation


def add_axes(
    scn,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    length: float = 0.08,
    radius: float = 0.004,
    alpha: float = 1.0,
) -> None:
    """Draw an RGB axis triad in the MuJoCo viewer scene.

    Args:
        scn: MuJoCo viewer scene (mujoco.MjvScene).
        pos: (3,) position.
        quat_wxyz: (4,) orientation in wxyz format.
        length: Axis length in meters.
        radius: Axis cylinder radius.
        alpha: Opacity.
    """
    q_xyzw = quat_wxyz[[1, 2, 3, 0]]
    rot = Rotation.from_quat(q_xyzw).as_matrix()

    colors = [
        [1, 0, 0, alpha],  # X = red
        [0, 1, 0, alpha],  # Y = green
        [0, 0, 1, alpha],  # Z = blue
    ]

    for axis_idx in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        tip = pos + rot[:, axis_idx] * length
        rgba = np.array(colors[axis_idx], dtype=np.float32)
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            np.asarray(pos, dtype=np.float64),
            np.asarray(tip, dtype=np.float64),
        )
        scn.geoms[scn.ngeom].rgba = rgba
        scn.ngeom += 1


def _inject_scene_objects(
    xml_path, object_mesh_paths=None, support_usda_path=None, support_transform=None
):
    """Inject object meshes and support surface into MuJoCo XML.

    Args:
        xml_path: Base robot XML path.
        object_mesh_paths: List of .obj mesh paths (one per object body), or None.
        support_usda_path: Path to support surface .usda file, or None.
        support_transform: Dict for transforming support positions.

    Returns:
        Modified XML string with injected bodies.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_dir = os.path.dirname(os.path.abspath(xml_path))

    # Resolve any include files by inlining them
    # (from_xml_string can't resolve relative includes)
    for inc in root.findall(".//include"):
        inc_file = inc.get("file")
        if inc_file and not os.path.isabs(inc_file):
            inc_path = os.path.join(xml_dir, inc_file)
            if os.path.exists(inc_path):
                inc_tree = ET.parse(inc_path)
                inc_root = inc_tree.getroot()
                parent = root
                for child in list(inc_root):
                    parent.append(child)
                parent.remove(inc)

    # Absolutize meshdir so from_xml_string can find meshes
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
        if meshdir and not os.path.isabs(meshdir):
            compiler.set("meshdir", os.path.abspath(os.path.join(xml_dir, meshdir)))

    worldbody = root.find("worldbody")

    # Inject one visual-only free body per object mesh
    if object_mesh_paths:
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        for i, mesh_path in enumerate(object_mesh_paths):
            if os.path.exists(mesh_path):
                mesh_name = f"object_mesh_{i}"
                ET.SubElement(
                    asset,
                    "mesh",
                    name=mesh_name,
                    file=mesh_path,
                    scale="0.001 0.001 0.001",
                )
                body_name = f"object_vis_{i}"
                obj_body = ET.SubElement(worldbody, "body", name=body_name, pos="0 0 0")
                ET.SubElement(obj_body, "freejoint", name=f"{body_name}_joint")
                ET.SubElement(
                    obj_body,
                    "geom",
                    type="mesh",
                    mesh=mesh_name,
                    contype="0",
                    conaffinity="0",
                    rgba="0.8 0.6 0.4 0.8",
                )

    # Inject support surface from USDA (parse cylinder primitives)
    if support_usda_path is not None and os.path.exists(support_usda_path):
        with open(support_usda_path) as f:
            usda_content = f.read()

        # Parse cylinder definitions from USDA
        cylinders = re.findall(
            r'def Cylinder "(\w+)".*?height = ([\d.]+).*?radius = ([\d.]+).*?'
            r"xformOp:translate = \(([-\d., ]+)\)",
            usda_content,
            re.DOTALL,
        )

        for name, height, radius, translate in cylinders:
            pos = np.array([float(x.strip()) for x in translate.split(",")])

            # Transform from V2P frame to G1 viewer frame
            if support_transform is not None:
                src_mid = support_transform["src_midpoint"]
                r_yaw = Rotation.from_euler("z", support_transform["delta_yaw"])
                pos = r_yaw.apply(pos - src_mid) + src_mid
                pos += support_transform["offset"]

            pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
            support_body = ET.SubElement(
                worldbody, "body", name=f"support_{name}", pos=pos_str
            )
            ET.SubElement(
                support_body,
                "geom",
                type="cylinder",
                size=f"{radius} {float(height)/2}",
                contype="0",
                conaffinity="0",
                rgba="0.7 0.6 0.5 0.6",
            )

    return ET.tostring(root, encoding="unicode")


def visualize(
    xml_path: str,
    qpos_traj: np.ndarray,
    traj_lp: np.ndarray,
    traj_lq: np.ndarray,
    traj_rp: np.ndarray,
    traj_rq: np.ndarray,
    left_finger_joints: np.ndarray | None = None,
    right_finger_joints: np.ndarray | None = None,
    recorded_left_names: list[str] | None = None,
    recorded_right_names: list[str] | None = None,
    object_pos: np.ndarray | None = None,
    object_quat: np.ndarray | None = None,
    fps: float = 30.0,
    ref_start: int = 0,
    object_mesh_paths: list[str] | None = None,
    support_usda_path: str | None = None,
    support_transform: dict | None = None,
) -> None:
    """Interactive playback of planned motion with target/achieved EE axes.

    Args:
        xml_path: MuJoCo XML path.
        qpos_traj: (T, nq) full qpos trajectory.
        traj_lp: (T, 3) target left wrist positions.
        traj_lq: (T, 4) target left wrist quaternions (wxyz).
        traj_rp: (T, 3) target right wrist positions.
        traj_rq: (T, 4) target right wrist quaternions (wxyz).
        left_finger_joints: Optional (T_ref, J_left) finger joint angles.
        right_finger_joints: Optional (T_ref, J_right) finger joint angles.
        recorded_left_names: Finger joint names for left hand.
        recorded_right_names: Finger joint names for right hand.
        object_pos: Optional (T_ref, 3) object positions.
        object_quat: Optional (T_ref, 4) object quaternions (wxyz).
        fps: Playback frame rate.
        ref_start: Frame index where reference data begins.
        object_mesh_path: Path to object .obj mesh for visual overlay.
        support_usda_path: Path to support surface USDA file.
    """
    # Inject scene objects into XML if provided
    if object_mesh_paths or support_usda_path:
        xml_str = _inject_scene_objects(
            xml_path,
            object_mesh_paths,
            support_usda_path,
            support_transform=support_transform,
        )
        model = mujoco.MjModel.from_xml_string(xml_str)
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    left_id = model.body("left_wrist_yaw_link").id
    right_id = model.body("right_wrist_yaw_link").id

    # Find injected object bodies (one per mesh)
    obj_vis_jnt_adrs = []
    for i in range(len(object_mesh_paths or [])):
        try:
            jnt_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"object_vis_{i}_joint"
            )
            if jnt_id >= 0:
                obj_vis_jnt_adrs.append(model.jnt_qposadr[jnt_id])
        except Exception:
            pass

    # Build finger joint mappings
    l_finger_map = _build_finger_map(model, recorded_left_names)
    r_finger_map = _build_finger_map(model, recorded_right_names)

    T_play = min(qpos_traj.shape[0], len(traj_lp))
    paused = False

    def key_callback(key):
        nonlocal paused
        if key == 32:  # spacebar
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 180

        frame = 0
        while viewer.is_running() and frame < T_play:
            if paused:
                time.sleep(0.01)
                viewer.sync()
                continue

            # Set body qpos
            data.qpos[: qpos_traj.shape[1]] = qpos_traj[frame]

            # Update object body positions
            if obj_vis_jnt_adrs and object_pos is not None and frame >= ref_start:
                obj_idx = min(frame - ref_start, object_pos.shape[0] - 1)
                for i, jnt_adr in enumerate(obj_vis_jnt_adrs):
                    if object_pos.ndim == 3 and i < object_pos.shape[1]:
                        data.qpos[jnt_adr : jnt_adr + 3] = object_pos[obj_idx, i]
                        if object_quat is not None:
                            data.qpos[jnt_adr + 3 : jnt_adr + 7] = object_quat[
                                obj_idx, i
                            ]
                    elif object_pos.ndim == 2 and i == 0:
                        data.qpos[jnt_adr : jnt_adr + 3] = object_pos[obj_idx]
                        if object_quat is not None:
                            data.qpos[jnt_adr + 3 : jnt_adr + 7] = object_quat[obj_idx]

            # Set finger joints from reference
            if frame >= ref_start:
                f_idx = min(
                    frame - ref_start,
                    (
                        (left_finger_joints.shape[0] - 1)
                        if left_finger_joints is not None
                        else 0
                    ),
                )
                _set_fingers(data, model, l_finger_map, left_finger_joints, f_idx)
                _set_fingers(data, model, r_finger_map, right_finger_joints, f_idx)

            mujoco.mj_forward(model, data)

            # Draw axes
            viewer.user_scn.ngeom = 0

            # Target EE (thick)
            add_axes(
                viewer.user_scn,
                traj_lp[frame],
                traj_lq[frame],
                length=0.12,
                radius=0.006,
            )
            add_axes(
                viewer.user_scn,
                traj_rp[frame],
                traj_rq[frame],
                length=0.12,
                radius=0.006,
            )

            # Achieved EE (thin)
            add_axes(
                viewer.user_scn,
                data.xpos[left_id],
                data.xquat[left_id],
                length=0.08,
                radius=0.003,
                alpha=0.6,
            )
            add_axes(
                viewer.user_scn,
                data.xpos[right_id],
                data.xquat[right_id],
                length=0.08,
                radius=0.003,
                alpha=0.6,
            )

            viewer.sync()
            time.sleep(1.0 / fps)
            frame += 1


def _build_finger_map(model, joint_names: list[str] | None) -> list[int]:
    """Map finger joint names to model qpos indices."""
    if not joint_names:
        return []
    result = []
    for name in joint_names:
        try:
            jid = model.joint(name).id
            result.append(int(model.jnt_qposadr[jid]))
        except Exception:
            result.append(-1)
    return result


def _set_fingers(data, model, finger_map, finger_joints, f_idx):
    """Set finger joint positions from reference data."""
    if finger_joints is None or not finger_map:
        return
    for j, qi in enumerate(finger_map):
        if qi >= 0 and j < finger_joints.shape[1]:
            data.qpos[qi] = finger_joints[f_idx, j]
