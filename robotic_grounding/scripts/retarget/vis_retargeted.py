# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Visualize retargeted hand-object data from any dataset (ARCTIC, TACO, etc.).

Loads ManoSharpaData Parquet from a given directory; object meshes are loaded from
the object_mesh_paths field (one path per object body). Plays back robot hands + object poses in viser.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation

try:
    from pxr import Usd, UsdGeom

    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False
from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import (
    ManoSharpaData,
    add_sequence_filter_args,
    filter_sequence_ids,
    list_sequence_ids,
)
from robotic_grounding.retarget.dataset_registry import (
    get_all_dataset_names,
    get_dataset_config,
)
from robotic_grounding.retarget.hand_kinematics import HandKinematics
from robotic_grounding.retarget.params import MANO_FINGERTIP_INDICES, MANO_HAND_LINKS
from robotic_grounding.retarget.read_mano import MANO
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics

DEFAULT_HTML_DIR = HUMAN_MOTION_DATA_DIR / "html"

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def distance_to_color(d: float) -> tuple[int, int, int]:
    """Map distance to a green-to-red color gradient.

    Green (0, 255, 0) at d <= 0.01m (contact), red (255, 0, 0) at d >= 0.05m (far).
    """
    t = np.clip((d - 0.01) / (0.05 - 0.01), 0.0, 1.0)
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    return (r, g, 0)


def load_object_meshes_from_paths(
    viser_server: viser.ViserServer,
    object_mesh_paths: list[str],
    object_body_names: list[str],
) -> dict[str, Any]:
    """Load object meshes from schema paths (one per body) and add them to the viser scene.

    Paths ending with _cm.obj are scaled by 0.01 (cm -> m). Returns dict mapping body name to handle.
    """
    handles: dict[str, Any] = {}
    for part, path in zip(object_body_names, object_mesh_paths, strict=True):
        if not path or not Path(path).exists():
            continue
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        if path.endswith("_cm.obj"):
            mesh.vertices *= 0.01
        # Use a FrameHandle parent so per-frame position updates are sent as
        # full add_frame messages — property-setter updates on mesh handles are
        # recorded as delta messages that the viser player doesn't replay on
        # subsequent loop iterations, causing the object to freeze after loop 1.
        frame_handle = viser_server.scene.add_frame(
            name=f"/object/{part}",
            position=np.array([0.0, 0.0, 0.0]),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            show_axes=False,
        )
        viser_server.scene.add_mesh_trimesh(
            name=f"/object/{part}/mesh",
            mesh=mesh,
        )
        handles[part] = frame_handle
    return handles


def load_support_surfaces_from_usd(
    viser_server: viser.ViserServer,
    usd_path: Path,
) -> dict[str, Any]:
    """Load support-surface disks from a .usda file and add them to the viser scene.

    Each ``UsdGeom.Cylinder`` prim is converted to a trimesh cylinder and rendered
    with the ``displayColor`` stored in the USD (falls back to light-blue if absent).

    Args:
        viser_server: The running viser server.
        usd_path: Path to the ``.usda`` file produced by ``reconstruct_support_surfaces.py``.

    Returns:
        Dict mapping prim path to the viser mesh handle.
    """
    stage = Usd.Stage.Open(str(usd_path))
    handles: dict[str, Any] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Cylinder):
            continue
        cyl = UsdGeom.Cylinder(prim)
        radius = cyl.GetRadiusAttr().Get()
        height = cyl.GetHeightAttr().Get()
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        translate = ops[0].Get() if ops else (0.0, 0.0, 0.0)

        display_color = cyl.GetDisplayColorAttr().Get()
        if display_color and len(display_color) > 0:
            c = display_color[0]
            rgba = [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 180]
        else:
            rgba = [150, 200, 255, 160]

        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
        mesh.apply_translation(translate)
        mesh.visual.face_colors = rgba

        prim_path = str(prim.GetPath())
        handles[prim_path] = viser_server.scene.add_mesh_trimesh(
            name=prim_path,
            mesh=mesh,
        )
    return handles


def _dataset_processed_dir(name: str) -> str:
    """Resolve ``{name}/{name}_processed`` using the dataset registry."""
    cfg = get_dataset_config(name)
    return f"{cfg.name}/{cfg.name}{cfg.processed_suffix}"


DATASET_DIRS: dict[str, str] = {
    name: _dataset_processed_dir(name) for name in get_all_dataset_names()
}


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize retargeted Parquet data (hands + optional object meshes)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_DIRS),
        default=None,
        help=(
            "Dataset shorthand; sets --input_dir to the corresponding processed directory "
            f"under HUMAN_MOTION_DATA_DIR. Choices: {list(DATASET_DIRS)}. "
            "Ignored when --input_dir is set explicitly."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help=(
            "Root directory of retargeted Parquet (e.g. .../arctic_processed). "
            "Defaults to arctic_processed when neither --input_dir nor --dataset is given."
        ),
    )
    parser.add_argument(
        "-tid",
        "--trajectory_id",
        type=int,
        default=0,
        help="Row index when multiple rows match filters (default 0).",
    )
    # Adds --sequence_id, --sequence_pattern, --max_sequences (shared with
    # the loader + retarget CLIs so workflow/retarget.yaml can pass the same
    # FILTER_ARGS through every stage).
    add_sequence_filter_args(parser)
    parser.add_argument(
        "--show_mano",
        action="store_true",
        default=False,
        help="Show MANO hand meshes and joint frames alongside robot hands.",
    )
    parser.add_argument(
        "--visualize_contacts",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--visualize_fingertip_distances",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--support_usd",
        type=Path,
        default=None,
        help="Explicit path to a support-surface .usda file (overrides auto-discovery).",
    )
    parser.add_argument(
        "--save_html",
        action="store_true",
        default=False,
        help=(
            "Record the animation to a .viser file and build a local viser client for offline "
            "playback. Outputs to --html_dir. After running, scp that directory to your local "
            "machine, then: cd <html_dir> && python -m http.server 8000"
        ),
    )
    parser.add_argument(
        "--html_dir",
        type=Path,
        default=None,
        help=f"Output directory for --save_html (default: {DEFAULT_HTML_DIR}).",
    )
    parser.add_argument(
        "--save_mp4",
        action="store_true",
        default=False,
        help=(
            "Also render an offline MP4 via pyrender next to the .viser file. "
            "Requires --save_html so the output directory exists. No browser or "
            "Isaac Sim needed — uses the pinocchio/visual meshes plus MANO verts."
        ),
    )
    return parser.parse_args()


def mano_kwargs_from_data(logger_data: Any) -> dict[str, Any]:
    """Read MANO model kwargs stored in the Parquet data."""
    kwargs: dict[str, Any] = {}
    if hasattr(logger_data, "mano_flat_hand_mean"):
        kwargs["flat_hand_mean"] = logger_data.mano_flat_hand_mean
    if (
        hasattr(logger_data, "mano_center_idx")
        and logger_data.mano_center_idx is not None
    ):
        kwargs["center_idx"] = logger_data.mano_center_idx
    return kwargs


def run_mano_forward_from_data(
    mano: MANO,
    logger_data: Any,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run MANO forward pass for both hands using stored Parquet parameters.

    Returns dict keyed by "right"/"left", each containing MANO forward outputs.
    """
    results: dict[str, dict[str, torch.Tensor]] = {}
    for side in ("right", "left"):
        trans = torch.tensor(
            getattr(logger_data, f"mano_{side}_trans"),
            dtype=torch.float32,
            device=device,
        )
        global_orient = torch.tensor(
            getattr(logger_data, f"mano_{side}_global_orient"),
            dtype=torch.float32,
            device=device,
        )
        finger_pose = torch.tensor(
            getattr(logger_data, f"mano_{side}_finger_pose"),
            dtype=torch.float32,
            device=device,
        )
        betas = torch.tensor(
            getattr(logger_data, f"mano_{side}_betas"),
            dtype=torch.float32,
            device=device,
        )
        results[side] = mano.forward(
            side=side,
            global_orient=global_orient,
            finger_pose=finger_pose,
            transl=trans,
            betas=betas,
        )
    return results


def find_support_usd(input_dir: Path, sequence_id: str) -> Path | None:
    """Try to find a support-surface .usda file for the given sequence.

    Looks in ``<input_dir_parent>/reconstructed_stage/<sequence_id>_support.usda``.
    """
    candidate = input_dir.parent / "reconstructed_stage" / f"{sequence_id}_support.usda"
    if candidate.exists():
        return candidate
    return None


def visualize_one_trajectory(
    viser_server: viser.ViserServer,
    right_sharpa_kinematics: HandKinematics,
    left_sharpa_kinematics: HandKinematics,
    viser_object_handles: dict[str, Any],
    input_dir: Path,
    sequence_id: str,
    trajectory_id: int,
    show_mano: bool = False,
    visualize_contacts: bool = False,
    visualize_fingertip_distances: bool = False,
    support_usd: Path | None = None,
    serializer: Any = None,
    mp4_out_path: Path | None = None,
) -> dict[str, Any]:
    """Load one sequence and visualize playback (hands + objects from object_mesh_paths)."""
    for _, handle in viser_object_handles.items():
        handle.remove()
    viser_object_handles.clear()

    # Auto-discover or use explicit support-surface USD
    usd_path = support_usd or find_support_usd(input_dir, sequence_id)
    if usd_path is not None:
        if not _USD_AVAILABLE:
            print(
                "WARNING: pxr (USD) is not available on this platform; skipping support surfaces."
            )
        else:
            print(f"  Loading support surfaces from {usd_path}")
            support_handles = load_support_surfaces_from_usd(viser_server, usd_path)
            viser_object_handles.update(support_handles)

    contact_points_handles: list[Any] = []

    logger_data = ManoSharpaData.from_parquet(
        root_path=str(input_dir),
        filters=[("sequence_id", "=", sequence_id)],
        trajectory_id=trajectory_id,
    )
    H = len(logger_data.robot_right_wrist_position)

    mano: MANO | None = None
    mano_results: dict[str, dict[str, torch.Tensor]] | None = None
    if show_mano:
        mano_kwargs = mano_kwargs_from_data(logger_data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mano = MANO(gender="neutral", device=device, **mano_kwargs)
        print(f"MANO model initialized ({mano_kwargs}) on {device}")
        mano_results = run_mano_forward_from_data(mano, logger_data, device)

    # Optional: world frame
    viser_object_handles["frame"] = viser_server.scene.add_frame(
        name="/object/frame",
        position=np.array([0, 0, 0]),
        wxyz=np.array([1, 0, 0, 0]),
        axes_length=0.2,
        axes_radius=0.007,
    )

    # Object meshes from schema paths (one per body); placeholders for any missing
    object_mesh_paths = getattr(logger_data, "object_mesh_paths", None) or []
    if object_mesh_paths and len(object_mesh_paths) == len(
        logger_data.object_body_names
    ):
        handles = load_object_meshes_from_paths(
            viser_server,
            object_mesh_paths,
            logger_data.object_body_names,
        )
        viser_object_handles.update(handles)
    for part in logger_data.object_body_names:
        if part not in viser_object_handles:
            frame_handle = viser_server.scene.add_frame(
                name=f"/object/{part}",
                position=np.array([0.0, 0.0, 0.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                show_axes=False,
            )
            viser_server.scene.add_icosphere(
                name=f"/object/{part}/placeholder",
                radius=0.02,
                color=(128, 128, 128),
            )
            viser_object_handles[part] = frame_handle

    # Optional: offline MP4 renderer mirroring the viser scene.
    video_renderer = None
    if mp4_out_path is not None:
        # Same directory as this script — Python puts scripts/retarget/ on
        # sys.path automatically when vis_retargeted.py is invoked directly.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _offline_video import (  # type: ignore[import-not-found]  # noqa: PLC0415
            OfflineVideoRenderer,
        )

        video_renderer = OfflineVideoRenderer(fps=int(round(float(logger_data.fps))))
        video_renderer.add_robot("right", right_sharpa_kinematics)
        video_renderer.add_robot("left", left_sharpa_kinematics)
        for obj_idx, obj_name in enumerate(logger_data.object_body_names):
            if obj_idx < len(object_mesh_paths) and object_mesh_paths[obj_idx]:
                try:
                    obj_mesh = trimesh.load(object_mesh_paths[obj_idx], force="mesh")
                    if getattr(logger_data, "dataset", None) == "taco":
                        obj_mesh.vertices *= 0.01
                    video_renderer.add_object(obj_name, obj_mesh)
                except Exception as e:  # noqa: BLE001
                    print(f"  [mp4] skip object {obj_name}: {e}")
        # Auto-fit the camera to the trajectory: use object positions + robot
        # wrist positions across every frame so the subject stays in frame.
        frame_points = [
            np.asarray(logger_data.object_body_position).reshape(-1, 3),
            np.asarray(logger_data.robot_right_wrist_position),
            np.asarray(logger_data.robot_left_wrist_position),
        ]
        video_renderer.fit_camera(np.concatenate(frame_points, axis=0))

    for frame_id in range(H):
        # Right hand
        right_qpos = right_sharpa_kinematics.robot.q0.copy()
        right_qpos[:3] = np.array(logger_data.robot_right_wrist_position[frame_id])
        right_qpos[3:7] = np.array(logger_data.robot_right_wrist_wxyz[frame_id])[
            [1, 2, 3, 0]
        ]
        right_qpos[7:] = np.array(logger_data.robot_right_finger_joints[frame_id])
        right_sharpa_kinematics.visualize(viser_server, right_qpos)
        # Left hand
        left_qpos = left_sharpa_kinematics.robot.q0.copy()
        left_qpos[:3] = np.array(logger_data.robot_left_wrist_position[frame_id])
        left_qpos[3:7] = np.array(logger_data.robot_left_wrist_wxyz[frame_id])[
            [1, 2, 3, 0]
        ]
        left_qpos[7:] = np.array(logger_data.robot_left_finger_joints[frame_id])
        left_sharpa_kinematics.visualize(viser_server, left_qpos)

        # Update object poses from Parquet.
        # Use add_frame() (a full "add" message) rather than property-setter
        # updates so the viser recorder replays positions correctly on every
        # loop iteration — delta-update messages are not re-sent on loop.
        for object_body_idx, object_body_name in enumerate(
            logger_data.object_body_names
        ):
            if object_body_name not in viser_object_handles:
                continue
            viser_server.scene.add_frame(
                name=f"/object/{object_body_name}",
                position=np.asarray(
                    logger_data.object_body_position[frame_id][object_body_idx]
                ),
                wxyz=np.asarray(
                    logger_data.object_body_wxyz[frame_id][object_body_idx]
                ),
                show_axes=False,
            )

        # Fingertip distance spheres (if available)
        if visualize_fingertip_distances:
            for side, joints_data, dist_data in [
                (
                    "right",
                    logger_data.mano_right_joints,
                    logger_data.mano_right_tips_distance,
                ),
                (
                    "left",
                    logger_data.mano_left_joints,
                    logger_data.mano_left_tips_distance,
                ),
            ]:
                if not dist_data:
                    continue
                fingertip_positions = np.array(joints_data[frame_id])[
                    MANO_FINGERTIP_INDICES
                ]
                distances = dist_data[frame_id]
                for i, finger_name in enumerate(FINGER_NAMES):
                    viser_server.scene.add_icosphere(
                        name=f"/tips/{side}_{finger_name}",
                        radius=0.005,
                        color=distance_to_color(distances[i]),
                        position=fingertip_positions[i],
                    )

        # Link contact visualization
        if visualize_contacts:
            for contact_handle in contact_points_handles:
                contact_handle.remove()
            contact_points_handles.clear()
            for side, contact_positions, contact_normals in [
                (
                    "right",
                    logger_data.mano_right_object_contact_positions[frame_id],
                    logger_data.mano_right_link_contact_normals[frame_id],
                ),
                (
                    "left",
                    logger_data.mano_left_object_contact_positions[frame_id],
                    logger_data.mano_left_link_contact_normals[frame_id],
                ),
            ]:
                for contact_position, (link_name, _) in zip(
                    contact_positions, MANO_HAND_LINKS.items(), strict=False
                ):
                    if np.sum(contact_position) > 0.0:
                        contact_handle = viser_server.scene.add_icosphere(
                            name=f"/mano/{side}_contact_points/{link_name}",
                            radius=0.005,
                            color=np.array([0, 0, 255]),
                            position=np.array(contact_position[:3]),
                        )
                        contact_points_handles.append(contact_handle)
                normal_lines = np.stack(
                    [
                        np.asarray(contact_positions),
                        (
                            np.asarray(contact_positions)
                            + np.asarray(contact_normals) * 0.01
                        ),
                    ],
                    axis=1,
                )
                normal_handle = viser_server.scene.add_line_segments(
                    name=f"/mano/{side}_contact_normals",
                    points=normal_lines,
                    colors=np.zeros_like(normal_lines),
                    line_width=2.0,
                )
                contact_points_handles.append(normal_handle)

        # MANO hand mesh + joint frames
        if mano is not None and mano_results is not None:
            for side in ("right", "left"):
                mano.visualize(
                    viser_server,
                    side,
                    vertices=mano_results[side]["vertices"][frame_id],
                    faces=mano_results[side]["faces"],
                    joints=mano_results[side]["joints"][frame_id],
                    joints_wxyz=mano_results[side]["joints_wxyz"][frame_id],
                )

        # Offline MP4 capture — mirror the current viser scene state.
        if video_renderer is not None:
            video_renderer.update_robot("right", right_qpos)
            video_renderer.update_robot("left", left_qpos)
            for object_body_idx, object_body_name in enumerate(
                logger_data.object_body_names
            ):
                pos = np.asarray(
                    logger_data.object_body_position[frame_id][object_body_idx]
                )
                wxyz = np.asarray(
                    logger_data.object_body_wxyz[frame_id][object_body_idx]
                )
                # Convert wxyz -> scipy xyzw, then to rotation matrix.
                rot = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = pos
                video_renderer.update_object(object_body_name, T)
            if mano is not None and mano_results is not None:
                for side in ("right", "left"):
                    video_renderer.update_mano(
                        side,
                        mano_results[side]["vertices"][frame_id].detach().cpu().numpy(),
                        mano_results[side]["faces"].detach().cpu().numpy(),
                    )
            video_renderer.capture()

        dt = 1.0 / logger_data.fps
        if serializer is not None:
            serializer.insert_sleep(dt)
        else:
            time.sleep(dt)

    if video_renderer is not None:
        video_renderer.save(mp4_out_path)
        video_renderer.close()

    return viser_object_handles


def _build_viser_client(html_dir: Path) -> None:
    """Copy the viser JS client build into html_dir/viser-client/.

    Copies directly from the installed viser package rather than calling the
    viser-build-client binary, which fails in Isaac Sim's environment because
    imageio is only on sys.path when launched via isaaclab.sh.
    """
    client_dir = html_dir / "viser-client"

    viser_client_build = Path(viser.__file__).parent / "client" / "build"
    if not viser_client_build.is_dir():
        print(
            f"WARNING: viser client build not found at {viser_client_build}; skipping."
        )
        sys.exit(-1)
        return

    # ``dirs_exist_ok`` makes this idempotent under parallel shards — the first
    # shard populates the dir and the others no-op over the same files instead
    # of racing on ``client_dir.exists()`` + ``copytree``.
    print(f"Copying viser client → {client_dir}")
    shutil.copytree(viser_client_build, client_dir, dirs_exist_ok=True)
    print(f"  Viser client ready at {client_dir}")


def main(args: argparse.Namespace) -> None:
    """List or use sequence, setup kinematics, run visualization."""
    if args.input_dir is not None:
        input_dir = args.input_dir
    elif args.dataset is not None:
        input_dir = HUMAN_MOTION_DATA_DIR / DATASET_DIRS[args.dataset]
    else:
        input_dir = HUMAN_MOTION_DATA_DIR / DATASET_DIRS["arctic"]

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    available = list_sequence_ids(str(input_dir))
    if not available:
        raise ValueError(f"No sequences found in {input_dir}")

    sequence_ids = filter_sequence_ids(available, args)
    if not sequence_ids:
        # Small datasets sharded many ways routinely produce empty buckets
        # (md5 over ~30 sequences into 8 shards often leaves one with zero).
        # Treat that as a clean no-op so sibling shards can finish the stage.
        num_shards = getattr(args, "num_shards", 1) or 1
        shard_id = getattr(args, "shard_id", 0) or 0
        if num_shards > 1:
            print(
                f"[vis_retargeted] Shard {shard_id}/{num_shards}: "
                f"no sequences matched (available: {len(available)}); exiting cleanly."
            )
            return
        raise ValueError(
            f"No sequences match the provided filters (available: {len(available)})."
        )
    if len(sequence_ids) == len(available):
        print(
            f"No filter specified, will iterate through all {len(sequence_ids)} sequences."
        )
    else:
        print(f"Filter selected {len(sequence_ids)}/{len(available)} sequences.")

    # Resolve HTML output directory and build the viser client (once)
    html_dir: Path | None = None
    if args.save_html:
        _html_dir: Path = args.html_dir or DEFAULT_HTML_DIR
        (_html_dir / "recordings").mkdir(parents=True, exist_ok=True)
        _build_viser_client(_html_dir)
        index_src = DEFAULT_HTML_DIR / "index.html"
        index_dst = _html_dir / "index.html"
        if index_src.exists() and not index_dst.exists():
            shutil.copy2(index_src, index_dst)
        html_dir = _html_dir

    viser_server = viser.ViserServer()
    viser_object_handles: dict[str, Any] = {}

    support_usd = args.support_usd
    if support_usd is not None and not support_usd.exists():
        raise FileNotFoundError(f"Support USD not found: {support_usd}")

    right_sharpa_kinematics = setup_sharpa_kinematics(
        side="right", frame_tasks_converged_threshold=1e-6
    )
    left_sharpa_kinematics = setup_sharpa_kinematics(
        side="left", frame_tasks_converged_threshold=1e-6
    )

    for seq_idx, sequence_id in enumerate(sequence_ids):
        print(
            f"[{seq_idx + 1}/{len(sequence_ids)}] Visualizing sequence: {sequence_id}"
        )
        serializer = viser_server.get_scene_serializer() if args.save_html else None
        mp4_out_path = (
            html_dir / "recordings" / f"{sequence_id}.mp4"
            if args.save_mp4 and html_dir is not None
            else None
        )
        visualize_one_trajectory(
            viser_server,
            right_sharpa_kinematics,
            left_sharpa_kinematics,
            viser_object_handles,
            input_dir=input_dir,
            sequence_id=sequence_id,
            trajectory_id=args.trajectory_id,
            show_mano=args.show_mano,
            visualize_contacts=args.visualize_contacts,
            visualize_fingertip_distances=args.visualize_fingertip_distances,
            support_usd=support_usd,
            serializer=serializer,
            mp4_out_path=mp4_out_path,
        )
        if serializer is not None and html_dir is not None:
            out_path = html_dir / "recordings" / f"{sequence_id}.viser"
            out_path.write_bytes(serializer.serialize())
            print(f"  Saved → {out_path}")
            print(
                f"\n  Playback instructions:\n"
                f"    scp -r {html_dir} <local-machine>:/tmp/viser_replay\n"
                f"    cd /tmp/viser_replay && python -m http.server 8000\n"
                f"    http://localhost:8000/viser-client/"
                f"?playbackPath=http://localhost:8000/recordings/{sequence_id}.viser\n"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
