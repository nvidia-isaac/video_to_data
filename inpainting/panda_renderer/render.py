"""Render bimanual Panda arms from a parallel-jaw target trajectory."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from inpainting.mecka_panda.contracts import (
    ROBOT_RENDER_SCHEMA,
    artifact,
    load_npz,
    validate_parallel_jaw_arrays,
    write_json_atomic,
)
from inpainting.mecka_panda.video_io import Mp4Writer
from inpainting.panda_renderer.kinematics import (
    PandaIK,
    build_panda_model,
    gravity_axes,
)

RGB_FILENAME = "robot_rgb.mp4"
MASK_FILENAME = "robot_mask.npy"
DEPTH_FILENAME = "robot_depth.npy"
METADATA_FILENAME = "render_metadata.json"
DEFAULT_PANDA_DIR = (
    Path(__file__).resolve().parents[2]
    / "debug"
    / "third_party"
    / "mujoco_menagerie"
    / "franka_emika_panda"
)


def _load_intrinsic(path: str | Path) -> np.ndarray:
    intrinsic = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic must be a finite (3,3) NumPy array")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("intrinsic focal lengths must be positive")
    return intrinsic


def _load_camera_rotation(path: str | Path, frame_count: int) -> np.ndarray:
    values = np.asarray(np.load(Path(path), allow_pickle=False), dtype=np.float64)
    if values.shape != (frame_count, 4) or not np.isfinite(values).all():
        raise ValueError(
            f"camera_to_world_xyzw must be finite with shape ({frame_count},4)"
        )
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("camera_to_world_xyzw contains non-unit quaternions")
    return values


def _load_rig(path: str | Path) -> dict[str, float]:
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    required = {"x", "y", "z", "pitch", "distance"}
    if set(raw) != required:
        raise ValueError(f"rig config keys must be exactly {sorted(required)}")
    values = {key: float(raw[key]) for key in required}
    if not np.isfinite(list(values.values())).all() or values["distance"] <= 0:
        raise ValueError("rig values must be finite and distance must be positive")
    return values


def _render_layer(
    renderer: mujoco.Renderer, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera="ego")
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render()[:, :, 0]
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    depth = renderer.render().astype(np.float32, copy=True)
    renderer.disable_depth_rendering()
    rgb = renderer.render()[:, :, ::-1].copy()
    mask = segmentation >= 0
    rgb[~mask] = 0
    depth[~mask] = np.inf
    return rgb, mask, depth


def execute(
    *,
    trajectory: str | Path,
    intrinsic: str | Path,
    camera_to_world_xyzw: str | Path,
    rig_config: str | Path,
    output_dir: str | Path,
    width: int,
    height: int,
    fps: float,
    panda_dir: str | Path = DEFAULT_PANDA_DIR,
    ik_backend: str = "dls",
    orientation_weight: float = 0.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Solve, render, validate, and atomically publish one robot bundle."""
    if ik_backend not in {"dls", "hybrid"}:
        raise ValueError("ik_backend must be 'dls' or 'hybrid'")
    targets_path = Path(trajectory).expanduser().resolve()
    targets = load_npz(targets_path)
    frame_count = validate_parallel_jaw_arrays(targets)
    camera_rotations = _load_camera_rotation(camera_to_world_xyzw, frame_count)
    intrinsic_matrix = _load_intrinsic(intrinsic)
    rig = _load_rig(rig_config)
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height and fps must be positive")
    if not (0 <= intrinsic_matrix[0, 2] < width and 0 <= intrinsic_matrix[1, 2] < height):
        raise ValueError("intrinsic principal point is outside the render geometry")
    fovy = 2.0 * np.degrees(np.arctan((height / 2.0) / intrinsic_matrix[1, 1]))

    output = Path(output_dir).expanduser().resolve()
    final_paths = {
        "rgb": output / RGB_FILENAME,
        "mask": output / MASK_FILENAME,
        "depth": output / DEPTH_FILENAME,
        "metadata": output / METADATA_FILENAME,
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    temporary = {
        "rgb": output / ".robot_rgb.partial.mp4",
        "mask": output / ".robot_mask.partial.npy",
        "depth": output / ".robot_depth.partial.npy",
    }
    for path in temporary.values():
        path.unlink(missing_ok=True)

    model = build_panda_model(panda_dir, fovy, width, height)
    solvers = {side: PandaIK(model) for side in ("left", "right")}
    renderers = {
        side: mujoco.Renderer(model, height=height, width=width)
        for side in ("left", "right")
    }
    masks = np.lib.format.open_memmap(
        temporary["mask"],
        mode="w+",
        dtype=np.bool_,
        shape=(frame_count, height, width),
    )
    depths = np.lib.format.open_memmap(
        temporary["depth"],
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, height, width),
    )
    writer = Mp4Writer(temporary["rgb"], fps, (width, height))
    previous_q: dict[str, np.ndarray | None] = {"left": None, "right": None}
    residuals: dict[str, list[float]] = {"left": [], "right": []}
    joint_steps: list[float] = []
    ssik_hits = 0
    ssik_fallbacks = 0
    try:
        for frame in range(frame_count):
            composite = np.zeros((height, width, 3), dtype=np.uint8)
            zbuffer = np.full((height, width), np.inf, dtype=np.float32)
            combined_mask = np.zeros((height, width), dtype=np.bool_)
            down, back, right = gravity_axes(camera_rotations[frame])
            for side in ("left", "right"):
                if not targets[f"{side}_valid"][frame]:
                    previous_q[side] = None
                    continue
                position = targets[f"{side}_position"][frame]
                semantic_rotation = Rotation.from_quat(
                    targets[f"{side}_wxyz"][frame], scalar_first=True
                ).as_matrix()
                aperture = float(targets[f"{side}_aperture_m"][frame])
                sign = 1.0 if side == "right" else -1.0
                base = (
                    rig["x"] * right
                    + rig["y"] * back
                    + rig["z"] * down
                    + sign * 0.5 * rig["distance"] * right
                )
                base_up = Rotation.from_rotvec(
                    np.deg2rad(rig["pitch"]) * right
                ).apply(-down)
                solver = solvers[side]
                solver.set_base(
                    base,
                    position,
                    up_camera=base_up,
                    reset_arm=previous_q[side] is None,
                )
                residual: float | None = None
                if ik_backend == "hybrid":
                    try:
                        residual = solver.solve_ssik(
                            position, semantic_rotation, aperture
                        )
                    except RuntimeError:
                        residual = None
                    if residual is None:
                        ssik_fallbacks += 1
                        if solver._ssik is not None:
                            solver.data.qpos[solver.arm_qadr] = solver._ssik_seed
                    else:
                        ssik_hits += 1
                if residual is None:
                    residual = solver.solve_dls(
                        position,
                        semantic_rotation,
                        aperture,
                        previous_q=previous_q[side],
                        elbow_outward=sign * right,
                        orientation_weight=orientation_weight,
                    )
                current_q = solver.data.qpos[solver.arm_qadr].copy()
                if previous_q[side] is not None:
                    joint_steps.append(
                        float(np.max(np.abs(current_q - previous_q[side])))
                    )
                previous_q[side] = current_q
                residuals[side].append(residual)
                rgb, mask, depth = _render_layer(renderers[side], solver.data)
                take = mask & (depth < zbuffer)
                composite[take] = rgb[take]
                zbuffer[take] = depth[take]
                combined_mask |= take
            writer.write(composite)
            masks[frame] = combined_mask
            depths[frame] = zbuffer
    finally:
        writer.close()
        masks.flush()
        depths.flush()
        del masks
        del depths
        renderers.clear()

    os.replace(temporary["rgb"], final_paths["rgb"])
    os.replace(temporary["mask"], final_paths["mask"])
    os.replace(temporary["depth"], final_paths["depth"])
    metadata = {
        "schema_version": ROBOT_RENDER_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "geometry": {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "ik": {
            "backend": ik_backend,
            "orientation_weight": orientation_weight,
            "ssik_hits": ssik_hits,
            "ssik_fallbacks": ssik_fallbacks,
            "max_joint_step_rad": max(joint_steps, default=0.0),
            "residual_m": {
                side: {
                    "median": float(np.median(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                }
                for side, values in residuals.items()
            },
        },
        "rig": rig,
        "source": {
            "trajectory": artifact(targets_path),
            "intrinsic": artifact(intrinsic),
            "camera_to_world_xyzw": artifact(camera_to_world_xyzw),
            "rig_config": artifact(rig_config),
            "panda_xml": artifact(Path(panda_dir) / "panda.xml"),
            "implementation": {
                "render": artifact(__file__),
                "kinematics": artifact(Path(__file__).with_name("kinematics.py")),
            },
        },
        "output": {
            key: artifact(path)
            for key, path in final_paths.items()
            if key != "metadata"
        },
    }
    write_json_atomic(final_paths["metadata"], metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--intrinsic", required=True, type=Path)
    parser.add_argument("--camera-to-world-xyzw", required=True, type=Path)
    parser.add_argument("--rig-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--panda-dir", type=Path, default=DEFAULT_PANDA_DIR)
    parser.add_argument("--ik", choices=("dls", "hybrid"), default="dls")
    parser.add_argument("--orientation-weight", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    metadata = execute(
        trajectory=args.trajectory,
        intrinsic=args.intrinsic,
        camera_to_world_xyzw=args.camera_to_world_xyzw,
        rig_config=args.rig_config,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        panda_dir=args.panda_dir,
        ik_backend=args.ik,
        orientation_weight=args.orientation_weight,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

