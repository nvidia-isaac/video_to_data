#!/usr/bin/env python3
"""Scale a V2D result bundle without baking scale into pose matrices."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np


def _fmt_float(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def _scale_obj_vertices(src: Path, dst: Path, scale: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="surrogateescape") as src_f, dst.open(
        "w", encoding="utf-8", errors="surrogateescape"
    ) as dst_f:
        for line in src_f:
            if line.startswith("v "):
                parts = line.rstrip("\n").split()
                if len(parts) >= 4:
                    xyz = [float(parts[i]) * scale for i in range(1, 4)]
                    rest = parts[4:]
                    dst_f.write(
                        "v "
                        + " ".join(_fmt_float(v) for v in xyz)
                        + ((" " + " ".join(rest)) if rest else "")
                        + "\n"
                    )
                    continue
            dst_f.write(line)


def _copy_sidecar_assets(src_dir: Path, dst_dir: Path, mesh_name: str) -> None:
    for path in src_dir.iterdir():
        if path.name in {"result.npz", mesh_name, "threejs_scene"}:
            continue
        target = dst_dir / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def _valid_frames(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 3:
        return np.array([], dtype=bool)
    return np.isfinite(arr).all(axis=tuple(range(1, arr.ndim)))


def _constant_similarity_scale(transforms: np.ndarray, name: str) -> float:
    valid = _valid_frames(transforms)
    if not valid.any():
        return 1.0

    linear = transforms[valid, :3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    per_frame = singular_values.mean(axis=1)
    spread = np.max(np.abs(singular_values - per_frame[:, None]))
    if spread > 1e-4:
        raise ValueError(
            f"{name} is not a uniform similarity transform; max singular-value spread is {spread:.6g}"
        )

    scale = float(np.median(per_frame))
    if not np.allclose(per_frame, scale, rtol=1e-4, atol=1e-5):
        raise ValueError(f"{name} has non-constant scale across frames")
    return scale


def _rigidify_similarity(transforms: np.ndarray, similarity_scale: float) -> np.ndarray:
    out = transforms.copy()
    if not np.isfinite(similarity_scale) or similarity_scale <= 0:
        raise ValueError(f"Invalid similarity scale {similarity_scale}")
    valid = _valid_frames(out)
    if valid.any():
        out[valid, :3, :3] /= similarity_scale
    return out


def _scale_translation(arr: np.ndarray, scale: float) -> np.ndarray:
    out = arr.copy()
    if out.ndim == 2 and out.shape == (4, 4):
        if np.isfinite(out).all():
            out[:3, 3] *= scale
    elif out.ndim == 3 and out.shape[-2:] == (4, 4):
        valid = _valid_frames(out)
        if valid.any():
            out[valid, :3, 3] *= scale
    elif out.ndim >= 1 and out.shape[-1] == 3:
        finite = np.isfinite(out).all(axis=-1)
        out[finite] *= scale
    else:
        raise ValueError(f"Cannot scale translations for array with shape {arr.shape}")
    return out


def _compose_world_transforms(camera_to_world: np.ndarray, local_to_camera: np.ndarray) -> np.ndarray:
    out = np.full_like(local_to_camera, np.nan)
    valid = _valid_frames(camera_to_world) & _valid_frames(local_to_camera)
    if valid.any():
        out[valid] = camera_to_world[valid] @ local_to_camera[valid]
    return out


def _mesh_extents(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        return np.zeros(3, dtype=np.float64)
    verts = np.asarray(vertices, dtype=np.float64)
    return verts.max(axis=0) - verts.min(axis=0)


def _write_zip(bundle_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                zf.write(path, Path(bundle_dir.name) / path.relative_to(bundle_dir))


def scale_bundle(src_dir: Path, dst_dir: Path, scale: float, force: bool) -> dict[str, object]:
    if scale <= 0:
        raise ValueError("--scale must be positive")
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    src_npz = src_dir / "result.npz"
    src_mesh = src_dir / "mesh.obj"
    if not src_npz.exists():
        raise FileNotFoundError(src_npz)
    if not src_mesh.exists():
        raise FileNotFoundError(src_mesh)

    if dst_dir.exists():
        if not force:
            raise FileExistsError(f"{dst_dir} already exists; pass --force to overwrite it")
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    data = dict(np.load(src_npz, allow_pickle=False))
    object_pose_scale = _constant_similarity_scale(data["object_to_camera_transform"], "object_to_camera_transform")
    input_object_scale = float(np.asarray(data.get("object_scale", np.array(1.0))).reshape(-1)[0])
    mesh_vertex_scale = scale * object_pose_scale * input_object_scale

    data["camera_to_world_transform"] = _scale_translation(data["camera_to_world_transform"], scale)
    data["object_to_camera_transform"] = _rigidify_similarity(data["object_to_camera_transform"], object_pose_scale)
    data["object_to_camera_transform"] = _scale_translation(data["object_to_camera_transform"], scale)
    data["object_to_world_transform"] = _compose_world_transforms(
        data["camera_to_world_transform"], data["object_to_camera_transform"]
    )
    data["object_scale"] = np.asarray(1.0, dtype=np.float32)

    for side in ("left", "right"):
        wrist_trans_key = f"hand_{side}_wrist_trans_in_camera"
        wrist_cam_key = f"hand_{side}_wrist_to_camera_transform"
        wrist_world_key = f"hand_{side}_wrist_to_world_transform"
        hand_scale_key = f"hand_{side}_scale"
        if wrist_trans_key in data:
            data[wrist_trans_key] = _scale_translation(data[wrist_trans_key], scale)
        if wrist_cam_key in data:
            data[wrist_cam_key] = _scale_translation(data[wrist_cam_key], scale)
        if wrist_world_key in data and wrist_cam_key in data:
            data[wrist_world_key] = _compose_world_transforms(
                data["camera_to_world_transform"], data[wrist_cam_key]
            )
        elif wrist_world_key in data:
            data[wrist_world_key] = _scale_translation(data[wrist_world_key], scale)
        if hand_scale_key in data:
            data[hand_scale_key] = data[hand_scale_key] * scale

    if "gravity_alignment_transform" in data:
        data["gravity_alignment_transform"] = _scale_translation(data["gravity_alignment_transform"], scale)

    np.savez_compressed(dst_dir / "result.npz", **data)
    _scale_obj_vertices(src_mesh, dst_dir / "mesh.obj", mesh_vertex_scale)
    _copy_sidecar_assets(src_dir, dst_dir, "mesh.obj")

    manifest_path = dst_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {}
    manifest.update(
        {
            "scaled_from": str(src_dir),
            "scene_scale_applied": scale,
            "object_transform_convention": "rigid_pose_no_scale",
            "object_scale_saved_in_result_npz": 1.0,
            "input_object_scale_baked_into_mesh": input_object_scale,
            "object_pose_scale_baked_into_mesh": object_pose_scale,
            "object_mesh_vertex_scale_applied": mesh_vertex_scale,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "source_mesh_extents": _mesh_extents(src_mesh).tolist(),
        "output_mesh_extents": _mesh_extents(dst_dir / "mesh.obj").tolist(),
        "scene_scale_applied": scale,
        "input_object_scale_baked_into_mesh": input_object_scale,
        "object_pose_scale_baked_into_mesh": object_pose_scale,
        "object_mesh_vertex_scale_applied": mesh_vertex_scale,
        "object_scale_out": float(np.asarray(data["object_scale"])),
    }


def export_threejs(bundle_dir: Path, mano_assets_root: Path | None) -> None:
    reconstruction_root = Path(__file__).resolve().parents[1]
    no_depth_dir = Path("/tmp/v2d_no_depth")
    no_depth_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "modules/v2d_pipelines/export_result_threejs_scene.py",
        "--result-dir",
        str(bundle_dir.resolve()),
        "--depth-dir",
        str(no_depth_dir),
        "--output-dir",
        str((bundle_dir / "threejs_scene").resolve()),
        "--max-depth-points",
        "0",
    ]
    if mano_assets_root is not None:
        cmd.extend(["--mano-assets-root", str(mano_assets_root.resolve())])
    subprocess.run(cmd, cwd=reconstruction_root, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale a result bundle while keeping object transforms as rigid poses."
    )
    parser.add_argument("input_dir", type=Path, help="Input result bundle directory")
    parser.add_argument("--scale", type=float, required=True, help="Scene scale factor")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output bundle directory. Defaults to <input_dir>_scaled",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory if it exists")
    parser.add_argument("--zip", action="store_true", help="Create <output-dir>.zip")
    parser.add_argument("--zip-path", type=Path, help="Explicit zip output path")
    parser.add_argument(
        "--export-threejs",
        action="store_true",
        help="Regenerate a Three.js viewer without depth points",
    )
    parser.add_argument(
        "--mano-assets-root",
        type=Path,
        help="MANO assets root for --export-threejs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir.with_name(input_dir.name + "_scaled")

    summary = scale_bundle(input_dir, output_dir, args.scale, args.force)
    if args.export_threejs:
        export_threejs(output_dir, args.mano_assets_root)

    zip_path = None
    if args.zip or args.zip_path:
        zip_path = (args.zip_path.resolve() if args.zip_path else output_dir.with_suffix(".zip"))
        _write_zip(output_dir, zip_path)

    print(json.dumps({**summary, "output_dir": str(output_dir), "zip_path": str(zip_path) if zip_path else None}, indent=2))


if __name__ == "__main__":
    main()
