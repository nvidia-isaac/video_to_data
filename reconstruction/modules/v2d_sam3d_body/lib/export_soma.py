# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export MHR parameters from v2d sam3d_body to SOMA format.

Reads the multi-view MHR output (.pt files) and uses SOMA-X's PoseInversion
to produce a SOMA-format .npz file.

Two paths for obtaining MHR vertices:
  Path A: Pre-computed vertices from --mesh_path (undo Y/Z flip; translation baked in)
  Path B: MHR JIT forward pass from --params_path (fallback when --mesh_path absent)

Usage:
  python -m v2d.sam3d_body.lib.export_soma \
      --params_path /path/to/mhr_params_mv.pt \
      --mesh_path /path/to/mhr_mesh_mv.pt \
      --output_path /path/to/soma_params.npz

  python -m v2d.sam3d_body.lib.export_soma \
      --params_path /path/to/mhr_params_mv.pt \
      --weights_dir /path/to/sam3d_body_weights \
      --output_path /path/to/soma_params.npz
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from soma import SOMALayer
from soma.geometry.rig_utils import remove_joint_orient_local
from soma.geometry.transforms import matrix_to_rotvec
from soma.io import save_soma_npz
from soma.pose_inversion import PoseInversion
from soma.units import Unit

logger = logging.getLogger(__name__)

MHR_JIT_RELPATH = "sam-3d-body-dinov3/assets/mhr_model.pt"


def load_vertices_from_mesh(
    mesh_path: Path,
) -> tuple[torch.Tensor, np.ndarray]:
    """Path A: load pre-computed vertices from mhr_mesh_mv.pt.

    pred_vertices already includes world translation (from global_trans in the
    MHR forward pass). Undoes the Y/Z flip to recover MHR-native positioned vertices.

    Returns ((N, V, 3) vertices, (F, 3) faces).
    """
    logger.info("Loading pre-computed vertices from %s", mesh_path)
    mesh_data = torch.load(mesh_path, map_location="cpu", weights_only=True)
    pred_vertices = mesh_data["pred_vertices"]  # (N, V, 3) Y/Z flipped, includes translation
    faces = mesh_data["faces"].cpu().numpy()  # (F, 3)

    verts = pred_vertices.clone()
    verts[..., [1, 2]] *= -1  # undo Y/Z flip -> MHR native
    return verts, faces


def load_vertices_from_forward(
    params_data: dict, weights_dir: Path, device: str = "cuda",
) -> torch.Tensor:
    """Path B: run MHR JIT forward pass to obtain vertices.

    mhr_model_params[0:3] already contains global_trans * 10 from the MV pipeline.
    Zeros out flexible bone-length parameters not representable in SOMA.
    """
    mhr_path = weights_dir / MHR_JIT_RELPATH
    if not mhr_path.exists():
        raise FileNotFoundError(
            f"MHR JIT model not found at {mhr_path}. "
            "Pass --weights_dir pointing to the sam3d_body weights directory."
        )

    logger.info("Running MHR forward pass (JIT model: %s)", mhr_path)
    mhr_jit = torch.jit.load(str(mhr_path), map_location=device)

    shape_params = params_data["shape_params"].float()  # (N, 45)
    model_params = params_data["mhr_model_params"].float().clone()  # (N, 204)

    model_params[:, 130:136] = 0.0

    face_expr = torch.zeros(1, 72, device=device)

    N = shape_params.shape[0]
    batch_size = 64
    all_verts = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ic = shape_params[start:end].to(device)
        mp = model_params[start:end].to(device)
        fe = face_expr.expand(end - start, -1)
        with torch.no_grad():
            verts, _ = mhr_jit(ic, mp, fe)
        all_verts.append(verts.cpu())

    return torch.cat(all_verts, dim=0)


def _reconstruct_soma_vertices(
    soma: SOMALayer,
    rotations: torch.Tensor,
    root_transl: torch.Tensor,
    shape_params: torch.Tensor,
    scale_params: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Run SOMA forward pass to reconstruct vertices from inverted poses."""
    N = rotations.shape[0]
    bs = soma.batched_skinning
    all_verts = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        soma.prepare_identity(
            shape_params[start:end].to(device),
            scale_params[start:end].to(device),
        )
        bs.rebind(
            soma._cached_bind_transforms_world,
            soma._cached_rest_shape,
        )
        with torch.no_grad():
            v, _ = bs.pose(
                rotations[start:end].to(device),
                root_transl[start:end].to(device),
                absolute_pose=True,
                return_transforms=True,
            )
        all_verts.append(v.cpu())
    return torch.cat(all_verts, dim=0)


def _compute_look_at_camera(
    vertices: np.ndarray, image_size: tuple[int, int], dist_scale: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a synthetic intrinsic K and extrinsic T for viewing a mesh.

    Returns (K (3,3), T (4,4)) where T is world-from-camera in CV convention
    (X right, Y down, Z forward).
    """
    W, H = image_size
    vmin, vmax = vertices.min(axis=0), vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    extent = np.linalg.norm(vmax - vmin) + 1e-6
    dist = max(extent * dist_scale, 0.5)

    eye = center + np.array([0.0, 0.0, dist])

    # CV convention: Z forward (eye -> center), Y down, X right
    z_cam = center - eye
    z_cam /= np.linalg.norm(z_cam) + 1e-8
    y_cam = np.array([0.0, -1.0, 0.0])  # down in world (MHR Y is up)
    x_cam = np.cross(y_cam, z_cam)
    x_cam /= np.linalg.norm(x_cam) + 1e-8
    y_cam = np.cross(z_cam, x_cam)

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_cam
    T[:3, 1] = y_cam
    T[:3, 2] = z_cam
    T[:3, 3] = eye

    focal = dist * max(W, H) / extent
    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    return K, T


def _render_debug(
    output_path: Path,
    verts_mhr: torch.Tensor,
    mhr_faces: np.ndarray | None,
    verts_soma: torch.Tensor,
    soma_faces: np.ndarray,
    fps: int = 30,
    image_size: tuple[int, int] = (1024, 1024),
) -> None:
    """Render a blended comparison video of MHR (green) vs SOMA (purple) meshes."""
    import cv2
    import trimesh
    from tqdm import tqdm
    from v2d.common.video import get_video_writer
    from v2d.mv.vis.renderer import Renderer

    video_path = output_path.parent / "mhr_soma_comparison.mp4"

    if mhr_faces is None:
        logger.warning("MHR faces unavailable (Path B), using SOMA faces for both meshes")
        mhr_faces = soma_faces

    N = verts_mhr.shape[0]
    # Render in meters to stay within Renderer's zfar=100 clip plane.
    # Input vertices are in centimeters (SOMA working unit).
    verts_mhr_np = verts_mhr.numpy() / 100.0
    verts_soma_np = verts_soma.numpy() / 100.0

    centroids = verts_mhr_np.mean(axis=1, keepdims=True)
    verts_mhr_np = verts_mhr_np - centroids
    verts_soma_np = verts_soma_np - centroids

    K, T = _compute_look_at_camera(verts_mhr_np[0], image_size)
    bg = np.full((*image_size[::-1], 3), 255, dtype=np.uint8)

    color_mhr = [102, 230, 179, 255]
    color_soma = [160, 40, 220, 255]

    logger.info("Rendering debug comparison video -> %s", video_path)
    writer = get_video_writer(video_path, fps=fps, crf=23)
    with Renderer(image_size=image_size) as renderer:
        for i in tqdm(range(N), desc="Rendering SOMA debug"):
            mesh_mhr = trimesh.Trimesh(
                vertices=verts_mhr_np[i], faces=mhr_faces, process=False,
            )
            mesh_mhr.visual.vertex_colors = np.full(
                (len(verts_mhr_np[i]), 4), color_mhr, dtype=np.uint8,
            )
            mesh_soma = trimesh.Trimesh(
                vertices=verts_soma_np[i], faces=soma_faces, process=False,
            )
            mesh_soma.visual.vertex_colors = np.full(
                (len(verts_soma_np[i]), 4), color_soma, dtype=np.uint8,
            )
            frame = renderer.render_overlay([mesh_mhr, mesh_soma], K, T, image=bg)
            frame = (frame * 255.0).astype(np.uint8)
            cv2.putText(frame, "MHR", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(color_mhr[:3]), 2)
            cv2.putText(frame, "SOMA", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(color_soma[:3]), 2)
            writer.write_frame(frame)
    writer.close()
    logger.info("Saved: %s (green=MHR, purple=SOMA)", video_path)


def _render_chamfer_heatmap(
    output_path: Path,
    verts_mhr: torch.Tensor,
    mhr_faces: np.ndarray | None,
    verts_soma: torch.Tensor,
    soma_faces: np.ndarray,
    fps: int = 30,
    image_size: tuple[int, int] = (1024, 1024),
    max_dist_cm: float = 6.0,
    device: str = "cuda",
) -> None:
    """Render a side-by-side bidirectional chamfer-distance heatmap video.

    Left panel: MHR mesh colored by per-vertex distance to the nearest SOMA
    vertex (MHR -> SOMA). Right panel: SOMA mesh colored by per-vertex
    distance to the nearest MHR vertex (SOMA -> MHR). Colormap is JET
    (blue=0, red=max_dist_cm); both panels share the same scale so colors
    are directly comparable. A colorbar legend is overlaid on each frame.
    """
    import cv2
    import trimesh
    from tqdm import tqdm
    from v2d.common.video import get_video_writer
    from v2d.mv.vis.renderer import Renderer

    video_path = output_path.parent / "mhr_soma_chamfer_heatmap.mp4"

    if mhr_faces is None:
        logger.warning("MHR faces unavailable, skipping chamfer heatmap")
        return

    N = verts_mhr.shape[0]
    verts_mhr_np = verts_mhr.numpy() / 100.0
    verts_soma_np = verts_soma.numpy() / 100.0
    centroids = verts_mhr_np.mean(axis=1, keepdims=True)
    verts_mhr_np = verts_mhr_np - centroids
    verts_soma_np = verts_soma_np - centroids

    # Per-frame bidirectional nearest-vertex distance (cm).
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    mhr_dists_all = np.empty((N, verts_mhr.shape[1]), dtype=np.float32)
    soma_dists_all = np.empty((N, verts_soma.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            mhr_t = verts_mhr[i].to(device_t)
            soma_t = verts_soma[i].to(device_t)
            d = torch.cdist(mhr_t, soma_t)
            mhr_dists_all[i] = d.min(dim=1).values.cpu().numpy()
            soma_dists_all[i] = d.min(dim=0).values.cpu().numpy()

    logger.info(
        "Chamfer heatmap scale: 0 to %.2f cm (JET, blue=0 red=max); "
        "MHR->SOMA observed max=%.2f cm, SOMA->MHR observed max=%.2f cm",
        max_dist_cm, float(mhr_dists_all.max()), float(soma_dists_all.max()),
    )

    K, T = _compute_look_at_camera(verts_mhr_np[0], image_size)
    bg = np.full((*image_size[::-1], 3), 255, dtype=np.uint8)

    def colorize(d_cm: np.ndarray) -> np.ndarray:
        norm = np.clip(d_cm / max_dist_cm, 0.0, 1.0) * 255.0
        bgr = cv2.applyColorMap(norm.astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
        rgb = bgr.reshape(-1, 3)[:, ::-1]
        return np.concatenate(
            [rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1,
        )

    # Pre-build the colorbar legend (constant across frames).
    cbar_w, cbar_h = 240, 20
    gradient = np.repeat(
        np.linspace(0, 255, cbar_w, dtype=np.uint8).reshape(1, -1), cbar_h, axis=0,
    )
    cbar_bgr = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    cbar_rgb = cbar_bgr[:, :, ::-1].copy()

    def draw_colorbar(frame: np.ndarray, x0: int, y0: int) -> None:
        """Paste cbar_rgb at (x0, y0) and add tick labels below it."""
        frame[y0:y0 + cbar_h, x0:x0 + cbar_w] = cbar_rgb
        cv2.rectangle(frame, (x0, y0), (x0 + cbar_w - 1, y0 + cbar_h - 1), (0, 0, 0), 1)
        for tick, label in (
            (0.0, "0"),
            (0.5, f"{max_dist_cm * 0.5:.1f}"),
            (1.0, f"{max_dist_cm:.1f} cm"),
        ):
            tx = x0 + int(tick * (cbar_w - 1))
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(
                frame, label, (tx - tw // 2, y0 + cbar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

    logger.info("Rendering chamfer heatmap video -> %s", video_path)
    writer = get_video_writer(video_path, fps=fps, crf=23)
    with Renderer(image_size=image_size) as renderer:
        for i in tqdm(range(N), desc="Rendering chamfer heatmap"):
            mesh_mhr = trimesh.Trimesh(
                vertices=verts_mhr_np[i], faces=mhr_faces, process=False,
            )
            mesh_mhr.visual.vertex_colors = colorize(mhr_dists_all[i])
            frame_mhr = renderer.render_overlay([mesh_mhr], K, T, image=bg)
            frame_mhr = (frame_mhr * 255.0).astype(np.uint8)

            mesh_soma = trimesh.Trimesh(
                vertices=verts_soma_np[i], faces=soma_faces, process=False,
            )
            mesh_soma.visual.vertex_colors = colorize(soma_dists_all[i])
            frame_soma = renderer.render_overlay([mesh_soma], K, T, image=bg)
            frame_soma = (frame_soma * 255.0).astype(np.uint8)

            cv2.putText(
                frame_mhr, "Chamfer Distance on MHR",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
            )
            cv2.putText(
                frame_mhr, f"(distance to SOMA, max {mhr_dists_all[i].max():.2f} cm)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
            )
            cv2.putText(
                frame_soma, "Chamfer Distance on SOMA",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
            )
            cv2.putText(
                frame_soma, f"(distance to MHR, max {soma_dists_all[i].max():.2f} cm)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
            )

            tiled = np.concatenate([frame_mhr, frame_soma], axis=1)
            draw_colorbar(tiled, x0=20, y0=tiled.shape[0] - cbar_h - 30)
            writer.write_frame(tiled)
    writer.close()
    logger.info("Saved: %s", video_path)


def export_soma(
    params_path: Path,
    output_path: Path,
    mesh_path: Path | None = None,
    weights_dir: Path | None = None,
    body_iters: int = 2,
    full_iters: int = 1,
    finger_iters: int = 0,
    autograd_iters: int = 0,
    autograd_lr: float = 5e-3,
    leaf_weight: float = 1.0,
    foot_weight: float | None = None,
    batch_size: int = 64,
    output_unit: str = "meters",
    device: str = "cuda",
    debug: int = 0,
) -> None:
    if not params_path.exists():
        raise FileNotFoundError(f"Required file not found: {params_path}")

    params_data = torch.load(params_path, map_location="cpu", weights_only=True)

    # --- Obtain MHR-native positioned vertices ---
    mhr_faces = None
    if mesh_path is not None and mesh_path.exists():
        verts, mhr_faces = load_vertices_from_mesh(mesh_path)
        # Path A: vertices are in meters (MHR head divides JIT output by 100).
        # SOMA operates in centimeters, so scale up.
        verts = verts * 100.0
    else:
        if mesh_path is not None:
            logger.warning("Mesh file not found: %s, falling back to forward pass", mesh_path)
        if weights_dir is None:
            raise ValueError(
                "--mesh_path not provided (or file missing) and --weights_dir not provided. "
                "Either provide the mesh file or the MHR JIT model weights."
            )
        # Path B: JIT model outputs in centimeters directly.
        verts = load_vertices_from_forward(params_data, weights_dir, device=device)

    N = verts.shape[0]
    logger.info("Converting %d frames to SOMA format (vertices in cm)", N)

    shape_params = params_data["shape_params"].float()  # (N, 45)
    model_params = params_data["mhr_model_params"].float()  # (N, 204)
    scale_params = model_params[:, 136:]  # (N, 68) resolved body-part scales

    # --- SOMA PoseInversion ---
    logger.info("Initializing SOMA layer (identity_model_type=mhr)")
    soma = SOMALayer(
        identity_model_type="mhr",
        device=device,
        mode="warp",
        output_unit=Unit.CENTIMETERS,
    )
    inv = PoseInversion(soma, low_lod=True)

    all_ic = shape_params.to(device)
    all_sp = scale_params.to(device)

    all_rotations = []
    all_root_transl = []
    all_errors = []

    if foot_weight is not None:
        leaf_weight_arg: dict | float = {
            "head": leaf_weight,
            "hands": leaf_weight,
            "feet": foot_weight,
        }
    else:
        leaf_weight_arg = leaf_weight

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        inv.prepare_identity(all_ic[start:end], all_sp[start:end])
        result = inv.fit(
            verts[start:end].to(device),
            body_iters=body_iters,
            finger_iters=finger_iters,
            full_iters=full_iters,
            autograd_iters=autograd_iters,
            autograd_lr=autograd_lr,
            leaf_weight=leaf_weight_arg,
            batch_size=None,
        )
        all_rotations.append(result["rotations"].cpu())
        all_root_transl.append(result["root_translation"].cpu())
        all_errors.append(result["per_vertex_error"].cpu())

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    rotations = torch.cat(all_rotations, dim=0)
    root_transl = torch.cat(all_root_transl, dim=0)
    err = torch.cat(all_errors, dim=0)

    unit_label = "cm" if soma.output_unit == Unit.CENTIMETERS else "m"
    logger.info("Inversion time: %.2fs (%.0f FPS)", dt, N / dt if dt > 0 else 0)
    logger.info("Mean vertex error: %.4f %s", err.mean().item(), unit_label)
    logger.info("Median vertex error: %.4f %s", err.median().item(), unit_label)
    logger.info("Max vertex error: %.4f %s", err.max().item(), unit_label)

    # --- Convert to T-pose-relative rotvec and save ---
    soma_device = soma._t_pose_orient.device
    rel_rotations = remove_joint_orient_local(
        rotations.to(soma_device),
        soma._t_pose_orient,
        soma._t_pose_orient_parent_T,
    )
    poses_rotvec = matrix_to_rotvec(
        rel_rotations.reshape(-1, 3, 3),
    ).reshape(rotations.shape[0], rotations.shape[1], 3).cpu()

    save_transl = root_transl.clone()
    target_unit = Unit.from_name(output_unit)
    unit_scale = soma.output_unit.meters_per_unit / target_unit.meters_per_unit
    if unit_scale != 1.0:
        save_transl = save_transl * unit_scale

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_soma_npz(
        str(output_path),
        poses_rotvec,
        save_transl,
        joint_names=list(soma.rig_data["joint_names"]),
        identity_model_type=soma.identity_model_type,
        identity_coeffs=shape_params.numpy(),
        scale_params=scale_params.numpy(),
        joint_orient=soma._t_pose_orient,
        unit=output_unit,
        keep_root=False,
    )

    # --- Debug: render MHR vs SOMA comparison video ---
    if debug > 0:
        soma_verts = _reconstruct_soma_vertices(
            soma, rotations, root_transl,
            shape_params, scale_params,
            batch_size=batch_size, device=device,
        )
        soma_faces = soma.faces.cpu().numpy()
        _render_debug(
            output_path, verts, mhr_faces, soma_verts, soma_faces,
        )
        _render_chamfer_heatmap(
            output_path, verts, mhr_faces, soma_verts, soma_faces,
            device=device,
        )



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MHR parameters to SOMA format",
    )
    parser.add_argument(
        "--params_path", type=str, required=True,
        help="Path to mhr_params_mv.pt",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output .npz file path",
    )
    parser.add_argument(
        "--mesh_path", type=str, default=None,
        help="Path to mhr_mesh_mv.pt (optional; enables Path A vertex loading)",
    )
    parser.add_argument(
        "--weights_dir", type=str, default=None,
        help="sam3d_body weights directory (contains MHR JIT model; "
             "fallback when --mesh_path is absent)",
    )
    parser.add_argument("--body_iters", type=int, default=2)
    parser.add_argument("--full_iters", type=int, default=1)
    parser.add_argument("--finger_iters", type=int, default=0)
    parser.add_argument("--autograd_iters", type=int, default=0)
    parser.add_argument("--autograd_lr", type=float, default=5e-3)
    parser.add_argument("--leaf_weight", type=float, default=1.0,
                        help="Uniform extremity vertex weight passed to PoseInversion.fit")
    parser.add_argument("--foot_weight", type=float, default=None,
                        help="Override foot vertex weight (default: same as --leaf_weight). "
                             "Pair with --autograd_iters > 0 for it to take effect.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--output_unit", type=str, default="meters",
        choices=["meters", "centimeters", "millimeters"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--debug", type=int, default=0,
        help="Debug level. >0 renders MHR vs SOMA comparison video",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    export_soma(
        params_path=Path(args.params_path),
        output_path=Path(args.output_path),
        mesh_path=Path(args.mesh_path) if args.mesh_path else None,
        weights_dir=Path(args.weights_dir) if args.weights_dir else None,
        body_iters=args.body_iters,
        full_iters=args.full_iters,
        finger_iters=args.finger_iters,
        autograd_iters=args.autograd_iters,
        autograd_lr=args.autograd_lr,
        leaf_weight=args.leaf_weight,
        foot_weight=args.foot_weight,
        batch_size=args.batch_size,
        output_unit=args.output_unit,
        device=args.device,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
