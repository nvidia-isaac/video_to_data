from __future__ import annotations

import math
from pathlib import Path
import time

import cv2
import imageio.v3 as iio
import numpy as np
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    euler_angles_to_matrix,
)
import torch
import trimesh
from tqdm import tqdm

from v2d.common.datatypes import Mask
from v2d.mv.math.numpy_fn import xyz_to_uv
from v2d.mv.rig import RigConfig
from v2d.common.video import FrameSource, get_video_writer
from v2d.mv.math.torch_fn import (
    geman_mcclure_distance,
    l2_distance,
    reproject,
    reproject_multiview,
    se3_inv,
)
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.models.heads import MHRHead
from sam_3d_body.models.modules import mhr_utils
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from .estimate_mhr_params import (
    estimate_mhr_params,
    mhr_estimation_cache_matches,
)
from v2d.mv.vis.renderer import Renderer
from .renderer_gpu import GPURenderer
from .visibility import (
    compute_keypoint_visibility_raycast,
    compute_keypoint_visibility_raycast_gpu,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEYPOINT_WEIGHTS = torch.ones(70, device=DEVICE)
KEYPOINT_WEIGHTS[0:5] = (70.0 / 5) / 5  # nose, eyes, ears
KEYPOINT_WEIGHTS[5:15] = (70.0 / 5) / 10  # shoulders, elbows, hips, knees, ankles
KEYPOINT_WEIGHTS[15:21] = (70.0 / 5) / 6  # big toes, small toes, heels
KEYPOINT_WEIGHTS[21:63] = (70.0 / 5) / 42  # thumbs, index, middle, ring, pinky
KEYPOINT_WEIGHTS[63:70] = (70.0 / 5) / 7  # wrist, olecranon, cubital fossa, acromion, neck


class MHRLayer(torch.nn.Module):
    """Wraps MHRHead with postprocessing (Y/Z flip, keypoint slicing)."""

    def __init__(self, mhr_head: MHRHead):
        super().__init__()
        self.mhr_head = mhr_head

    @property
    def faces(self) -> torch.Tensor:
        return self.mhr_head.faces

    def forward(self, mhr_inputs: dict, keypoints_only: bool = False) -> dict:
        batch_size = mhr_inputs["global_rot"].shape[0]

        if keypoints_only:
            verts, j3d = self.mhr_head.mhr_forward(**mhr_inputs, return_keypoints=True)
            j3d = j3d[:, :70]
            if verts is not None:
                verts[..., [1, 2]] *= -1
            j3d[..., [1, 2]] *= -1
            return {
                "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
                "pred_vertices": verts.reshape(batch_size, -1, 3) if verts is not None else None,
            }

        verts, j3d, jcoords, mhr_model_params, joint_global_rots = self.mhr_head.mhr_forward(
            **mhr_inputs,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )
        j3d = j3d[:, :70]
        if verts is not None:
            verts[..., [1, 2]] *= -1
        j3d[..., [1, 2]] *= -1
        if jcoords is not None:
            jcoords[..., [1, 2]] *= -1

        return {
            "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
            "pred_vertices": verts.reshape(batch_size, -1, 3) if verts is not None else None,
            "pred_joint_coords": jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None,
            "pred_global_rots": joint_global_rots,
            "mhr_model_params": mhr_model_params,
        }


def transform_mhr_params(mhr_params: dict, T_target_from_src: torch.Tensor):
    """Transform MHR params from source camera frame to target frame.

    Handles the MHR-native (RH Y-up) ↔ OpenCV (RH Y-down) flip via F = diag(1,-1,-1).
    Transforms both global_rot and global_trans.
    """
    R_w = T_target_from_src[:3, :3]
    t_w = T_target_from_src[:3, 3]
    F = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=R_w.device))
    R_w_conj = F @ R_w @ F
    R_c = euler_angles_to_matrix(mhr_params["global_rot"], "ZYX")
    mhr_params["global_rot"] = matrix_to_euler_angles(R_w_conj @ R_c, "ZYX")
    mhr_params["global_trans"] = (mhr_params["global_trans"] @ R_w_conj.T) + (F @ t_w)
    return mhr_params


def average_quaternions(quats: torch.Tensor) -> torch.Tensor:
    """Average quaternions across dim 0 with hemisphere alignment.

    Args:
        quats: (C, ..., 4) quaternions to average over the first dim.
    Returns:
        Averaged unit quaternion with shape (..., 4).
    """
    signs = torch.sign((quats * quats[:1]).sum(dim=-1, keepdim=True))
    quats = quats * signs
    avg = quats.mean(dim=0)
    return avg / avg.norm(dim=-1, keepdim=True)


def average_euler_angles(euler_angles: torch.Tensor) -> torch.Tensor:
    """Average ZYX Euler angles across dim 0 via 6D rotation representation.

    Args:
        euler_angles: (C, ..., 3) ZYX Euler angles to average over the first dim.
    Returns:
        Averaged Euler angles with shape (..., 3).
    """
    rotmats = euler_angles_to_matrix(euler_angles, "ZYX")
    rot6d = matrix_to_rotation_6d(rotmats)
    avg_6d = rot6d.mean(dim=0)
    return matrix_to_euler_angles(rotation_6d_to_matrix(avg_6d), "ZYX")


def extract_mhr_inputs(mhr_outputs: dict):
    """Extract the MHR parameters that are targets for optimization, i.e. input to the MHR head.

    global_trans is derived from pred_cam_t by undoing the Y/Z flip applied by mhr_head.
    The MHR head expects global_trans in meters (MHR-native coords); it internally applies *10.
    """
    global_trans = mhr_outputs["pred_cam_t"].clone()
    global_trans[..., [1, 2]] *= -1  # undo Y/Z flip -> MHR-native
    return {
        "global_trans": global_trans,
        "global_rot": mhr_outputs["global_rot"],
        "body_pose_params": mhr_outputs["body_pose_params"],
        "hand_pose_params": mhr_outputs["hand_pose_params"],
        "scale_params": mhr_outputs["scale_params"],
        "shape_params": mhr_outputs["shape_params"],
    }


def average_mhr_inputs(mhr_inputs_all: list[dict]) -> dict:
    """Average MHR head inputs across cameras, using continuous representations for rotations.

    Args:
        mhr_inputs_all: List of C dicts, each with tensors of shape (N, ...).
    Returns:
        Dict with averaged tensors of shape (N, ...).
    """
    def _stack(key):
        return torch.stack([inp[key] for inp in mhr_inputs_all])  # (C, N, ...)

    # Global rotation: average in 6D rotation space to avoid Euler discontinuities
    global_rot = average_euler_angles(_stack("global_rot"))  # (N, 3)

    # Body pose: convert to continuous repr, average, convert back
    body_params_stacked = _stack("body_pose_params")  # (C, N, 133)
    body_cont = mhr_utils.compact_model_params_to_cont_body(body_params_stacked)  # (C, N, 260)
    body_pose_params = mhr_utils.compact_cont_to_model_params_body(body_cont.mean(dim=0))  # (N, 133)

    # Hand pose: already in continuous space (sin/cos pairs), safe to average directly
    hand_pose_params = _stack("hand_pose_params").mean(dim=0)

    # Linear params: simple mean
    global_trans = _stack("global_trans").mean(dim=0)
    scale_params = _stack("scale_params").mean(dim=0)
    shape_params = _stack("shape_params").mean(dim=0)

    return {
        "global_trans": global_trans,
        "global_rot": global_rot,
        "body_pose_params": body_pose_params,
        "hand_pose_params": hand_pose_params,
        "scale_params": scale_params,
        "shape_params": shape_params,
    }


def export_mhr_outputs(
    mhr_layer: MHRLayer,
    mhr_inputs: dict,
) -> tuple[dict, dict]:
    """Run full MHRLayer forward and package inputs + outputs for saving.

    Derives pred_cam_t from global_trans (Y/Z flip) for backward compatibility.

    Args:
        mhr_layer: The MHR layer.
        mhr_inputs: The MHR inputs (including global_trans in MHR-native world coords).
    Returns:
        (mhr_params dict, mhr_mesh dict).
    """
    processed = mhr_layer(mhr_inputs)

    pred_cam_t = mhr_inputs["global_trans"].clone()
    pred_cam_t[..., [1, 2]] *= -1  # MHR-native -> Y/Z-flipped (camera convention)

    mhr_params = {
        "global_rot": mhr_inputs["global_rot"],
        "body_pose_params": mhr_inputs["body_pose_params"],
        "hand_pose_params": mhr_inputs["hand_pose_params"],
        "scale_params": mhr_inputs["scale_params"],
        "shape_params": mhr_inputs["shape_params"],
        "pred_cam_t": pred_cam_t,
        "pred_keypoints_3d": processed["pred_keypoints_3d"],
        "pred_joint_coords": processed["pred_joint_coords"],
        "pred_global_rots": processed["pred_global_rots"],
        "mhr_model_params": processed["mhr_model_params"],
    }

    mhr_mesh = {
        "faces": mhr_layer.faces,
        "pred_vertices": processed["pred_vertices"],
    }

    return mhr_params, mhr_mesh


def mhr_inputs_to_opt_params(mhr_inputs: dict) -> dict:
    """Convert MHR inputs to optimization-friendly parameters.

    Transforms rotation/pose parameters into continuous representations and
    collapses per-person-constant parameters (shape, scale) to a single row.

    Args:
        mhr_inputs: Dict from extract_mhr_inputs with (N, ...) tensors.
    Returns:
        Dict of optimization parameters.
    """
    return {
        "global_trans": mhr_inputs["global_trans"],                             # (N, 3)
        "global_rot_6d": matrix_to_rotation_6d(
            euler_angles_to_matrix(mhr_inputs["global_rot"], "ZYX")
        ),                                                                      # (N, 6)
        "body_cont": mhr_utils.compact_model_params_to_cont_body(
            mhr_inputs["body_pose_params"]
        ),                                                                      # (N, 260)
        "hand_pose_params": mhr_inputs["hand_pose_params"],                     # (N, 108)
        "scale_params": mhr_inputs["scale_params"].mean(dim=0, keepdim=True),   # (1, 28)
        "shape_params": mhr_inputs["shape_params"].mean(dim=0, keepdim=True),   # (1, 45)
    }


def opt_params_to_mhr_inputs(opt_params: dict) -> dict:
    """Convert optimization parameters back to the dict expected by mhr_head.mhr_forward.

    Args:
        opt_params: Dict from mhr_inputs_to_opt_params.
    Returns:
        Dict compatible with mhr_head.mhr_forward(**result).
    """
    batch_size = opt_params["global_rot_6d"].shape[0]
    return {
        "global_trans": opt_params["global_trans"],                              # (N, 3)
        "global_rot": matrix_to_euler_angles(
            rotation_6d_to_matrix(opt_params["global_rot_6d"]), "ZYX"
        ),                                                                       # (N, 3)
        "body_pose_params": mhr_utils.compact_cont_to_model_params_body(
            opt_params["body_cont"]
        ),                                                                       # (N, 133)
        "hand_pose_params": opt_params["hand_pose_params"],                      # (N, 108)
        "scale_params": opt_params["scale_params"].expand(batch_size, -1),       # (N, 28)
        "shape_params": opt_params["shape_params"].expand(batch_size, -1),       # (N, 45)
    }


def reprojection_error(
    gt_keypoints_2d: torch.Tensor,
    gt_weights: torch.Tensor,
    pred_keypoints_3d: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    cam_extrinsics: torch.Tensor,
    keypoint_weights: torch.Tensor = KEYPOINT_WEIGHTS,
    gm_scale: float = 50,
):
    """Weighted reprojection error across multiple camera views.

    Projects pred_keypoints_3d into each camera and computes the weighted mean
    L2 distance to the ground-truth 2D detections.  Points behind the camera
    or with non-finite projections are zeroed via depth_mask / reproj_mask.
    If no valid points remain, returns zero (with grad).

    Args:
        gt_keypoints_2d: (C, N, P, 2) ground-truth 2D keypoints per camera,
            where C = cameras, N = batch (frames), P = keypoints.
        gt_weights: (C, N, P) float weights for ground truth keypoints per camera.
            0.0 = fully excluded, 1.0 = fully included.
        pred_keypoints_3d: (N, K, 3) predicted 3D keypoints in world frame.
        cam_intrinsics: (C, 3, 3) camera intrinsic matrices.
        cam_extrinsics: (C, 4, 4) world-to-camera extrinsic matrices.

    Returns:
        Scalar weighted mean reprojection error over all keypoints.
    """
    pred_keypoints_2d, depth_mask = reproject_multiview(
        pred_keypoints_3d,
        cam_intrinsics,
        se3_inv(cam_extrinsics),
    )  # (C, N, P, 2), (C, N, P)

    reproj_mask = torch.isfinite(pred_keypoints_2d).all(dim=-1, keepdim=True)  # (C, N, P, 1)
    pred_keypoints_2d_clean = torch.where(reproj_mask, pred_keypoints_2d, gt_keypoints_2d + 1000.0)

    weights = gt_weights * depth_mask.float() * reproj_mask.squeeze(-1).float()
    total_weight = weights.sum()
    if total_weight == 0:
        return torch.tensor(0.0, device=pred_keypoints_2d.device, requires_grad=True)

    error = geman_mcclure_distance(pred_keypoints_2d_clean, gt_keypoints_2d, gm_scale)  # (C, N, P)
    error = error * keypoint_weights.to(error.device)[None, None, :]
    error = error * weights
    return error.sum() / total_weight


def temporal_smoothness(opt_params: dict):
    """L2 penalty on frame-to-frame parameter changes (no mhr_forward needed)."""
    loss = 0.0
    for key in ["global_trans", "global_rot_6d", "body_cont", "hand_pose_params"]:
        loss += l2_distance(opt_params[key][1:], opt_params[key][:-1]).mean()
    return loss


def optimize_multiview(
    mhr_layer: MHRLayer,
    gt_keypoints_2d: torch.Tensor,
    gt_weights: torch.Tensor,
    mhr_inputs: dict,
    cam_intrinsics: torch.Tensor,
    cam_extrinsics: torch.Tensor,
    reproj_weight: float = 1.0,
    temporal_weight: float = 2.0,
    max_iterations: int = 200,
    lr: float = 3e-3,
    chunk_size: int = 64,
):
    """Optimize the MHR inputs to match the ground truth keypoints.

    global_trans is optimized as part of opt_params (no separate pred_cam_t).
    MHR forward output already includes translation in vertices/keypoints.

    Args:
        mhr_layer: The MHR layer.
        gt_keypoints_2d: (C, N, P, 2) ground truth 2D keypoints per camera.
        gt_weights: (C, N, P) float weights for ground truth keypoints per camera.
        mhr_inputs: The MHR inputs (including global_trans).
        cam_intrinsics: (C, 3, 3) tensor of camera intrinsic matrices.
        cam_extrinsics: (C, 4, 4) tensor of camera extrinsic matrices.
        max_iterations: The maximum number of iterations.
        lr: The learning rate.
        chunk_size: Number of frames per forward pass to avoid OOM.
    Returns:
        The optimized MHR inputs.
    """
    cam_intrinsics = cam_intrinsics.to(mhr_inputs["global_rot"].device)
    cam_extrinsics = cam_extrinsics.to(mhr_inputs["global_rot"].device)

    opt_params = mhr_inputs_to_opt_params(mhr_inputs)
    for p in opt_params.values():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(list(opt_params.values()), lr=lr)

    n_frames = opt_params["global_rot_6d"].shape[0]
    n_chunks = math.ceil(n_frames / chunk_size)
    nan_grad_count = 0

    for i in range(max_iterations):
        optimizer.zero_grad()

        total_reproj = 0.0
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)

            chunk_opt = {
                k: v[start:end] if v.shape[0] > 1 else v
                for k, v in opt_params.items()
            }
            chunk_mhr = opt_params_to_mhr_inputs(chunk_opt)

            chunk_out = mhr_layer(chunk_mhr, keypoints_only=True)

            chunk_reproj = reprojection_error(
                gt_keypoints_2d[:, start:end],
                gt_weights[:, start:end],
                chunk_out["pred_keypoints_3d"],
                cam_intrinsics,
                cam_extrinsics,
            )
            (reproj_weight * chunk_reproj / n_chunks).backward()
            total_reproj += chunk_reproj.item()

        temp_loss = temporal_smoothness(opt_params)
        (temporal_weight * temp_loss).backward()

        grad_norms = {}
        for k, p in opt_params.items():
            if p.grad is not None:
                grad_norms[k] = p.grad.norm().item()
                if not torch.isfinite(p.grad).all():
                    nan_grad_count += 1
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        if i % 10 == 0:
            avg_reproj = total_reproj / n_chunks
            grad_str = " ".join(f"{k}={v:.2e}" for k, v in grad_norms.items())
            print(f"Iteration {i}: reproj: {avg_reproj:.4f}, temporal: {temp_loss.item():.4f} | grad_norms: {grad_str}")

        optimizer.step()

    print(f"Sanitized non-finite grads on {nan_grad_count} param-iterations")

    for p in opt_params.values():
        p.requires_grad_(False)

    return opt_params_to_mhr_inputs(opt_params)


def render_mhr_mesh(
    source: FrameSource,
    output_path: Path,
    pred_vertices: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    cam_extrinsics: torch.Tensor,
    faces: torch.Tensor,
):
    """Render MHR outputs in the camera frame.
    Assumes that the mhr_outputs are in the world frame (translation included).
    Args:
        source: The frame source.
        output_path: The output path.
        pred_vertices: (N, V, 3) predicted vertices in world frame (includes translation).
        cam_intrinsics: (3, 3) camera intrinsic matrix.
        cam_extrinsics: (4, 4) camera extrinsic matrix.
        faces: (F, 3) face indices.
    """

    K = cam_intrinsics.cpu().numpy() if isinstance(cam_intrinsics, torch.Tensor) else cam_intrinsics
    T = cam_extrinsics.cpu().numpy() if isinstance(cam_extrinsics, torch.Tensor) else cam_extrinsics
    faces_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces

    writer = get_video_writer(output_path, fps=30, crf=23)
    with Renderer(image_size=source.image_size) as renderer:
        for i, image in enumerate(tqdm(source.iter_frames(), total=source.n_frames, desc="Rendering MHR outputs")):
            verts_i = pred_vertices[i].cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts_i, faces=faces_np, process=False)
            mesh.visual.vertex_colors = np.full((len(verts_i), 4), [102, 230, 179, 255], dtype=np.uint8)
            frame = renderer.render_overlay([mesh], K, T, image=image)
            frame = (frame * 255.0).astype(np.uint8)
            label = f"Frame {i}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(frame, label, (frame.shape[1] - tw - 10, th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write_frame(frame)
    writer.close()


def render_mhr_mesh_gpu(
    source: FrameSource,
    output_path: Path,
    pred_vertices: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    cam_extrinsics: torch.Tensor,
    faces: torch.Tensor,
    batch_size: int = 32,
):
    """GPU-batched version of render_mhr_mesh using PyTorch3D.

    Args:
        source: The frame source.
        output_path: The output path.
        pred_vertices: (N, V, 3) predicted vertices in world frame (includes translation).
        cam_intrinsics: (3, 3) camera intrinsic matrix.
        cam_extrinsics: (4, 4) camera extrinsic matrix.
        faces: (F, 3) face indices.
        batch_size: Number of frames to render in one GPU call.
    """

    renderer = GPURenderer(image_size=source.image_size, device=DEVICE)
    K = torch.as_tensor(cam_intrinsics, dtype=torch.float32, device=DEVICE)
    T = torch.as_tensor(cam_extrinsics, dtype=torch.float32, device=DEVICE)

    writer = get_video_writer(output_path, fps=30, crf=23)
    batch_images: list[np.ndarray] = []
    batch_start = 0
    for i, image in enumerate(tqdm(source.iter_frames(), total=source.n_frames, desc="Rendering MHR outputs (GPU)")):
        batch_images.append(image)
        if len(batch_images) == batch_size or i == source.n_frames - 1:
            images_t = torch.from_numpy(np.stack(batch_images)).to(DEVICE)
            verts_batch = pred_vertices[batch_start:batch_start + len(batch_images)]
            rendered = renderer.render_overlay(verts=verts_batch, faces=faces, K=K, T=T, images=images_t)
            rendered_np = (rendered * 255.0).cpu().numpy().astype(np.uint8)
            for j in range(len(batch_images)):
                frame = rendered_np[j]
                label = f"Frame {batch_start + j}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(frame, label, (frame.shape[1] - tw - 10, th + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                writer.write_frame(frame)
            batch_images = []
            batch_start = i + 1
    writer.close()


def render_keypoints(
    source: FrameSource,
    output_path: Path,
    gt_keypoints_2d: torch.Tensor,
    pred_keypoints_3d: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    cam_extrinsics: torch.Tensor,
    gt_weights: torch.Tensor | None = None,
):
    """Render the keypoints 2D in the camera frame.

    Args:
        source: The frame source.
        output_path: The output path.
        gt_keypoints_2d: (N, P, 2) ground-truth 2D keypoints.
        pred_keypoints_3d: (N, P, 3) predicted 3D keypoints in world frame (includes translation).
        cam_intrinsics: (3, 3) camera intrinsic matrix.
        cam_extrinsics: (4, 4) camera extrinsic matrix.
        gt_weights: (N, P) float in [0, 1] visibility weights for GT keypoints.
            If provided, GT circles are colored from black (0) to green (1).
    """
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()

    pred_keypoints_2d, _ = reproject(
        pred_keypoints_3d.reshape(-1, 3),
        cam_intrinsics.to(pred_keypoints_3d.device),
        se3_inv(cam_extrinsics.to(pred_keypoints_3d.device)),
    )
    pred_keypoints_2d = pred_keypoints_2d.cpu().numpy().reshape(pred_keypoints_3d.shape[0], -1, 2)

    writer = get_video_writer(output_path, fps=30, crf=23)
    for i, image in enumerate(tqdm(source.iter_frames(), total=source.n_frames, desc="Rendering keypoints 2D")):
        for j in range(gt_keypoints_2d.shape[1]):
            if gt_weights is not None:
                w = float(gt_weights[i, j])
                color = (0, int(255 * w), 0)
            else:
                color = (0, 255, 0)
            cv2.circle(image, tuple(gt_keypoints_2d[i, j].astype(int)), 2, color, -1)
        for j in range(pred_keypoints_2d.shape[1]):
            cv2.circle(image, tuple(pred_keypoints_2d[i, j].astype(int)), 2, (255, 0, 0), -1)
        label = f"Frame {i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(image, label, (image.shape[1] - tw - 10, th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "Green = GT", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "Red = Pred", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        writer.write_frame(image.astype(np.uint8))
    writer.close()


def mv_optimize_mhr_params(
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    weights_dir: Path,
    rgb_paths: list[Path],
    bbox_paths: list[Path | None] | None,
    mhr_params_paths: list[Path],
    mhr_mesh_paths: list[Path],
    mhr_params_mv_path: Path,
    mhr_mesh_mv_path: Path | None = None,
    mask_dirs: list[Path] | None = None,
    keypoint_invisible_weight: float = 0.3,
    keypoint_occluded_weight: float = 0.3,
    sam3d_body_batch_size: int = 1,
    debug: int = 0,
):
    if bbox_paths is None and mask_dirs is None:
        raise ValueError("Either bbox_paths or mask_dirs is required")
    if bbox_paths is None:
        bbox_paths = [None] * len(rgb_paths)
    if len(bbox_paths) != len(rgb_paths):
        raise ValueError(f"bbox_paths length {len(bbox_paths)} != rgb_paths length {len(rgb_paths)}")
    if mask_dirs is not None and len(mask_dirs) != len(rgb_paths):
        raise ValueError(f"mask_dirs length {len(mask_dirs)} != rgb_paths length {len(rgb_paths)}")

    frame_sources = [FrameSource.from_path(p) for p in rgb_paths]

    body_model_path = weights_dir / "sam-3d-body-dinov3/model.ckpt"
    mhr_path = weights_dir / "sam-3d-body-dinov3/assets/mhr_model.pt"
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(body_model_path),
        device=DEVICE,
        mhr_path=str(mhr_path),
    )
    mhr_layer = MHRLayer(model.head_pose)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
    )
    cam_intrinsics_all = torch.from_numpy(np.stack(cam_intrinsics)).to(DEVICE)
    cam_extrinsics_all = torch.from_numpy(np.stack(cam_extrinsics)).to(DEVICE)

    # --- Pass 1: Estimation & data collection ---
    mhr_inputs_all: list[dict] = []
    gt_keypoints_2d_all: list[torch.Tensor] = []
    image_size_all: list[tuple[int, int]] = []
    cam_verts_list: list[torch.Tensor] = []
    cam_kp3d_list: list[torch.Tensor] = []
    n_frames = -1
    for cam_idx, (K, T_world_from_cam, frame_source, bbox_path, params_path, mesh_path) in enumerate(zip(
        cam_intrinsics_all, cam_extrinsics_all, frame_sources, bbox_paths, mhr_params_paths, mhr_mesh_paths,
    )):
        params_path = Path(params_path)
        mesh_path = Path(mesh_path)
        mask_path = mask_dirs[cam_idx] if mask_dirs is not None else None
        cache_valid = mhr_estimation_cache_matches(
            params_path=params_path,
            rgb_path=frame_source.path,
            bbox_path=bbox_path,
            mask_path=mask_path,
            n_frames=frame_source.n_frames,
        )
        if cache_valid:
            print(f"Loading cached MHR params for camera {cam_idx}: {params_path}")
            mhr_outputs = torch.load(params_path)
        else:
            if params_path.exists():
                print(f"Ignoring stale MHR params cache for camera {cam_idx}: {params_path}")
            print(f"Running SAM3D body estimation for camera {cam_idx}")
            mhr_outputs = estimate_mhr_params(
                rgb_path=frame_source.path,
                bbox_path=bbox_path,
                mask_path=mask_path,
                cam_intrinsics=K.cpu().numpy(),
                output_params_path=params_path,
                output_mesh_path=mesh_path,
                estimator=estimator,
                batch_size=sam3d_body_batch_size,
                debug=debug,
            )

        if mesh_path.exists():
            mhr_mesh_cam = torch.load(mesh_path)
        else:
            print(f"Mesh not cached for camera {cam_idx}, generating from mhr_outputs")
            mhr_inputs_cam = extract_mhr_inputs(mhr_outputs)
            with torch.no_grad():
                cam_out = mhr_layer(mhr_inputs_cam, keypoints_only=True)
            mhr_mesh_cam = {
                "faces": mhr_layer.faces,
                "pred_vertices": cam_out["pred_vertices"],
            }
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mhr_mesh_cam, mesh_path)

        mhr_inputs = extract_mhr_inputs(mhr_outputs)
        mhr_inputs = transform_mhr_params(mhr_inputs, T_world_from_cam)
        mhr_inputs_all.append(mhr_inputs)

        gt_keypoints_2d_all.append(mhr_outputs["pred_keypoints_2d"])

        cam_verts_list.append(mhr_mesh_cam["pred_vertices"])
        cam_kp3d_list.append(mhr_outputs["pred_keypoints_3d"])

        image_size_all.append(frame_source.image_size)
        if n_frames == -1:
            n_frames = frame_source.n_frames
        elif n_frames != frame_source.n_frames:
            raise ValueError(f"Number of frames mismatch for camera {cam_idx}: {n_frames} != {frame_source.n_frames}")

    # Free the SAM3D model to reclaim GPU memory for visibility computation.
    # mhr_layer survives via its own reference to model.head_pose.
    del estimator, model
    torch.cuda.empty_cache()

    # --- Pass 2: Keypoint weights by raycasting visibility and optional occlusion mask ---
    gt_weights_all: list[torch.Tensor] = []
    for i in range(len(cam_intrinsics)):
        raycast_vis = compute_keypoint_visibility_raycast_gpu(
            pred_keypoints_3d=cam_kp3d_list[i],
            pred_vertices=cam_verts_list[i],
            faces=mhr_layer.faces,
            K=cam_intrinsics_all[i],
            T=torch.eye(4, device=DEVICE),
            image_size=image_size_all[i],
        )
        w_inv = keypoint_invisible_weight
        gt_weights = w_inv + (1.0 - w_inv) * raycast_vis

        if mask_dirs is not None:
            mask_source = FrameSource.from_path(mask_dirs[i])
            K_np = cam_intrinsics_all[i].cpu().numpy()
            kps_np = cam_kp3d_list[i].cpu().numpy()  # (N, P, 3) camera frame

            N, P = kps_np.shape[:2]
            mask_vis = np.ones((N, P), dtype=np.float32)
            n_masks = min(N, mask_source.n_frames)
            for n in range(n_masks):
                mask_arr = mask_source[n].astype(np.float32) / 255.0
                H_m, W_m = mask_arr.shape[:2]
                uv_int, in_bounds = xyz_to_uv(kps_np[n], K_np, image_size=(W_m, H_m))
                valid = np.where(in_bounds)[0]
                mask_vis[n] = 0.0
                mask_vis[n, valid] = (mask_arr[uv_int[valid, 1], uv_int[valid, 0]] > 0.5).astype(np.float32)

            w_occ = keypoint_occluded_weight
            gt_weights = gt_weights * (w_occ + (1.0 - w_occ) * torch.from_numpy(mask_vis).to(DEVICE))

        gt_weights_all.append(gt_weights)

    mhr_params_mv_path.parent.mkdir(parents=True, exist_ok=True)

    mhr_inputs_avg = average_mhr_inputs(mhr_inputs_all)

    if debug > 0:
        mhr_outputs_avg = mhr_layer(mhr_inputs_avg, keypoints_only=True)
        for i in range(len(cam_intrinsics)):
            render_keypoints(
                source=frame_sources[i],
                output_path=mhr_params_mv_path.parent / f"mhr_keypoints_avg_{i}.mp4",
                gt_keypoints_2d=gt_keypoints_2d_all[i],
                pred_keypoints_3d=mhr_outputs_avg["pred_keypoints_3d"],
                cam_intrinsics=cam_intrinsics_all[i],
                cam_extrinsics=cam_extrinsics_all[i],
                gt_weights=gt_weights_all[i],
            )
            if debug <= 1:
                break

    gt_keypoints_2d_all = torch.stack(gt_keypoints_2d_all)
    gt_weights_all = torch.stack(gt_weights_all)  # (C, N, P)

    mhr_inputs_opt = optimize_multiview(
        mhr_layer=mhr_layer,
        gt_keypoints_2d=gt_keypoints_2d_all,
        gt_weights=gt_weights_all,
        mhr_inputs=mhr_inputs_avg,
        cam_intrinsics=cam_intrinsics_all,
        cam_extrinsics=cam_extrinsics_all,
    )

    mhr_params_opt, mhr_mesh_opt = export_mhr_outputs(
        mhr_layer=mhr_layer,
        mhr_inputs=mhr_inputs_opt,
    )
    torch.save(mhr_params_opt, mhr_params_mv_path)
    if mhr_mesh_mv_path is not None:
        torch.save(mhr_mesh_opt, mhr_mesh_mv_path)

    if debug > 0:
        for i in range(len(cam_intrinsics)):
            render_keypoints(
                source=frame_sources[i],
                output_path=mhr_params_mv_path.parent / f"mhr_keypoints_opt_{i}.mp4",
                gt_keypoints_2d=gt_keypoints_2d_all[i],
                pred_keypoints_3d=mhr_params_opt["pred_keypoints_3d"],
                cam_intrinsics=cam_intrinsics_all[i],
                cam_extrinsics=cam_extrinsics_all[i],
                gt_weights=gt_weights_all[i],
            )
        for i in range(len(cam_intrinsics)):
            render_mhr_mesh(
                source=frame_sources[i],
                output_path=mhr_params_mv_path.parent / f"mhr_mesh_opt_{i}.mp4",
                pred_vertices=mhr_mesh_opt["pred_vertices"],
                cam_intrinsics=cam_intrinsics_all[i],
                cam_extrinsics=cam_extrinsics_all[i],
                faces=mhr_layer.faces,
            )
            if debug <= 1:
                break

    return mhr_params_opt


def mv_optimize_mhr_params_from_config(cfg):
    """Wrapper that resolves config fields into explicit arguments for mv_optimize_mhr_params."""
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)

    cam_intrinsics: list[np.ndarray] = []
    cam_extrinsics: list[np.ndarray] = []
    rgb_paths: list[Path] = []
    bbox_paths: list[Path | None] | None = None
    mhr_params_paths: list[Path] = []
    mhr_mesh_paths: list[Path] = []
    mask_dirs: list[Path] | None = None

    if cfg.get("bbox_dir", None) is not None:
        bbox_paths = []
    if cfg.get("mask_dir", None) is not None:
        mask_dirs = []
    if bbox_paths is None and mask_dirs is None:
        raise ValueError("Config must provide at least one of bbox_dir or mask_dir")

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        cam_intrinsics.append(cam.param.K)
        cam_extrinsics.append(cam.param.T)

        rgb_paths.append(
            Path(cfg.rgb_path_template.format(cam_name=cam.name))
        )
        if bbox_paths is not None:
            bbox_paths.append(
                Path(cfg.bbox_path_template.format(cam_name=cam.name))
            )
        mhr_params_paths.append(
            Path(cfg.mhr_params_path_template.format(cam_name=cam.name))
        )
        mhr_mesh_paths.append(
            Path(cfg.mhr_mesh_path_template.format(cam_name=cam.name))
        )
        if mask_dirs is not None:
            mask_dirs.append(
                Path(cfg.mask_path_template.format(cam_name=cam.name))
            )

    weights_dir = Path(cfg.weights_dir)
    mhr_params_mv_path = Path(cfg.mhr_params_mv_path)
    mhr_mesh_mv_path = Path(cfg.mhr_mesh_mv_path)

    return mv_optimize_mhr_params(
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        weights_dir=weights_dir,
        rgb_paths=rgb_paths,
        bbox_paths=bbox_paths,
        mhr_params_paths=mhr_params_paths,
        mhr_mesh_paths=mhr_mesh_paths,
        mhr_params_mv_path=mhr_params_mv_path,
        mhr_mesh_mv_path=mhr_mesh_mv_path,
        mask_dirs=mask_dirs,
        keypoint_invisible_weight=cfg.keypoint_invisible_weight,
        keypoint_occluded_weight=cfg.keypoint_occluded_weight,
        sam3d_body_batch_size=cfg.sam3d_body_batch_size,
        debug=cfg.debug,
    )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Multi-view MHR parameter optimization")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing input frames")
    parser.add_argument("--camera_params_path", type=str, required=True, help="Path to camera parameters")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing model weights")
    parser.add_argument("--bbox_dir", type=str, default=None, help="Directory containing bounding boxes")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing SAM2 masks (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    parser.add_argument("--debug", type=int, default=None, help="Debug level override")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_optimize_mhr_params.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides = {
        "rgb_dir": args.rgb_dir,
        "output_dir": args.output_dir,
        "camera_params_path": args.camera_params_path,
        "weights_dir": args.weights_dir,
    }
    if args.bbox_dir is not None:
        overrides["bbox_dir"] = args.bbox_dir
    if args.mask_dir is not None:
        overrides["mask_dir"] = args.mask_dir
    if args.debug is not None:
        overrides["debug"] = args.debug

    cfg = OmegaConf.merge(cfg, overrides)
    mv_optimize_mhr_params_from_config(cfg)
