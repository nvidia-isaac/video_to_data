"""Extrinsic calibration solvers: PnP initialization and Ceres bundle adjustment."""

import copy
import logging
from functools import partial
from typing import Any

import cv2
import numpy as np
import pyceres
from pytransform3d.transform_manager import TransformManager
import torch

from v2d.mv.rig import CameraParam
from v2d.mv.rig.edex import DistortionModel
from v2d.mv.math.numpy_fn import (
    se3_from_rot_trans,
    se3_inliers_trans,
    se3_split_mean,
)
from v2d.mv.math.torch_fn import (
    se3_exp_map,
    se3_inv as se3_inv_torch,
    se3_log_map,
    reproject as reproject_torch,
    distort_polynomial as distort_polynomial_torch,
)


logger = logging.getLogger(__name__)


def extrinsics_estimate_pnp_pairwise(
    correspondences: list[list[np.ndarray | None]],
    target_xyz: np.ndarray,
    param1: CameraParam,
    param2: CameraParam,
    cam_id1: int,
    cam_id2: int,
    pnp_flags: int = cv2.SOLVEPNP_IPPE,
):
    """Estimate relative pose between two cameras using PnP on chessboard correspondences.

    Args:
        correspondences: List of frames, each a list of per-camera observations (P, 2) or None.
        target_xyz: (P, 3) target 3D points.
        param1, param2: Camera parameters.
        cam_id1, cam_id2: Camera indices into the correspondences.
        pnp_flags: OpenCV PnP algorithm flag.

    Returns:
        rel_pose: (4, 4) relative pose from camera 2 to camera 1.
        target_poses: List of (frame_idx, (4, 4)) target poses in camera 1 frame.
    """
    K1, D1 = param1.K, param1.D
    K2, D2 = param2.K, param2.D

    rel_poses = []
    target_poses = []
    for frame_idx, frame in enumerate(correspondences):
        obs_uv1 = frame[cam_id1]
        obs_uv2 = frame[cam_id2]
        if obs_uv1 is None:
            continue

        ret1, rvec1, tvec1 = cv2.solvePnP(target_xyz, obs_uv1, K1, D1, flags=pnp_flags)
        if ret1:
            R_cam1_target, _ = cv2.Rodrigues(rvec1)
            target_poses.append((frame_idx, se3_from_rot_trans(R_cam1_target, tvec1.flatten())))

        if obs_uv2 is not None:
            assert len(obs_uv1) == len(obs_uv2) == len(target_xyz)
            ret2, rvec2, tvec2 = cv2.solvePnP(target_xyz, obs_uv2, K2, D2, flags=pnp_flags)
            if ret1 and ret2:
                R_cam2_target, _ = cv2.Rodrigues(rvec2)
                R = R_cam1_target @ R_cam2_target.T
                t = tvec1.flatten() - R @ tvec2.flatten()
                rel_poses.append(se3_from_rot_trans(R, t))

    if len(rel_poses) == 0:
        raise ValueError(
            f"No valid PnP pairs found between cameras {cam_id1} and {cam_id2}. "
            "Check that chessboard is visible in both cameras simultaneously."
        )
    rel_poses = np.stack(rel_poses, axis=0)
    rel_poses = se3_inliers_trans(rel_poses)
    return se3_split_mean(rel_poses), target_poses


def extrinsics_estimate_pnp(
    correspondences: list[list[np.ndarray | None]],
    target_xyz: np.ndarray,
    camera_params: list[CameraParam],
    calibration_order: list[int],
    stereo_pairs: list[tuple[int, int]],
    pnp_flags: int = cv2.SOLVEPNP_IPPE,
):
    """Estimate all camera extrinsics using chained PnP with stereo baseline constraints.

    Args:
        correspondences: List of frames, each a list of per-camera observations.
        target_xyz: (P, 3) target 3D points.
        camera_params: List of camera parameters (one per camera).
        calibration_order: List of left camera IDs defining the pairwise chain.
        stereo_pairs: List of (left_cam_id, right_cam_id) tuples.
        pnp_flags: OpenCV PnP algorithm flag.

    Returns:
        camera_params: Updated camera parameters with T field set.
        target_poses: (N, 4, 4) array of target poses.
    """
    logger.info(
        f"Estimating extrinsics using PnP"
        f"\n\t- Number of frames: {len(correspondences)}"
        f"\n\t- Number of cameras: {len(camera_params)}"
        f"\n\t- Calibration order: {calibration_order}"
    )

    tm = TransformManager()
    for i in range(len(calibration_order) - 1):
        cam_id1 = calibration_order[i]
        cam_id2 = calibration_order[i + 1]
        rel_pose, target_poses_pair = extrinsics_estimate_pnp_pairwise(
            target_xyz=target_xyz,
            correspondences=correspondences,
            param1=camera_params[cam_id1],
            param2=camera_params[cam_id2],
            cam_id1=cam_id1,
            cam_id2=cam_id2,
            pnp_flags=pnp_flags,
        )
        tm.add_transform(cam_id2, cam_id1, rel_pose)
        for frame_idx, target_pose in target_poses_pair:
            tm.add_transform(f"target_{frame_idx}", cam_id1, target_pose)

    # Add known stereo baselines
    left_cam_ids = [pair[0] for pair in stereo_pairs]
    for cam_id in left_cam_ids:
        right_cam_id = cam_id + 1
        if camera_params[cam_id].R is None or camera_params[right_cam_id].P is None:
            continue
        R_rect_left = camera_params[cam_id].R
        R_rect_right = camera_params[right_cam_id].R
        P_right = camera_params[right_cam_id].P

        R = R_rect_left.T @ R_rect_right
        t_rect = np.array([-P_right[0, 3] / P_right[0, 0], 0, 0])
        t = R_rect_left.T @ t_rect
        tm.add_transform(right_cam_id, cam_id, se3_from_rot_trans(R, t))

    for idx, params in enumerate(camera_params):
        params.T = tm.get_transform(idx, 0)

    target_poses = []
    last_target_pose = np.eye(4)
    last_target_pose[2, 3] = 1.0
    for frame_idx in range(len(correspondences)):
        if tm.has_frame(f"target_{frame_idx}"):
            last_target_pose = tm.get_transform(f"target_{frame_idx}", 0)
        target_poses.append(last_target_pose)

    logger.info("Extrinsics PnP estimation complete")
    return camera_params, np.array(target_poses)


# ---------- Ceres Bundle Adjustment ----------


def _reprojection_cost(
    obs_uv: torch.Tensor,
    target_xyz: torch.Tensor,
    T_orig_cam: torch.Tensor,
    T_orig_target: torch.Tensor,
    K: torch.Tensor,
    distort_fn,
):
    T_orig_cam = se3_exp_map(T_orig_cam)
    T_orig_target = se3_exp_map(T_orig_target)
    T_cam_target = se3_inv_torch(T_orig_cam) @ T_orig_target
    reproj_uv, _ = reproject_torch(target_xyz, K, T_cam_target, distort_fn)
    residuals = obs_uv - reproj_uv
    return residuals.reshape(-1)


_reprojection_cost_jac = torch.func.jacfwd(_reprojection_cost, argnums=(2, 3))


class ReprojectionCost(pyceres.CostFunction):
    def __init__(self, obs_uv, target_xyz, K, D, distort_model: str | None = None):
        super().__init__()
        self.num_points = obs_uv.shape[0]
        self.set_num_residuals(2 * self.num_points)
        self.set_parameter_block_sizes([6, 6])
        self.obs_uv = torch.from_numpy(obs_uv).double()
        self.target_xyz = torch.from_numpy(target_xyz).double()
        self.K = torch.from_numpy(K).double()
        if distort_model is not None:
            if distort_model == DistortionModel.POLYNOMIAL.value:
                self.distort_fn = partial(distort_polynomial_torch, coeffs=torch.from_numpy(D).double())
            else:
                raise ValueError(f"Unsupported distortion model: {distort_model}")
        else:
            self.distort_fn = None

    def Evaluate(self, parameters, residuals, jacobians=None):
        T_orig_cam = torch.from_numpy(parameters[0])
        T_orig_target = torch.from_numpy(parameters[1])
        res = _reprojection_cost(
            self.obs_uv,
            self.target_xyz,
            T_orig_cam,
            T_orig_target,
            self.K,
            self.distort_fn,
        )
        residuals[:] = res.cpu().numpy()
        if jacobians is not None:
            jac = _reprojection_cost_jac(
                self.obs_uv,
                self.target_xyz,
                T_orig_cam,
                T_orig_target,
                self.K,
                self.distort_fn,
            )
            if jacobians[0] is not None:
                jacobians[0][:] = jac[0].cpu().numpy().ravel()
            if jacobians[1] is not None:
                jacobians[1][:] = jac[1].cpu().numpy().ravel()
        return True


class ParamHistoryCallback(pyceres.IterationCallback):
    """Records a deep copy of parameter arrays at each solver iteration."""

    def __init__(self, params: Any):
        super().__init__()
        self.params = params
        self.history: list = []

    def __call__(self, summary: pyceres.SolverSummary):
        self.history.append(copy.deepcopy(self.params))
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


def extrinsics_solve_ba(
    correspondences: list[list[np.ndarray | None]],
    target_xyz: np.ndarray,
    camera_params: list[CameraParam],
    init_target_poses: np.ndarray,
    max_num_iterations: int = 50,
    return_history: bool = False,
):
    """Optimize camera extrinsics and target poses via bundle adjustment.

    Args:
        correspondences: List of frames, each a list of per-camera observations.
        target_xyz: (P, 3) target 3D points.
        camera_params: List of camera parameters with initial T estimates.
        init_target_poses: (N, 4, 4) initial target poses.
        max_num_iterations: Maximum BA iterations.
        return_history: If True, return per-iteration history of camera params
            and target poses instead of only the final result.

    Returns:
        If return_history is False:
            (summary, camera_params, target_poses)
        If return_history is True:
            (summary, camera_params_history, target_poses_history)
            where each history entry is per solver iteration.
    """
    logger.info(
        f"Running bundle adjustment"
        f"\n\t- Number of frames: {len(correspondences)}"
        f"\n\t- Number of cameras: {len(camera_params)}"
        f"\n\t- Max iterations: {max_num_iterations}"
    )

    camera_params = copy.deepcopy(camera_params)
    Ds, Ks = [], []
    camera_poses_vec = []
    for param in camera_params:
        Ds.append(param.D)
        Ks.append(param.K)
        if param.T is None:
            camera_poses_vec.append(np.zeros(6))
        else:
            pose_vec = se3_log_map(torch.from_numpy(param.T))
            camera_poses_vec.append(pose_vec.cpu().numpy())
    Ds = np.stack(Ds, axis=0)
    Ks = np.stack(Ks, axis=0)
    camera_poses_vec = np.stack(camera_poses_vec, axis=0)
    target_poses_vec = se3_log_map(torch.from_numpy(init_target_poses)).cpu().numpy()

    # Build and solve the problem
    problem = pyceres.Problem()
    for frame_idx, frame in enumerate(correspondences):
        for cam_idx, obs_uv in enumerate(frame):
            if obs_uv is None or len(obs_uv) == 0:
                continue
            assert len(obs_uv) == len(target_xyz)

            cost_fn = ReprojectionCost(
                obs_uv, target_xyz, Ks[cam_idx], Ds[cam_idx],
                camera_params[cam_idx].D_model,
            )
            problem.add_residual_block(cost_fn, None, [camera_poses_vec[cam_idx], target_poses_vec[frame_idx]])

    # Fix first camera as reference
    problem.set_parameter_block_constant(camera_poses_vec[0])

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = max_num_iterations
    options.minimizer_progress_to_stdout = True

    if return_history:
        options.update_state_every_iteration = True
        cam_pose_hist = ParamHistoryCallback(camera_poses_vec)
        target_pose_hist = ParamHistoryCallback(target_poses_vec)
        options.callbacks = [cam_pose_hist, target_pose_hist]

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    # Convert back to SE(3)
    for param, pose_vec in zip(camera_params, camera_poses_vec):
        param.T = se3_exp_map(torch.from_numpy(pose_vec)).cpu().numpy()
    target_poses = se3_exp_map(torch.from_numpy(target_poses_vec)).cpu().numpy()

    logger.info(f"Bundle adjustment complete:\n{summary.FullReport()}")

    if return_history:
        camera_params_history = []
        for hist in cam_pose_hist.history:
            temp = copy.deepcopy(camera_params)
            for param, pose_vec in zip(temp, hist):
                param.T = se3_exp_map(torch.from_numpy(pose_vec)).cpu().numpy()
            camera_params_history.append(temp)
        target_poses_history = [
            se3_exp_map(torch.from_numpy(h)).cpu().numpy()
            for h in target_pose_hist.history
        ]
        return summary, camera_params_history, target_poses_history

    return summary, camera_params, target_poses


def reprojection_error_stats(
    correspondences: list[list[np.ndarray | None]],
    target_xyz: np.ndarray,
    camera_params: list[CameraParam],
    target_poses: np.ndarray,
    camera_names: list[str] | None = None,
) -> dict[str, Any]:
    """Chessboard corner reprojection errors using the same geometry as bundle adjustment.

    Args:
        correspondences: Per-frame per-camera detected corners (u, v), or None.
        target_xyz: (P, 3) chessboard points in board frame (meters).
        camera_params: Camera intrinsics / extrinsics (``T`` is camera-to-world).
        target_poses: (N, 4, 4) board pose in world frame per correspondence frame.
        camera_names: Optional labels for each camera index.

    Returns:
        JSON-serializable statistics dict with overall RMSE (pixels), per-camera
        and per-frame breakdowns.
    """
    n_cams = len(camera_params)
    if camera_names is None:
        camera_names = [str(i) for i in range(n_cams)]
    elif len(camera_names) != n_cams:
        raise ValueError("camera_names length must match camera_params")

    target_xyz_t = torch.from_numpy(np.asarray(target_xyz, dtype=np.float64)).double()

    all_err_sq: list[float] = []
    per_cam_err_sq: list[list[float]] = [[] for _ in range(n_cams)]
    per_frame: list[dict[str, Any]] = []

    for frame_idx, frame in enumerate(correspondences):
        frame_err_sq: list[float] = []
        for cam_idx, obs_uv in enumerate(frame):
            if obs_uv is None or len(obs_uv) == 0:
                continue
            param = camera_params[cam_idx]
            if param.T is None:
                continue
            assert len(obs_uv) == len(target_xyz)

            T_cam = torch.from_numpy(param.T).double()
            T_tgt = torch.from_numpy(target_poses[frame_idx]).double()
            T_cam_tgt = se3_inv_torch(T_cam) @ T_tgt

            K = torch.from_numpy(param.K).double()
            if param.D_model == DistortionModel.POLYNOMIAL.value:
                distort_fn = partial(
                    distort_polynomial_torch,
                    coeffs=torch.from_numpy(np.asarray(param.D)).double(),
                )
            else:
                distort_fn = None

            obs_t = torch.from_numpy(np.asarray(obs_uv, dtype=np.float64)).double()
            pred_uv, _ = reproject_torch(target_xyz_t, K, T_cam_tgt, distort_fn)
            diff = obs_t - pred_uv
            err_sq = (diff * diff).sum(dim=-1).detach().cpu().numpy()
            frame_err_sq.extend(err_sq.tolist())
            per_cam_err_sq[cam_idx].extend(err_sq.tolist())

        if frame_err_sq:
            fe = np.asarray(frame_err_sq, dtype=np.float64)
            per_frame.append({
                "frame_index": frame_idx,
                "rmse_pixels": float(np.sqrt(np.mean(fe))),
                "mean_error_pixels": float(np.mean(np.sqrt(fe))),
                "num_corners": int(len(fe)),
            })
            all_err_sq.extend(frame_err_sq)

    if not all_err_sq:
        return {
            "rmse_pixels": None,
            "mean_error_pixels": None,
            "median_error_pixels": None,
            "max_error_pixels": None,
            "num_corners": 0,
            "per_camera": [
                {
                    "cam_id": i,
                    "name": camera_names[i],
                    "rmse_pixels": None,
                    "num_points": 0,
                }
                for i in range(n_cams)
            ],
            "per_frame": [],
        }

    err_sq_arr = np.asarray(all_err_sq, dtype=np.float64)
    err_px = np.sqrt(err_sq_arr)

    per_cam_out: list[dict[str, Any]] = []
    for cam_idx, sq_list in enumerate(per_cam_err_sq):
        if not sq_list:
            per_cam_out.append({
                "cam_id": cam_idx,
                "name": camera_names[cam_idx],
                "rmse_pixels": None,
                "mean_error_pixels": None,
                "max_error_pixels": None,
                "num_points": 0,
            })
            continue
        sq = np.asarray(sq_list, dtype=np.float64)
        e = np.sqrt(sq)
        per_cam_out.append({
            "cam_id": cam_idx,
            "name": camera_names[cam_idx],
            "rmse_pixels": float(np.sqrt(np.mean(sq))),
            "mean_error_pixels": float(np.mean(e)),
            "max_error_pixels": float(np.max(e)),
            "num_points": int(len(sq)),
        })

    return {
        "rmse_pixels": float(np.sqrt(np.mean(err_sq_arr))),
        "mean_error_pixels": float(np.mean(err_px)),
        "median_error_pixels": float(np.median(err_px)),
        "max_error_pixels": float(np.max(err_px)),
        "std_error_pixels": float(np.std(err_px)),
        "num_corners": int(len(err_sq_arr)),
        "per_camera": per_cam_out,
        "per_frame": per_frame,
    }
