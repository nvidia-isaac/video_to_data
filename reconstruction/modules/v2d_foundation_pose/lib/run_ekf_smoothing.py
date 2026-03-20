"""
EKF + RTS smoother for FoundationPose output pose sequences.

Applies a Bayesian forward-backward smoother to a directory of per-frame
6-DoF poses. Uses an Error-State Kalman Filter (ESKF) with a random walk
process model:
  - Translation in R³ (standard linear KF)
  - Rotation in SO(3) via error-state representation (perturbation in so(3))

Measurement noise is optionally scaled by per-frame mask IoU: low-IoU frames
(likely tracking loss or occlusion) are trusted less, high-IoU frames more.

The RTS (Rauch-Tung-Striebel) backward smoother pass gives the globally
optimal estimate conditioned on all observations, not just past frames.

Translation noise is anisotropic: separate XY (lateral) and Z (depth)
parameters are exposed because monocular depth estimates have systematically
higher noise along the optical axis than laterally. Increasing
measurement_noise_z relative to measurement_noise_xy causes the filter to
smooth depth more aggressively while leaving lateral estimates largely intact.

Tuning guidance:
  process_noise_*      — how much the pose can change per frame; increase for
                         fast-moving objects, decrease to enforce more smoothness
  measurement_noise_*  — baseline trust in FP pose estimates; increase to smooth
                         more aggressively, decrease to track measurements closely
  min_iou              — IoU floor that prevents infinite measurement noise on
                         completely occluded frames
"""
import argparse
import logging
import os

import numpy as np
from scipy.spatial.transform import Rotation

from v2d.common.datatypes import CameraIntrinsics, Mask, Transform3d
from v2d.mesh.lib.mesh import Mesh
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_poses(poses_dir: str) -> tuple[list[int], list[Transform3d]]:
    files = sorted(f for f in os.listdir(poses_dir) if f.endswith('.json'))
    indices = [int(os.path.splitext(f)[0]) for f in files]
    poses = [Transform3d.load(os.path.join(poses_dir, f)) for f in files]
    return indices, poses


def _compute_iou_scores(
    tracker: FoundationPoseTracker,
    poses: list[Transform3d],
    intrinsics: CameraIntrinsics,
    masks_folder: str | None,
    indices: list[int],
) -> np.ndarray:
    """Render mesh at each pose and compute IoU against observed mask."""
    if masks_folder is None:
        logger.info("No masks_folder provided — using uniform IoU weights")
        return np.ones(len(poses))

    scores = np.ones(len(poses))
    for i, (idx, pose) in enumerate(zip(indices, poses)):
        mask_path = os.path.join(masks_folder, f"{idx:06d}.png")
        if os.path.exists(mask_path):
            mask = Mask.load(mask_path)
            scores[i] = tracker._mask_iou(mask, intrinsics, pose)
        logger.debug(f"Frame {idx}: IoU={scores[i]:.3f}")

    logger.info(f"IoU scores — mean={scores.mean():.3f}  min={scores.min():.3f}  "
                f"frames<0.3: {(scores < 0.3).sum()}")
    return scores


def _pose_to_tr(pose: Transform3d) -> tuple[np.ndarray, Rotation]:
    M = pose.to_matrix()
    return M[:3, 3].copy(), Rotation.from_matrix(M[:3, :3])


def _tr_to_pose(t: np.ndarray, r: Rotation) -> Transform3d:
    M = np.eye(4)
    M[:3, :3] = r.as_matrix()
    M[:3, 3] = t
    return Transform3d.from_matrix(M)


def _solve_gain(P_pred: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Compute Kalman gain K = P_pred @ S^{-1} for symmetric PD matrices."""
    # Solve S @ K.T = P_pred.T  →  K = (S^{-1} @ P_pred.T).T
    return np.linalg.solve(S, P_pred.T).T


# ---------------------------------------------------------------------------
# ESKF + RTS core
# ---------------------------------------------------------------------------

def _eskf_rts(
    translations: np.ndarray,   # (N, 3)
    rotations: list[Rotation],  # N elements
    iou_scores: np.ndarray,     # (N,) in [0, 1]
    process_noise_xy: float,
    process_noise_z: float,
    process_noise_r: float,
    measurement_noise_xy: float,
    measurement_noise_z: float,
    measurement_noise_r: float,
    min_iou: float,
) -> tuple[np.ndarray, list[Rotation]]:
    """
    Forward ESKF pass followed by RTS backward smoother.

    Process model: random walk for both translation and rotation.
    Rotation is handled in error-state (so(3) perturbation) space.
    Translation uses anisotropic noise: XY (lateral) and Z (depth) are
    tuned separately because monocular depth has higher uncertainty along
    the optical axis.

    Returns:
        t_smooth: (N, 3) smoothed translations
        r_smooth: list of N smoothed Rotation objects
    """
    N = len(translations)
    Q_t = np.diag([process_noise_xy ** 2, process_noise_xy ** 2, process_noise_z ** 2])
    Q_r = process_noise_r ** 2 * np.eye(3)
    I3 = np.eye(3)

    def _R_t(iou: float) -> np.ndarray:
        s = 1.0 / iou
        return np.diag([(measurement_noise_xy * s) ** 2,
                        (measurement_noise_xy * s) ** 2,
                        (measurement_noise_z  * s) ** 2])

    # Forward pass storage
    t_filt   = np.zeros((N, 3))
    P_t_filt = np.zeros((N, 3, 3))
    t_pred   = np.zeros((N, 3))
    P_t_pred = np.zeros((N, 3, 3))

    r_filt   = [None] * N
    P_r_filt = np.zeros((N, 3, 3))
    P_r_pred = np.zeros((N, 3, 3))

    # Initialise with first measurement; uncertainty = measurement noise
    iou0 = max(float(iou_scores[0]), min_iou)
    t_filt[0]   = translations[0]
    P_t_filt[0] = _R_t(iou0)
    r_filt[0]   = rotations[0]
    P_r_filt[0] = (measurement_noise_r / iou0) ** 2 * I3

    # ---- Forward pass ----
    for k in range(1, N):
        iou   = max(float(iou_scores[k]), min_iou)
        R_t_k = _R_t(iou)
        R_r_k = (measurement_noise_r / iou) ** 2 * I3

        # Predict (random walk — nominal rotation carries forward unchanged)
        t_pred[k]   = t_filt[k - 1]
        P_t_pred[k] = P_t_filt[k - 1] + Q_t
        P_r_pred[k] = P_r_filt[k - 1] + Q_r
        r_pred_k    = r_filt[k - 1]          # predicted nominal rotation

        # Translation update
        S_t       = P_t_pred[k] + R_t_k
        K_t       = _solve_gain(P_t_pred[k], S_t)
        t_filt[k] = t_pred[k] + K_t @ (translations[k] - t_pred[k])
        P_t_filt[k] = (I3 - K_t) @ P_t_pred[k]

        # Rotation update (ESKF)
        # Innovation in so(3): rotation from predicted nominal to measurement
        innov_r   = (rotations[k] * r_pred_k.inv()).as_rotvec()
        S_r       = P_r_pred[k] + R_r_k
        K_r       = _solve_gain(P_r_pred[k], S_r)
        delta_r   = K_r @ innov_r
        r_filt[k] = Rotation.from_rotvec(delta_r) * r_pred_k
        P_r_filt[k] = (I3 - K_r) @ P_r_pred[k]

    # ---- RTS Backward smoother ----
    t_smooth   = t_filt.copy()
    r_smooth   = list(r_filt)
    P_t_smooth = P_t_filt.copy()
    P_r_smooth = P_r_filt.copy()

    for k in range(N - 2, -1, -1):
        # Translation RTS
        G_t          = _solve_gain(P_t_filt[k], P_t_pred[k + 1])
        t_smooth[k]  = t_filt[k] + G_t @ (t_smooth[k + 1] - t_pred[k + 1])
        P_t_smooth[k] = (P_t_filt[k]
                         + G_t @ (P_t_smooth[k + 1] - P_t_pred[k + 1]) @ G_t.T)

        # Rotation RTS
        # Smoothing correction: rotation from r_filt[k] (= predicted r_{k+1|k})
        # to the already-smoothed r_smooth[k+1]
        G_r          = _solve_gain(P_r_filt[k], P_r_pred[k + 1])
        delta_smooth = (r_smooth[k + 1] * r_filt[k].inv()).as_rotvec()
        r_smooth[k]  = Rotation.from_rotvec(G_r @ delta_smooth) * r_filt[k]
        P_r_smooth[k] = (P_r_filt[k]
                         + G_r @ (P_r_smooth[k + 1] - P_r_pred[k + 1]) @ G_r.T)

    return t_smooth, r_smooth


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ekf_smoothing(
    poses_dir: str,
    mesh_path: str,
    intrinsics_path: str,
    weights_dir: str,
    output_dir: str,
    masks_folder: str = None,
    process_noise_xy: float = 0.005,
    process_noise_z: float = 0.005,
    process_noise_r: float = 0.01,
    measurement_noise_xy: float = 0.02,
    measurement_noise_z: float = 0.08,
    measurement_noise_r: float = 0.05,
    min_iou: float = 0.1,
) -> None:
    """Smooth a FoundationPose pose sequence with an ESKF + RTS smoother.

    Args:
        poses_dir:             Directory of per-frame Transform3d JSON files.
        mesh_path:             Mesh used during tracking (for IoU rendering).
        intrinsics_path:       Camera intrinsics JSON.
        weights_dir:           FoundationPose weights directory.
        output_dir:            Destination for smoothed pose JSON files.
        masks_folder:          Optional SAM2 mask folder for IoU-weighted noise.
                               If None, all frames are weighted equally.
        process_noise_xy:      Lateral (X, Y) random-walk std per frame (metres).
                               Default 0.005.
        process_noise_z:       Depth (Z) random-walk std per frame (metres).
                               Default 0.005 — same as XY, since process noise
                               models true object dynamics (isotropic).
        process_noise_r:       Rotation random-walk std per frame (radians).
                               Default 0.01.
        measurement_noise_xy:  Baseline lateral measurement std (metres).
                               Default 0.02.
        measurement_noise_z:   Baseline depth measurement std (metres).
                               Default 0.1 — set higher than XY to reflect
                               monocular depth uncertainty along the optical axis;
                               increase further to smooth Z more aggressively.
        measurement_noise_r:   Baseline rotation measurement std (radians).
                               Default 0.05.
        min_iou:               IoU floor to cap measurement noise on occluded
                               frames. Default 0.1.
    """
    indices, poses = _load_poses(poses_dir)
    logger.info(f"Loaded {len(poses)} poses from {poses_dir}")

    mesh = Mesh.load(mesh_path)
    tracker = FoundationPoseTracker(mesh, weights_dir)
    intrinsics = CameraIntrinsics.load(intrinsics_path)

    iou_scores = _compute_iou_scores(tracker, poses, intrinsics, masks_folder, indices)

    translations = np.array([_pose_to_tr(p)[0] for p in poses])
    rotations    = [_pose_to_tr(p)[1] for p in poses]

    logger.info("Running ESKF forward pass + RTS backward smoother...")
    t_smooth, r_smooth = _eskf_rts(
        translations, rotations, iou_scores,
        process_noise_xy=process_noise_xy,
        process_noise_z=process_noise_z,
        process_noise_r=process_noise_r,
        measurement_noise_xy=measurement_noise_xy,
        measurement_noise_z=measurement_noise_z,
        measurement_noise_r=measurement_noise_r,
        min_iou=min_iou,
    )

    os.makedirs(output_dir, exist_ok=True)
    for i, idx in enumerate(indices):
        _tr_to_pose(t_smooth[i], r_smooth[i]).save(
            os.path.join(output_dir, f"{idx:06d}.json")
        )

    t_delta = np.linalg.norm(t_smooth - translations, axis=1)
    logger.info(f"Translation correction — mean={t_delta.mean()*100:.1f}cm  "
                f"max={t_delta.max()*100:.1f}cm")
    logger.info(f"Smoothed poses written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESKF + RTS pose smoother for FoundationPose output")
    parser.add_argument("--poses_dir",            required=True)
    parser.add_argument("--mesh_path",            required=True)
    parser.add_argument("--intrinsics_path",      required=True)
    parser.add_argument("--weights_dir",          required=True)
    parser.add_argument("--output_dir",           required=True)
    parser.add_argument("--masks_folder",          default=None)
    parser.add_argument("--process_noise_xy",      type=float, default=0.005)
    parser.add_argument("--process_noise_z",       type=float, default=0.005)
    parser.add_argument("--process_noise_r",       type=float, default=0.01)
    parser.add_argument("--measurement_noise_xy",  type=float, default=0.02)
    parser.add_argument("--measurement_noise_z",   type=float, default=0.1)
    parser.add_argument("--measurement_noise_r",   type=float, default=0.05)
    parser.add_argument("--min_iou",               type=float, default=0.1)

    args = parser.parse_args()
    run_ekf_smoothing(
        args.poses_dir,
        args.mesh_path,
        args.intrinsics_path,
        args.weights_dir,
        args.output_dir,
        masks_folder=args.masks_folder,
        process_noise_xy=args.process_noise_xy,
        process_noise_z=args.process_noise_z,
        process_noise_r=args.process_noise_r,
        measurement_noise_xy=args.measurement_noise_xy,
        measurement_noise_z=args.measurement_noise_z,
        measurement_noise_r=args.measurement_noise_r,
        min_iou=args.min_iou,
    )
