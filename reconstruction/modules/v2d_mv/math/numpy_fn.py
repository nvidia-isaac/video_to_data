from typing import Callable

import numpy as np
import pytransform3d.transformations as pt
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.spatial.transform import Rotation, Slerp


#### Projective Geometry ####

def depth_to_xyz(
    depth: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray | None = None,
    uv: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project depth to 3D points in the frame defined by T.

    Args:
        depth: (H, W) depth map in meters.
        K: (3, 3) camera intrinsic matrix.
        T: (4, 4) transform applied to camera-frame points (e.g. camera-to-world).
        mask: (H, W) optional boolean or float mask; only True / >0.5 pixels are kept.
        uv: (N, 2) optional pixel coordinates to sample; mutually exclusive with mask.

    Returns:
        (P, 3) array of 3D points.
    """
    H, W = depth.shape[:2]

    if uv is None:
        v, u = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
    else:
        assert mask is None, "mask and uv cannot be used together"
        assert (uv.ndim == 2 and uv.shape[1] == 2), (
            f"uv must be a 2D array with shape (N, 2), but got {uv.shape}"
        )
        uv = uv.round().astype(int)
        u = uv[:, 0]
        v = uv[:, 1]

    if mask is not None:
        assert (mask.shape == (H, W)), (
            f"mask must have shape ({H}, {W}), but got {mask.shape}"
        )
        mask = mask.astype(bool)
        u = u[mask]
        v = v[mask]
    else:
        u = u.reshape(-1)
        v = v.reshape(-1)

    z = depth[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    xyz = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1), np.ones_like(z)), axis=1)
    xyz = xyz @ T.T
    return xyz[:, :3]


def xyz_to_uv(
    xyz: np.ndarray,
    K: np.ndarray,
    T: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D pixel coordinates.

    Args:
        xyz: (N, 3) points. In world frame if T is given, otherwise camera frame.
        K: (3, 3) camera intrinsic matrix.
        T: (4, 4) camera-to-world extrinsic. If None, xyz is in camera frame.
        image_size: (W, H). When given, returns integer uv and in-bounds mask.

    Returns:
        If image_size is None: (N, 2) float pixel coordinates [u, v].
        If image_size is given: ((N, 2) int pixel coords, (N,) bool in-bounds mask).
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3, (
        f"points must be a 2D array with shape (N, 3), but got {xyz.shape}"
    )

    if T is not None:
        xyz_hom = np.concatenate((xyz, np.ones_like(xyz[:, :1])), axis=1)
        cam = (xyz_hom @ se3_inv(T).T)[:, :3]
    else:
        cam = xyz

    cam_z = cam[:, 2]
    uv = (cam / cam_z[:, None]) @ K.T
    uv = uv[:, :2]

    if image_size is not None:
        W, H = image_size
        uv_int = np.round(uv).astype(int)
        in_bounds = (
            (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
            & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
            & (cam_z > 0)
        )
        return uv_int, in_bounds

    return uv


def reproject(
    xyz: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    distort_fn: Callable | None = None,
):
    """
    Args:
        xyz: (P, 3) array of points in world coordinates [x, y, z]
        K: (3, 3) array of camera intrinsic matrix
        T: (4, 4) array of camera extrinsic matrix (world to camera)
    Returns:
        reprojection: (P, 2) array of reprojection in 2D image coordinates [u, v]
    """
    rot = T[:3, :3]  # (3, 3)
    trans = T[:3, 3]  # (3,)
    xyz_cam = xyz @ rot.T + trans
    xy = xyz_cam[:, :2] / xyz_cam[:, 2:3]  # (P, 2)
    if distort_fn is not None:
        xy = distort_fn(xy)
    return xy @ K[:2, :2].T + K[:2, 2]  # (P, 2)


def distort_polynomial(
    xy: np.ndarray,
    coeffs: np.ndarray,
):
    """
    Args:
        xy: (N, D, 2) array of points from N cameras in 2D image coordinates [x, y]
        coeffs: (N, 8) array of distortion coefficients
    Returns:
        uv_distorted: (N, D, 2) array of distorted points in 2D image coordinates [x, y]
    """
    x = xy[..., 0]
    y = xy[..., 1]
    r2 = x**2 + y**2  # (N, D)
    k1, k2, p1, p2, k3, k4, k5, k6 = np.split(coeffs, 8, axis=-1)  # (N, 1)
    kr = (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2)
    x_distorted = x * kr + p1 * (2.0 * x * y) + p2 * (r2 + 2.0 * x**2)
    y_distorted = y * kr + p1 * (r2 + 2.0 * y**2) + p2 * (2.0 * x * y)
    return np.stack([x_distorted, y_distorted], axis=-1)  # (N, D, 2)


#### Manifold Geometry ####

def so3_relative_angle(rot1: np.ndarray, rot2: np.ndarray) -> np.ndarray:
    """
    Compute the geodesic distance of two rotation matrices.
    Arguments:
        rot1: (N, 3, 3), N=batch dimension
        rot2: (N, 3, 3), N=batch dimension
    Returns:
        dist: (N,)
    """
    rot1 = rot1.as_matrix()
    rot2 = rot2.as_matrix()
    cos = ((rot1 @ rot2.swapaxes(-1, -2)).trace(axis1=-1, axis2=-2) - 1) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def se3_split_distance(pose1: np.array, pose2: np.array) -> tuple[np.array, np.array]:
    """
    Compute the distance of SE(3) matrices by separately computing the distance of rotation and translation.
    Arguments:
        pose1: (N, 4, 4), N=batch dimension
        pose2: (N, 4, 4), N=batch dimension
    Returns:
        rot_dist: (N,)
        trans_dist: (N,)
    """
    rot1 = Rotation.from_matrix(pose1[:, :3, :3])
    rot2 = Rotation.from_matrix(pose2[:, :3, :3])
    trans1 = pose1[:, :3, 3]
    trans2 = pose2[:, :3, 3]
    return so3_relative_angle(rot1, rot2), np.linalg.norm(trans1 - trans2, axis=-1)


def so3_quat_mean(rots: np.array) -> np.array:
    """
    Quaternion mean of SO(3) matrices.
    Arguments:
        rots: (D, 3, 3), D=mean dimension
    Returns:
        mean_rot: (3, 3)
    """
    quats = Rotation.from_matrix(rots).as_quat()
    ref_quat = quats[0]  # (4,)

    # Align quaternions to the same hemisphere
    dot_products = quats @ ref_quat
    mask = dot_products < 0
    quats[mask] = -quats[mask]
    
    # Simple averaging
    mean_quat = np.mean(quats, axis=0)

    # Normalize to ensure unit length
    mean_quat = mean_quat / np.linalg.norm(mean_quat)

    return Rotation.from_quat(mean_quat).as_matrix()


def se3_from_rot_trans(rot: np.array, trans: np.array) -> np.array:
    """
    Construct SE(3) matrix from rotation and translation.
    Arguments:
        rot: (..., 3, 3)
        trans: (..., 3)
    Returns:
        pose: (..., 4, 4)
    """
    # Ensure inputs are numpy arrays
    rot = np.asanyarray(rot)
    trans = np.asanyarray(trans)

    # Check shapes
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rot must have shape (..., 3, 3), got {rot.shape}")
    if trans.shape[-1] != 3:
        raise ValueError(f"trans must have shape (..., 3), got {trans.shape}")

    # Create the output array
    out_shape = rot.shape[:-2] + (4, 4)
    pose = np.zeros(out_shape, dtype=rot.dtype)
    
    # Fill values
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = trans
    pose[..., 3, 3] = 1.0
    
    return pose


def se3_split_mean(poses: np.array) -> np.array:
    """
    Compute the mean of SE(3) matrices by separately computing the mean of rotation and translation.
    Arguments:
        poses: (D, 4, 4), D=mean dimension
    Returns:
        mean_pose: (4, 4)
    """
    rot = poses[:, :3, :3]  # (D, 3, 3)
    trans = poses[:, :3, 3]  # (D, 3)
    mean_rot = Rotation.mean(Rotation.from_matrix(rot)).as_matrix()  # (3, 3)
    mean_trans = np.mean(trans, axis=0)  # (3,)

    mean_pose = se3_from_rot_trans(mean_rot, mean_trans)
    return mean_pose


def se3_split_mean_anisotropic(
    poses: np.ndarray,
    frame_rotations: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """
    Mean of SE(3) with per-axis weighted translation averaging.

    Each pose's translation is weighted by a precision matrix W rotated
    into world coordinates via the corresponding frame rotation.

    Args:
        poses: (D, 4, 4) world-frame poses
        frame_rotations: (D, 3, 3) rotations defining each pose's local frame
        W: (3, 3) diagonal precision matrix in the local frame
    Returns:
        mean_pose: (4, 4)
    """
    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3]  # (D, 3)
    mean_rot = Rotation.mean(Rotation.from_matrix(rot)).as_matrix()

    P_sum = np.zeros((3, 3))
    Pt_sum = np.zeros(3)
    for j in range(len(poses)):
        R_j = frame_rotations[j]  # (3, 3)
        P_j = R_j @ W @ R_j.T   # precision in world frame
        P_sum += P_j
        Pt_sum += P_j @ trans[j]
    mean_trans = np.linalg.solve(P_sum, Pt_sum)

    return se3_from_rot_trans(mean_rot, mean_trans)


# Requires scipy>=1.16.0
# def se3_karcher_mean(poses: np.array) -> np.array:
#     """
#     Karcher mean of SE(3) matrices.
#     Arguments:
#         poses: (D, 4, 4), D=mean dimension
#     Returns:
#         mean_pose: (4, 4)
#     """
#     mean_pose = se3_split_mean(poses)  # (4, 4)
#     while True:
#         rel_poses = np.linalg.inv(mean_pose) @ poses  # (D, 4, 4)
#         log_rel_poses = RigidTransform.from_matrix(rel_poses).as_exp_coords()  # (D, 6)
#         mean_log_rel_pose = np.mean(log_rel_poses, axis=0)  # (6,)
#         norm = np.linalg.norm(mean_log_rel_pose)
#         if norm < 1e-5:
#             break
#         mean_pose = mean_pose @ RigidTransform.from_exp_coords(mean_log_rel_pose).as_matrix()
#     return mean_pose


def se3_metric(
    T: np.ndarray,
    T_ref: np.ndarray,
    view_distance: float = 2.5,
) -> float:
    # Compute relative weights based on translation offset 
    # that is equivalent to a rotation of 5 degrees at view_distance
    Rt_ratio = (2 * view_distance * np.tan(np.pi / 72))**2 / (np.pi / 36)**2
    w_R = Rt_ratio / (Rt_ratio + 1)
    w_t = 1 - w_R

    T_ref_start = pt.invert_transform(T_ref) @ T
    log_T_ref_start = pt.exponential_coordinates_from_transform(T_ref_start)
    d_r = w_R * np.dot(log_T_ref_start[:3], log_T_ref_start[:3])
    d_t = w_t * np.dot(log_T_ref_start[3:], log_T_ref_start[3:])
    return np.sqrt(d_r + d_t)


# Not used?
def se3_mean_ransac(
    R_samples: np.ndarray,
    t_samples: np.ndarray,
    max_iters: int = 50,
    view_distance: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    # Randomly sample and count inliers
    T_samples = np.concatenate([R_samples, t_samples[..., np.newaxis]], axis=2)
    best_inliers = []
    for _ in range(max_iters):
        j = np.random.randrange(len(T_samples))
        T_ref = T_samples[j]
        inliers = []
        for i, T in enumerate(T_samples):
            if se3_metric(T, T_ref, view_distance) < 0.1:
                inliers.append(i)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Compute mean transform from inliers
    T_inliers = T_samples[best_inliers]
    R_inliers = T_inliers[:, :3, :3]
    t_inliers = T_inliers[:, :3, 3]
    R = Rotation.from_matrix(R_inliers).mean().as_matrix()
    t = t_inliers.mean(axis=0)
    return R, t


def se3_inv(pose_mat: np.ndarray) -> np.ndarray:
    """
    Invert an SE(3) matrix.
    Args:
        pose_mat: (..., 4, 4) array
    Returns:
        inv_pose: (..., 4, 4) array
    """
    R = pose_mat[..., :3, :3]
    t = pose_mat[..., :3, 3]
    R_inv = R.swapaxes(-1, -2)
    t_inv = -np.einsum("...ij,...j->...i", R_inv, t)
    
    inv_pose = np.zeros_like(pose_mat)
    inv_pose[..., :3, :3] = R_inv
    inv_pose[..., :3, 3] = t_inv
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def se3_inliers_trans(
    poses: np.ndarray,
    max_translation: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    trans = poses[:, :3, 3]
    centroid = trans.mean(axis=0)
    dist = np.linalg.norm(trans - centroid, axis=1)
    return poses[dist < max_translation]


# Not used?
def se3_inliers_incidence_angle(
    R_samples: np.ndarray,
    t_samples: np.ndarray,
    max_incidence_angle: float = np.pi * 50/180,
) -> tuple[np.ndarray, np.ndarray]:
    def incidence_angle(R: np.ndarray, t: np.ndarray) -> float:
        n = R @ np.array([0, 0, -1])
        dot = np.sum(t * n, axis=1)
        return np.arccos(dot / np.linalg.norm(t, axis=1) / np.linalg.norm(n, axis=1))
    
    theta = incidence_angle(R_samples, t_samples)
    valid_indices = theta < max_incidence_angle
    R_samples = R_samples[valid_indices]
    t_samples = t_samples[valid_indices]
    return R_samples, t_samples


def se3_pose_select(
    poses: np.array,
    dist_weight: float = 10.0,
    top_n: int = 2,
) -> np.array:
    """
    Select the best pose from the given poses by the minimum pairwise distance.
    Arguments:
        poses: (D, 4, 4), D=compare dimension
        dist_weight: float, weight of translation distance
        top_n: int, number of best poses to average
    Returns:
        best_pose: (4, 4)
    """
    D = poses.shape[0]
    repeated_poses = np.repeat(poses, D, axis=0)  # (D*D, 4, 4)
    tiled_poses = np.tile(poses, (D, 1, 1))  # (D*D, 4, 4)
    rot_dist, trans_dist = se3_split_distance(repeated_poses, tiled_poses)  # (D*D,), (D*D,)
    rot_dist = rot_dist.reshape(D, D).mean(axis=-1)
    trans_dist = trans_dist.reshape(D, D).mean(axis=-1)
    dist = rot_dist + dist_weight * trans_dist
    best_idx = np.argsort(dist)[:top_n]
    return best_idx, se3_split_mean(poses[best_idx])


#### Filtering ####

class LinearOneEuroFilter:
    """ Linear OneEuroFilter.

    Based on https://jaantollander.com/post/noise-filtering-using-one-euro-filter/
    """
    def __init__(
        self,
        x_shape: tuple[int, ...],
        min_f_cutoff: float = 1.0,
        beta: float = 0.0,
        df_cutoff: float = 1.0,
        f_sample: float = 30.0,
    ):
        self.min_f_cutoff = np.ones(x_shape) * min_f_cutoff
        self.beta = np.ones(x_shape) * beta
        self.df_cutoff = np.ones(x_shape) * df_cutoff
        self.f_sample = f_sample

        self.dx_prev = np.zeros(x_shape)
        self.x_prev = None
        self.t_prev = None
    
    def _smoothing_factor(self, t_e, f_cutoff):
        r = 2 * np.pi * f_cutoff * t_e
        return r / (r + 1)

    def _linear_update(self, a, x, x_prev):
        return (a * x) +  ((1 - a) * x_prev)

    def __call__(self, x: np.array, t: float = None) -> np.array:
        """Compute the filtered signal."""
        if self.x_prev is None:
            # Initialize x_prev and t_prev.
            self.x_prev = x
            if t is None:
                self.t_prev = 0.0
            else:
                self.t_prev = t
            return x

        # Compute the time delta.
        if t is None:
            t_e = 1.0 / self.f_sample
            self.t_prev += t_e
        else:
            t_e = t - self.t_prev
            self.t_prev = t

        # The filtered derivative of the signal.
        a_d = self._smoothing_factor(t_e, self.df_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._linear_update(a_d, dx, self.dx_prev)

        # The filtered signal.
        f_cutoff = self.min_f_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(t_e, f_cutoff)
        x_hat = self._linear_update(a, x, self.x_prev)

        # Update the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


class RotationOneEuroFilter:
    """ Rotation OneEuroFilter.

    Uses Slerp to interpolate between the previous and current rotation.
    """
    def __init__(
        self,
        min_f_cutoff: float = 1.0,
        beta: float = 0.0,
        df_cutoff: float = 1.0,
        f_sample: float = 30.0,
    ):
        self.min_f_cutoff = min_f_cutoff
        self.beta = beta
        self.df_cutoff = df_cutoff
        self.f_sample = f_sample

        self.w_prev = np.zeros(3)  # angular velocity
        self.rot_prev: Rotation | None = None  # rotation
        self.t_prev = None
    
    def _smoothing_factor(self, t_e, f_cutoff):
        r = 2 * np.pi * f_cutoff * t_e
        return r / (r + 1)

    def _linear_update(self, a, w, w_prev):
        return (a * w) +  ((1 - a) * w_prev)

    def _slerp_update(self, a, rot, rot_prev):
        slerp = Slerp([0, 1], Rotation.concatenate([rot_prev, rot]))
        return slerp(a)
    
    def __call__(self, mat: np.array, t: float = None) -> np.array:
        """Compute the filtered signal."""
        rot = Rotation.from_matrix(mat)  # [x, y, z, w]
        if self.rot_prev is None:
            # Initialize q_prev and t_prev.
            self.rot_prev = rot
            if t is None:
                self.t_prev = 0.0
            else:
                self.t_prev = t
            return mat

        # Compute the time delta.
        if t is None:
            t_e = 1.0 / self.f_sample
            self.t_prev += t_e
        else:
            t_e = t - self.t_prev
            self.t_prev = t

        # Compute the filtered derivative (angular velocity) of the signal.
        a_d = self._smoothing_factor(t_e, self.df_cutoff)
        rot_rel = rot * self.rot_prev.inv()
        w = rot_rel.as_rotvec() / t_e
        w_hat = self._linear_update(a_d, w, self.w_prev)

        # Compute the filtered signal.
        f_cutoff = self.min_f_cutoff + self.beta * np.linalg.norm(w_hat)
        a = self._smoothing_factor(t_e, f_cutoff)
        rot_hat = self._slerp_update(a, rot, self.rot_prev)

        # Update the previous values.
        self.rot_prev = rot_hat
        self.w_prev = w_hat

        return rot_hat.as_matrix()


class PoseOneEuroFilter:
    def __init__(
        self,
        trans_min_f_cutoff: float = 1.0,
        trans_beta: float = 2.0,
        trans_df_cutoff: float = 1.0,
        rot_min_f_cutoff: float = 0.5,
        rot_beta: float = 8.0,
        rot_df_cutoff: float = 1.0,
        f_sample: float = 30.0,
    ):
        self.trans_filter = LinearOneEuroFilter(x_shape=(3,), min_f_cutoff=trans_min_f_cutoff, beta=trans_beta, df_cutoff=trans_df_cutoff, f_sample=f_sample)
        self.rot_filter = RotationOneEuroFilter(min_f_cutoff=rot_min_f_cutoff, beta=rot_beta, df_cutoff=rot_df_cutoff, f_sample=f_sample)

    def __call__(self, pose: np.array, t: float = None) -> np.array:
        res = pose.copy()
        res[:3, 3] = self.trans_filter(pose[:3, 3], t)
        res[:3, :3] = self.rot_filter(pose[:3, :3], t)
        return res


def linear_one_euro_filter(
    signal: np.array,
    min_f_cutoff: float = 1.0,
    beta: float = 0.0,
    df_cutoff: float = 1.0,
    f_sample: float = 30.0,
):
    """
    Args:
        signal: (..., L) tensor of signal.
        min_cutoff: Minimum cutoff frequency.
        beta: Speed coefficient. Increase to reduce speed lag.
        d_cutoff: cutoff frequency for the derivative.
        freq: frequency of the signal.
    Returns:
        filtered_signal: (..., L) tensor of filtered signal.
    """
    oefilter = LinearOneEuroFilter(
        x_shape=signal.shape[:-1],
        min_f_cutoff=min_f_cutoff,
        beta=beta,
        df_cutoff=df_cutoff,
        f_sample=f_sample,
    )
    new_signal = np.zeros_like(signal)
    for i in range(0, signal.shape[-1]):
        new_signal[..., i] = oefilter(signal[..., i])
    return new_signal


class LinearButterworthFilter:
    def __init__(
        self,
        order: int = 2,
        f_cutoff: float = 5.0,
        f_sample: float = 30.0,
    ):
        self.f_cutoff = f_cutoff
        self.order = order
        self.sos = butter(N=order, Wn=f_cutoff, btype="lowpass", analog=False, output="sos", fs=f_sample)
        self.z_prev = None
    
    def __call__(self, x: np.array) -> np.array:
        if self.z_prev is None:
            z_prev = sosfilt_zi(self.sos)  # (n_sections, 2)
            z_prev = z_prev[:, :, None] * x[None, None, :]
            res, self.z_prev = sosfilt(self.sos, [x], zi=z_prev, axis=0)
            return res[0]

        res, self.z_prev = sosfilt(self.sos, [x], zi=self.z_prev, axis=0)
        return res[0]


class RotationButterworthFilter:
    def __init__(
        self,
        order: int = 2,
        f_cutoff: float = 5.0,
        f_sample: float = 30.0,
    ):
        self.f_cutoff = f_cutoff
        self.order = order
        self.sos = butter(N=order, Wn=f_cutoff, btype="lowpass", analog=False, output="sos", fs=f_sample)
        self.rot_prev = None
        self.z_prev = None

    def __call__(self, mat: np.array) -> np.array:
        rot = Rotation.from_matrix(mat)

        if self.rot_prev is None:
            self.rot_prev = rot
            z_prev = sosfilt_zi(self.sos)  # (n_sections, 2)
            self.z_prev = z_prev[:, :, None] * np.zeros((1, 1, 3))
            return mat

        d_rot = rot * self.rot_prev.inv()
        d_rotvec_filtered, self.z_prev = sosfilt(self.sos, [d_rot.as_rotvec()], zi=self.z_prev, axis=0)
        d_rot_filtered = Rotation.from_rotvec(d_rotvec_filtered[0])
        self.rot_prev = d_rot_filtered * self.rot_prev
        return self.rot_prev.as_matrix()


class PoseButterworthFilter:
    def __init__(
        self,
        trans_order: int = 1,
        trans_f_cutoff: float = 5.0,
        rot_order: int = 1,
        rot_f_cutoff: float = 5.0,
        f_sample: float = 30.0,
    ):
        self.trans_filter = LinearButterworthFilter(order=trans_order, f_cutoff=trans_f_cutoff, f_sample=f_sample)
        self.rot_filter = RotationButterworthFilter(order=rot_order, f_cutoff=rot_f_cutoff, f_sample=f_sample)

    def __call__(self, pose: np.array) -> np.array:
        res = pose.copy()
        res[:3, 3] = self.trans_filter(pose[:3, 3])
        res[:3, :3] = self.rot_filter(pose[:3, :3])
        return res


def run_pose_filter_offline(poses: np.array, filter_fn: Callable) -> np.array:
    """
    Run pose filter offline.
    Arguments:
        poses: (T, 4, 4), T=time dimension
        filter_fn: Callable, filter function
    Returns:
        Filtered poses of shape (N, 4, 4).
    """
    assert poses.shape[1:] == (4, 4), f"poses must be of shape (T, 4, 4), but got {poses.shape}"
    filtered_poses = np.zeros_like(poses)
    for i in range(poses.shape[0]):
        filtered_poses[i] = filter_fn(poses[i])
    return filtered_poses


def _cutoff_to_gain(f_cutoffs: np.array, f_sample: float) -> np.array:
    """Convert from cutoff frequency to the gain factor that is applied to the new sample.
    Arguments:
        f_cutoffs: array, cutoff frequencies
        f_sample: float, sample frequency
    Returns:
        Gains in the same shape as f_cutoffs.
    """
    # This is a common approximation, for computational efficiency.
    # r = 2 * np.pi * cutoff / f_sample
    # return r / (r + 1)
    
    # This is the exact formula for a first-order low-pass filter.
    return 1 - np.exp(-2 * np.pi * f_cutoffs / f_sample)


def _forward_backward_filter(x: np.array, gain_schedule: np.array) -> np.array:
    """A forward-backward zero-phase filter with a predefined gain schedule.
    Arguments:
        x: (T, ...), T=time dimension
        gain_schedule: array, gain schedule
    Returns:
        Filtered signal of shape (T, ...).
    """
    assert gain_schedule.shape == x.shape, f"gain_schedule must be of shape {x.shape}"
    x = x.copy()
    # Forward pass
    for i in range(1, x.shape[0]):
        x[i] = x[i] * gain_schedule[i] + x[i-1] * (1 - gain_schedule[i])
    # Backward pass
    for i in range(x.shape[0] - 2, -1, -1):
        x[i] = x[i] * gain_schedule[i] + x[i+1] * (1 - gain_schedule[i])
    return x


def linear_two_euro_filter(
    x: np.array,
    min_f_cutoff: float = 1.0,
    beta: float = 0.5,
    df_cutoff: float = 1.0,
    f_sample: float = 30.0,
):
    """
    Zero-phase offline adaptive filter for linear signal based on the One Euro filter.
    Arguments:
        x: (T, ...), T=time dimension
        min_f_cutoff: float, minimum cutoff frequency of the signal filter
        beta: float, controls the sensitivity of adaptive gain to the derivative
        df_cutoff: float, cutoff frequency of the derivative filter
        f_sample: float, sample frequency
    Returns:
        Filtered signal of shape (T, ...).
    """
    assert min_f_cutoff > 0, f"min_f_cutoff must be greater than 0"
    assert f_sample > min_f_cutoff / 2, f"f_sample must be greater than min_f_cutoff / 2"
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        
    # Compute the gain schedule for the full signal.
    # To do so, we compute the discrete derivative and apply a forward-backward filter.
    dx = np.zeros_like(x)
    dx[1:] = x[1:] - x[:-1]
    # df_cutoff is constant, so this behaves like a first-order low-pass filter.
    a_d = _cutoff_to_gain(df_cutoff * np.ones_like(x), f_sample)
    dx_filt = _forward_backward_filter(dx, a_d)

    # Apply a forward-backward filter with the adaptive gain.
    f_cutoff = min_f_cutoff + beta * np.abs(dx_filt)
    a = _cutoff_to_gain(f_cutoff, f_sample)
    x_filt = _forward_backward_filter(x, a)
    return x_filt


def _forward_backward_filter_slerp(rot: Rotation, gain_schedule: np.array) -> Rotation:
    """
    A forward-backward zero-phase filter with a predefined gain schedule using Slerp.
    Arguments:
        rot: Rotation object containing a sequence of rotations
        gain_schedule: (T,), T=time dimension
    Returns:
        Filtered rotations.
    """
    assert gain_schedule.shape == (len(rot),), f"gain_schedule must be of shape ({len(rot)},)"
    
    # Forward pass
    res_fwd = [rot[0]]
    for i in range(1, len(rot)):
        res_fwd.append(Slerp([0, 1], Rotation.concatenate([res_fwd[i-1], rot[i]]))(gain_schedule[i]))
    
    # Backward pass
    res_bwd = [None] * len(rot)
    res_bwd[-1] = res_fwd[-1]
    for i in range(len(rot) - 2, -1, -1):
        res_bwd[i] = Slerp([0, 1], Rotation.concatenate([res_bwd[i+1], res_fwd[i]]))(gain_schedule[i])
        
    return Rotation.concatenate(res_bwd)


def rotation_two_euro_filter(
    rot: Rotation,
    min_f_cutoff: float = 1.0,
    beta: float = 0.5,
    df_cutoff: float = 1.0,
    f_sample: float = 30.0,
):
    """
    Zero-phase offline adaptive filter for rotation based on the One Euro filter.
    Arguments:
        rot: Rotation object containing a sequence of rotations
        min_f_cutoff: float, minimum cutoff frequency of the signal filter
        beta: float, controls the sensitivity of adaptive gain to the derivative
        df_cutoff: float, cutoff frequency of the derivative filter
        f_sample: float, sample frequency
    Returns:
        Filtered rotations.
    """
    assert min_f_cutoff > 0, f"min_f_cutoff must be greater than 0"
    assert f_sample > min_f_cutoff / 2, f"f_sample must be greater than min_f_cutoff / 2"

    # Compute the gain schedule for the full signal.
    # To do so, we compute the discrete derivative and apply a forward-backward filter.
    d_rot = rot[1:] * rot[:-1].inv()
    w = np.zeros((len(rot), 3))  # Angular velocity
    w[1:] = d_rot.as_rotvec() * f_sample
    # df_cutoff is constant, so this behaves like a first-order low-pass filter.
    a_d = _cutoff_to_gain(df_cutoff * np.ones_like(w), f_sample)
    w_filt = _forward_backward_filter(w, a_d)

    # Apply a forward-backward filter with the adaptive gain.
    f_cutoff = min_f_cutoff + beta * np.linalg.norm(w_filt, axis=-1)
    a = _cutoff_to_gain(f_cutoff, f_sample)
    rot_filt = _forward_backward_filter_slerp(rot, a)
    return rot_filt


def pose_two_euro_filter(
    poses: np.array,
    trans_min_f_cutoff: float = 2.0,
    trans_beta: float = 2.0,
    trans_df_cutoff: float = 2.0,
    rot_min_f_cutoff: float = 2.0,
    rot_beta: float = 2.0,
    rot_df_cutoff: float = 2.0,
    f_sample: float = 30.0,
):
    """
    Zero-phase offline adaptive filter for pose based on the One Euro filter.
    Arguments:
        poses: (T, 4, 4), T=time dimension
        trans_min_f_cutoff: float, minimum cutoff frequency of the translation filter
        trans_beta: float, controls the sensitivity of adaptive gain to the derivative
        trans_df_cutoff: float, cutoff frequency of the derivative filter
        rot_min_f_cutoff: float, minimum cutoff frequency of the rotation filter
        rot_beta: float, controls the sensitivity of adaptive gain to the derivative
        rot_df_cutoff: float, cutoff frequency of the derivative filter
        f_sample: float, sample frequency
    Returns:
        Filtered poses of shape (T, 4, 4).
    """
    trans = poses[:, :3, 3]  # (T, 3)
    rot = Rotation.from_matrix(poses[:, :3, :3])  # (T, 3, 3)
    trans_filt = linear_two_euro_filter(trans, trans_min_f_cutoff, trans_beta, trans_df_cutoff, f_sample)  # (T, 3)
    rot_filt = rotation_two_euro_filter(rot, rot_min_f_cutoff, rot_beta, rot_df_cutoff, f_sample)  # (T, 3, 3)
    
    T = poses.shape[0]
    poses_filt = np.zeros((T, 4, 4))
    poses_filt[:, :3, :3] = rot_filt.as_matrix()
    poses_filt[:, :3, 3] = trans_filt
    poses_filt[:, 3, 3] = 1
    return poses_filt


#### Visibility ####


def visible_vertices(
    verts: np.ndarray,
    mesh_zbuf: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    zbuf_eps: float = 0.005,
) -> np.ndarray:
    """Determine vertex visibility using a rasterized mesh z-buffer.

    A vertex is visible if:
    1. It projects inside image bounds with positive z
    2. Its camera-frame z matches the mesh z-buffer within zbuf_eps

    Args:
        verts: (V, 3) world-frame vertices.
        mesh_zbuf: (H, W) mesh z-buffer from pyrender (0 = background).
        K: (3, 3) intrinsics at render resolution.
        T: (4, 4) camera-to-world extrinsic.
        zbuf_eps: Tolerance for z-buffer matching (meters).

    Returns:
        (V,) boolean array.
    """
    H, W = mesh_zbuf.shape[:2]
    T_inv = se3_inv(T)

    verts_hom = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
    verts_cam = (verts_hom @ T_inv.T)[:, :3]
    vert_z = verts_cam[:, 2]

    uv = (verts_cam / vert_z[:, None]) @ K.T
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (vert_z > 0)
    visible = np.zeros(verts.shape[0], dtype=bool)

    idx = np.where(in_bounds)[0]
    u_valid, v_valid = u[idx], v[idx]
    zbuf_at_pixel = mesh_zbuf[v_valid, u_valid]

    zbuf_matches = np.abs(vert_z[idx] - zbuf_at_pixel) < zbuf_eps
    visible[idx] = zbuf_matches & (zbuf_at_pixel > 0)
    return visible
