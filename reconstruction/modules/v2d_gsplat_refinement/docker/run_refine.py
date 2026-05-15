from v2d.docker.container import run_in_container
from v2d.gsplat_refinement.docker._config import IMAGE_NAME, MODULES_DIR


def run_refine(
    frames_dir: str,
    intrinsics_path: str,
    object_mesh_path: str,
    object_poses_dir: str,
    object_mask_dir: str,
    refined_object_poses_dir: str,
    overlay_path: str,
    refined_object_scale_path: str | None = None,
    left_hand_pose_dir: str | None = None,
    left_hand_mask_dir: str | None = None,
    right_hand_pose_dir: str | None = None,
    right_hand_mask_dir: str | None = None,
    refined_left_hand_pose_dir: str | None = None,
    refined_right_hand_pose_dir: str | None = None,
    depth_dir: str | None = None,
    mano_assets_root: str | None = None,
    n_epochs: int = 30,
    n_gaussian_only_epochs: int = 5,
    batch_size: int = 16,
    lr_gaussians: float = 1e-2,
    lr_hand_gaussians: float | None = None,
    lr_mul_delta_p: float = 1.0,
    lr_mul_quat: float = 1.0,
    lr_mul_scale: float = 1.0,
    lr_mul_opacity: float = 1.0,
    lr_mul_color: float = 1.0,
    lr_mul_obj_global_scale: float = 1.0,
    lr_object_pose: float = 1e-2,
    lr_object_rot: float | None = None,
    lr_object_trans: float | None = None,
    lr_hand_pose: float = 1e-2,
    lr_hand_global_orient: float | None = None,
    lr_hand_finger: float | None = None,
    lr_hand_trans: float | None = None,
    lr_betas: float = 1e-3,
    learn_hand_scale: bool = False,
    lr_hand_scale:    float = 1e-3,
    w_hand_scale_prior: float = 10.0,
    render_every: int = 25,
    progress_dir: str | None = None,
    debug_frame_idx: int | None = None,
    w_photometric: float = 1.0,
    w_silhouette: float = 0.5,
    w_silhouette_obj: float = 1.0,
    w_silhouette_hand: float = 1.0,
    w_depth: float = 0.05,
    w_log_depth_grad: float = 0.0,
    w_photometric_ssim: float = 0.0,
    w_depth_ssim: float = 0.0,
    w_smooth_obj_rot: float = 0.01,
    w_smooth_obj_trans: float = 0.01,
    w_smooth_hand_rot: float = 0.01,
    w_smooth_hand_finger: float = 0.001,
    w_smooth_hand_trans: float = 0.01,
    w_beta_prior: float = 1.0,
    w_delta_p_reg: float = 100.0,
    w_delta_p_reg_obj:  float | None = None,
    w_delta_p_reg_hand: float | None = None,
    w_delta_p_reg_bg:   float | None = None,
    w_obj_scale_prior: float = 1.0,
    mask_background_to_black: bool = False,
    balance_photometric_by_mask: bool = False,
    freeze_object_rot: bool = False,
    freeze_object_trans: bool = False,
    freeze_object_scale: bool = False,
    freeze_hand_rot: bool = False,
    freeze_hand_trans: bool = False,
    with_background: bool = False,
    bg_ref_frame: int | None = None,
    lr_bg_gaussians: float | None = None,
    lr_bg_pose: float = 1e-3,
    lr_bg_rot: float | None = None,
    lr_bg_trans: float | None = None,
    bg_max_points: int = 50000,
    background_pose_init_dir: str | None = None,
    bg_init_stride: int   = 10,
    bg_voxel_size:  float = 0.005,
    w_scale_aniso_bg: float = 0.0,
    w_density_bg:     float = 0.0,
    n_density_neighbors: int = 8,
    density_subsample_frac_bg: float = 0.2,
    w_sdf_density_bg:        float = 0.0,
    w_normal_consistency_bg: float = 0.0,
    n_sdf_samples_bg:        int   = 1000,
    n_sdf_neighbors_bg:      int   = 8,
    valid_mask_threshold:    float = 0.04,
    valid_mask_erode_iters:  int   = 2,
    n_obj_gaussians: int | None = None,
    n_hand_gaussians: int | None = None,
    object_anchor_mode: str = "vertex",
    hand_anchor_mode:   str = "vertex",
    face_normal_thin_factor_obj:  float = 0.25,
    face_normal_thin_factor_hand: float = 0.25,
    w_face_delta_p_tangent_obj:        float = 1.0,
    w_face_delta_p_normal_outward_obj: float = 100.0,
    w_face_delta_p_normal_inward_obj:  float = 0.0,
    w_face_delta_p_tangent_hand:        float = 1.0,
    w_face_delta_p_normal_outward_hand: float = 100.0,
    w_face_delta_p_normal_inward_hand:  float = 0.0,
    use_cosine_lr_schedule: bool = False,
    cosine_lr_min_ratio: float = 0.0,
    coarse_init_scale_factor: float = 1.0,
    coarse_decay_epochs: int | None = None,
    pose_confidence_decay: float = 0.0,
    pose_confidence_ref_frame: int | None = None,
    pose_confidence_dynamic_tau: float = 0.0,
    w_pose_init_prior: float = 0.0,
    rotation_search_n_candidates: int = 0,
    rotation_search_period: int = 0,
    rotation_search_local_frac: float = 0.5,
    rotation_search_local_max_deg: float = 30.0,
    rotation_search_silhouette_weight: float = 1.0,
    rotation_search_smoothness_weight: float = 1.0,
    use_l2_photometric: bool = False,
    use_l2_silhouette: bool = False,
    train_resolution_scale: float = 1.0,
    multiview_include_background: bool = False,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 0,
    resume_from_checkpoint: str | None = None,
    ignore_optimizer_state: bool = False,
    freeze_gaussians: bool = False,
    random_init_obj_pose: bool = False,
    random_init_obj_pose_trans_std: float = 0.1,
    freeze_bg_rot: bool = False,
    freeze_bg_trans: bool = False,
    w_smooth_bg_rot: float = 0.1,
    w_smooth_bg_trans: float = 0.1,
    seed: int = 0,
    dev: bool = False,
) -> None:
    inputs: dict[str, str] = {
        "frames_dir":       frames_dir,
        "intrinsics_path":  intrinsics_path,
        "object_mesh_path": object_mesh_path,
        "object_poses_dir": object_poses_dir,
        "object_mask_dir":  object_mask_dir,
    }
    for k, v in {
        "left_hand_pose_dir":     left_hand_pose_dir,
        "left_hand_mask_dir":     left_hand_mask_dir,
        "right_hand_pose_dir":    right_hand_pose_dir,
        "right_hand_mask_dir":    right_hand_mask_dir,
        "depth_dir":              depth_dir,
        "mano_assets_root":       mano_assets_root,
        "resume_from_checkpoint": resume_from_checkpoint,
        "background_pose_init_dir": background_pose_init_dir,
    }.items():
        if v is not None:
            inputs[k] = v

    outputs: dict[str, str] = {
        "refined_object_poses_dir": refined_object_poses_dir,
        "overlay_path":             overlay_path,
    }
    if refined_object_scale_path is not None:
        outputs["refined_object_scale_path"] = refined_object_scale_path
    for k, v in {
        "refined_left_hand_pose_dir":  refined_left_hand_pose_dir,
        "refined_right_hand_pose_dir": refined_right_hand_pose_dir,
        "progress_dir":                progress_dir,
        "checkpoint_path":             checkpoint_path,
    }.items():
        if v is not None:
            outputs[k] = v

    extra_args: dict[str, object] = {
        "n_epochs":              n_epochs,
        "n_gaussian_only_epochs": n_gaussian_only_epochs,
        "batch_size":            batch_size,
        "lr_gaussians":            lr_gaussians,
        "lr_hand_gaussians":       lr_hand_gaussians,
        "lr_mul_delta_p":          lr_mul_delta_p,
        "lr_mul_quat":             lr_mul_quat,
        "lr_mul_scale":            lr_mul_scale,
        "lr_mul_opacity":          lr_mul_opacity,
        "lr_mul_color":            lr_mul_color,
        "lr_mul_obj_global_scale": lr_mul_obj_global_scale,
        "lr_object_pose":        lr_object_pose,
        "lr_object_rot":         lr_object_rot,
        "lr_object_trans":       lr_object_trans,
        "lr_hand_pose":          lr_hand_pose,
        "lr_hand_global_orient": lr_hand_global_orient,
        "lr_hand_finger":        lr_hand_finger,
        "lr_hand_trans":         lr_hand_trans,
        "lr_betas":          lr_betas,
        "learn_hand_scale":  learn_hand_scale,
        "lr_hand_scale":     lr_hand_scale,
        "w_hand_scale_prior": w_hand_scale_prior,
        "render_every":   render_every,
        "debug_frame_idx": debug_frame_idx,
        "w_photometric":  w_photometric,
        "w_silhouette":      w_silhouette,
        "w_silhouette_obj":  w_silhouette_obj,
        "w_silhouette_hand": w_silhouette_hand,
        "w_depth":           w_depth,
        "w_log_depth_grad":  w_log_depth_grad,
        "w_photometric_ssim": w_photometric_ssim,
        "w_depth_ssim":       w_depth_ssim,
        "w_smooth_obj_rot":    w_smooth_obj_rot,
        "w_smooth_obj_trans":  w_smooth_obj_trans,
        "w_smooth_hand_rot":    w_smooth_hand_rot,
        "w_smooth_hand_finger": w_smooth_hand_finger,
        "w_smooth_hand_trans":  w_smooth_hand_trans,
        "w_beta_prior":        w_beta_prior,
        "w_delta_p_reg":      w_delta_p_reg,
        "w_delta_p_reg_obj":  w_delta_p_reg_obj,
        "w_delta_p_reg_hand": w_delta_p_reg_hand,
        "w_delta_p_reg_bg":   w_delta_p_reg_bg,
        "w_obj_scale_prior": w_obj_scale_prior,
        "mask_background_to_black":    mask_background_to_black,
        "balance_photometric_by_mask": balance_photometric_by_mask,
        "freeze_object_rot":   freeze_object_rot,
        "freeze_object_trans": freeze_object_trans,
        "freeze_object_scale": freeze_object_scale,
        "freeze_hand_rot":     freeze_hand_rot,
        "freeze_hand_trans":   freeze_hand_trans,
        "with_background":     with_background,
        "bg_ref_frame":        bg_ref_frame,
        "lr_bg_gaussians":     lr_bg_gaussians,
        "lr_bg_pose":          lr_bg_pose,
        "lr_bg_rot":           lr_bg_rot,
        "lr_bg_trans":         lr_bg_trans,
        "bg_max_points":       bg_max_points,
        "bg_init_stride":      bg_init_stride,
        "bg_voxel_size":       bg_voxel_size,
        "w_scale_aniso_bg":         w_scale_aniso_bg,
        "w_density_bg":             w_density_bg,
        "n_density_neighbors":      n_density_neighbors,
        "density_subsample_frac_bg": density_subsample_frac_bg,
        "w_sdf_density_bg":         w_sdf_density_bg,
        "w_normal_consistency_bg":  w_normal_consistency_bg,
        "n_sdf_samples_bg":         n_sdf_samples_bg,
        "n_sdf_neighbors_bg":       n_sdf_neighbors_bg,
        "valid_mask_threshold":     valid_mask_threshold,
        "valid_mask_erode_iters":   valid_mask_erode_iters,
        "n_obj_gaussians":         n_obj_gaussians,
        "n_hand_gaussians":        n_hand_gaussians,
        "object_anchor_mode":          object_anchor_mode,
        "hand_anchor_mode":            hand_anchor_mode,
        "face_normal_thin_factor_obj":  face_normal_thin_factor_obj,
        "face_normal_thin_factor_hand": face_normal_thin_factor_hand,
        "w_face_delta_p_tangent_obj":         w_face_delta_p_tangent_obj,
        "w_face_delta_p_normal_outward_obj":  w_face_delta_p_normal_outward_obj,
        "w_face_delta_p_normal_inward_obj":   w_face_delta_p_normal_inward_obj,
        "w_face_delta_p_tangent_hand":        w_face_delta_p_tangent_hand,
        "w_face_delta_p_normal_outward_hand": w_face_delta_p_normal_outward_hand,
        "w_face_delta_p_normal_inward_hand":  w_face_delta_p_normal_inward_hand,
        "use_cosine_lr_schedule":     use_cosine_lr_schedule,
        "cosine_lr_min_ratio":        cosine_lr_min_ratio,
        "coarse_init_scale_factor":   coarse_init_scale_factor,
        "coarse_decay_epochs":        coarse_decay_epochs,
        "pose_confidence_decay":         pose_confidence_decay,
        "pose_confidence_ref_frame":     pose_confidence_ref_frame,
        "pose_confidence_dynamic_tau":   pose_confidence_dynamic_tau,
        "w_pose_init_prior":             w_pose_init_prior,
        "rotation_search_n_candidates":      rotation_search_n_candidates,
        "rotation_search_period":            rotation_search_period,
        "rotation_search_local_frac":        rotation_search_local_frac,
        "rotation_search_local_max_deg":     rotation_search_local_max_deg,
        "rotation_search_silhouette_weight": rotation_search_silhouette_weight,
        "rotation_search_smoothness_weight": rotation_search_smoothness_weight,
        "use_l2_photometric":                use_l2_photometric,
        "use_l2_silhouette":                 use_l2_silhouette,
        "train_resolution_scale":            train_resolution_scale,
        "multiview_include_background":      multiview_include_background,
        "ignore_optimizer_state":            ignore_optimizer_state,
        "freeze_gaussians":                  freeze_gaussians,
        "random_init_obj_pose":              random_init_obj_pose,
        "random_init_obj_pose_trans_std":    random_init_obj_pose_trans_std,
        "freeze_bg_rot":       freeze_bg_rot,
        "freeze_bg_trans":     freeze_bg_trans,
        "w_smooth_bg_rot":     w_smooth_bg_rot,
        "w_smooth_bg_trans":   w_smooth_bg_trans,
        "checkpoint_every":    checkpoint_every,
        "seed":           seed,
    }

    run_in_container(
        image       = IMAGE_NAME,
        module      = "v2d.gsplat_refinement.lib.refine",
        inputs      = inputs,
        outputs     = outputs,
        extra_args  = extra_args,
        dev         = dev,
        modules_dir = MODULES_DIR,
        gpus        = True,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Joint hand+object pose refinement via Gaussian splatting")
    p.add_argument("--frames_dir",                  required=True)
    p.add_argument("--intrinsics_path",             required=True)
    p.add_argument("--object_mesh_path",            required=True)
    p.add_argument("--object_poses_dir",            required=True)
    p.add_argument("--object_mask_dir",             required=True)
    p.add_argument("--refined_object_poses_dir",    required=True)
    p.add_argument("--overlay_path",                required=True)
    p.add_argument("--left_hand_pose_dir",          default=None)
    p.add_argument("--left_hand_mask_dir",          default=None)
    p.add_argument("--right_hand_pose_dir",         default=None)
    p.add_argument("--right_hand_mask_dir",         default=None)
    p.add_argument("--refined_left_hand_pose_dir",  default=None)
    p.add_argument("--refined_right_hand_pose_dir", default=None)
    p.add_argument("--depth_dir",                   default=None)
    p.add_argument("--mano_assets_root",            default=None)
    p.add_argument("--n_epochs",        type=int,   default=30)
    p.add_argument("--n_gaussian_only_epochs", type=int, default=5)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr_gaussians",      type=float, default=1e-2)
    p.add_argument("--lr_hand_gaussians", type=float, default=None)
    p.add_argument("--lr_mul_delta_p",   type=float, default=1.0)
    p.add_argument("--lr_mul_quat",      type=float, default=1.0)
    p.add_argument("--lr_mul_scale",     type=float, default=1.0)
    p.add_argument("--lr_mul_opacity",   type=float, default=1.0)
    p.add_argument("--lr_mul_color",     type=float, default=1.0)
    p.add_argument("--lr_mul_obj_global_scale", type=float, default=1.0)
    p.add_argument("--lr_object_pose",    type=float, default=1e-3)
    p.add_argument("--lr_object_rot",     type=float, default=None)
    p.add_argument("--lr_object_trans",   type=float, default=None)
    p.add_argument("--lr_hand_pose",         type=float, default=1e-3)
    p.add_argument("--lr_hand_global_orient", type=float, default=None)
    p.add_argument("--lr_hand_finger",        type=float, default=None)
    p.add_argument("--lr_hand_trans",         type=float, default=None)
    p.add_argument("--lr_betas",        type=float, default=1e-4)
    p.add_argument("--learn_hand_scale", action="store_true")
    p.add_argument("--lr_hand_scale",     type=float, default=1e-3)
    p.add_argument("--w_hand_scale_prior", type=float, default=10.0)
    p.add_argument("--render_every",    type=int,   default=0)
    p.add_argument("--progress_dir",                default=None)
    p.add_argument("--debug_frame_idx", type=int,   default=None)
    p.add_argument("--w_photometric",   type=float, default=1.0)
    p.add_argument("--w_silhouette",      type=float, default=0.5)
    p.add_argument("--w_silhouette_obj",  type=float, default=1.0)
    p.add_argument("--w_silhouette_hand", type=float, default=1.0)
    p.add_argument("--w_depth",          type=float, default=0.05)
    p.add_argument("--w_log_depth_grad", type=float, default=0.0)
    p.add_argument("--w_photometric_ssim", type=float, default=0.0,
                   help="SSIM photometric loss weight (1 - SSIM, 11x11 "
                        "Gaussian window). 3DGS pairs at ~0.2 with L1=1.0.")
    p.add_argument("--w_depth_ssim",       type=float, default=0.0,
                   help="SSIM depth loss weight (1 - SSIM on log-depth, "
                        "percentile-normalized). Try ~0.1.")
    p.add_argument("--w_smooth_obj_rot",    type=float, default=0.01)
    p.add_argument("--w_smooth_obj_trans",  type=float, default=0.01)
    p.add_argument("--w_smooth_hand_rot",    type=float, default=0.01)
    p.add_argument("--w_smooth_hand_finger", type=float, default=0.001)
    p.add_argument("--w_smooth_hand_trans",  type=float, default=0.01)
    p.add_argument("--w_beta_prior",    type=float, default=10.0)
    p.add_argument("--w_delta_p_reg",      type=float, default=100.0)
    p.add_argument("--w_delta_p_reg_obj",  type=float, default=None)
    p.add_argument("--w_delta_p_reg_hand", type=float, default=None)
    p.add_argument("--w_delta_p_reg_bg",   type=float, default=None)
    p.add_argument("--w_obj_scale_prior", type=float, default=1.0)
    p.add_argument("--mask_background_to_black", action="store_true")
    p.add_argument("--balance_photometric_by_mask", action="store_true")
    p.add_argument("--freeze_object_rot",   action="store_true")
    p.add_argument("--freeze_object_trans", action="store_true")
    p.add_argument("--freeze_object_scale", action="store_true")
    p.add_argument("--freeze_hand_rot",     action="store_true")
    p.add_argument("--freeze_hand_trans",   action="store_true")
    p.add_argument("--with_background",     action="store_true")
    p.add_argument("--bg_ref_frame",        type=int,   default=None)
    p.add_argument("--lr_bg_gaussians",     type=float, default=None)
    p.add_argument("--lr_bg_pose",          type=float, default=1e-3)
    p.add_argument("--lr_bg_rot",           type=float, default=None)
    p.add_argument("--lr_bg_trans",         type=float, default=None)
    p.add_argument("--bg_max_points",       type=int,   default=50000)
    p.add_argument("--background_pose_init_dir", default=None,
                   help="Optional folder of per-frame Transform3d JSONs "
                        "(cam-to-world; DROID/COLMAP convention) used to "
                        "seed the background pose field.")
    p.add_argument("--bg_init_stride", type=int,   default=10,
                   help="Stride for multi-frame BG point-cloud init "
                        "(used only with --background_pose_init_dir).")
    p.add_argument("--bg_voxel_size", type=float, default=0.005,
                   help="Voxel size (m) for BG point-cloud dedup before "
                        "random subsample.")
    p.add_argument("--w_scale_aniso_bg", type=float, default=0.0)
    p.add_argument("--w_density_bg",     type=float, default=0.0)
    p.add_argument("--n_density_neighbors", type=int, default=8)
    p.add_argument("--density_subsample_frac_bg", type=float, default=0.2)
    p.add_argument("--w_sdf_density_bg",        type=float, default=0.0)
    p.add_argument("--w_normal_consistency_bg", type=float, default=0.0)
    p.add_argument("--n_sdf_samples_bg",        type=int,   default=1000)
    p.add_argument("--n_sdf_neighbors_bg",      type=int,   default=8)
    p.add_argument("--valid_mask_threshold",    type=float, default=0.04,
                   help="Max-brightness threshold (in [0,1] image scale) for "
                        "the static valid-pixel mask derived from the input "
                        "video. Pixels whose max brightness across all frames "
                        "is below this are treated as fixed dead/black "
                        "regions (fisheye crop, vignette) and excluded from "
                        "photometric / depth / SuGaR supervision. Set to 0 "
                        "to disable.")
    p.add_argument("--valid_mask_erode_iters",  type=int,   default=2,
                   help="Number of 3x3 erosion passes on the valid-pixel "
                        "mask to peel back the soft boundary transition.")
    p.add_argument("--n_obj_gaussians",     type=int,   default=None)
    p.add_argument("--n_hand_gaussians",    type=int,   default=None)
    p.add_argument("--object_anchor_mode", choices=["vertex", "face"],
                   default="vertex",
                   help="Anchor mode for object Gaussians. 'face' switches to "
                        "one Gaussian per mesh face with Δp in face-local "
                        "(T, B, N) coords. See lib CLI for details.")
    p.add_argument("--hand_anchor_mode", choices=["vertex", "face"],
                   default="vertex",
                   help="Anchor mode for hand Gaussians (same options).")
    p.add_argument("--face_normal_thin_factor_obj",  type=float, default=0.25)
    p.add_argument("--face_normal_thin_factor_hand", type=float, default=0.25)
    p.add_argument("--w_face_delta_p_tangent_obj",        type=float, default=1.0)
    p.add_argument("--w_face_delta_p_normal_outward_obj", type=float, default=100.0)
    p.add_argument("--w_face_delta_p_normal_inward_obj",  type=float, default=0.0)
    p.add_argument("--w_face_delta_p_tangent_hand",        type=float, default=1.0)
    p.add_argument("--w_face_delta_p_normal_outward_hand", type=float, default=100.0)
    p.add_argument("--w_face_delta_p_normal_inward_hand",  type=float, default=0.0)
    p.add_argument("--use_cosine_lr_schedule", action="store_true")
    p.add_argument("--cosine_lr_min_ratio", type=float, default=0.0)
    p.add_argument("--coarse_init_scale_factor", type=float, default=1.0)
    p.add_argument("--coarse_decay_epochs", type=int, default=None)
    p.add_argument("--pose_confidence_decay", type=float, default=0.0)
    p.add_argument("--pose_confidence_ref_frame", type=int, default=None)
    p.add_argument("--pose_confidence_dynamic_tau", type=float, default=0.0)
    p.add_argument("--w_pose_init_prior", type=float, default=0.0)
    p.add_argument("--rotation_search_n_candidates",      type=int,   default=0)
    p.add_argument("--rotation_search_period",            type=int,   default=0)
    p.add_argument("--rotation_search_local_frac",        type=float, default=0.5)
    p.add_argument("--rotation_search_local_max_deg",     type=float, default=30.0)
    p.add_argument("--rotation_search_silhouette_weight", type=float, default=1.0)
    p.add_argument("--rotation_search_smoothness_weight", type=float, default=1.0)
    p.add_argument("--use_l2_photometric", action="store_true")
    p.add_argument("--use_l2_silhouette",  action="store_true")
    p.add_argument("--train_resolution_scale", type=float, default=1.0)
    p.add_argument("--multiview_include_background", action="store_true")
    p.add_argument("--ignore_optimizer_state", action="store_true")
    p.add_argument("--freeze_gaussians",       action="store_true")
    p.add_argument("--random_init_obj_pose",   action="store_true")
    p.add_argument("--random_init_obj_pose_trans_std", type=float, default=0.1)
    p.add_argument("--freeze_bg_rot",       action="store_true")
    p.add_argument("--freeze_bg_trans",     action="store_true")
    p.add_argument("--w_smooth_bg_rot",     type=float, default=0.1)
    p.add_argument("--w_smooth_bg_trans",   type=float, default=0.1)
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--dev", action="store_true")
    args = p.parse_args()
    run_refine(
        frames_dir                  = args.frames_dir,
        intrinsics_path             = args.intrinsics_path,
        object_mesh_path            = args.object_mesh_path,
        object_poses_dir            = args.object_poses_dir,
        object_mask_dir             = args.object_mask_dir,
        refined_object_poses_dir    = args.refined_object_poses_dir,
        overlay_path                = args.overlay_path,
        left_hand_pose_dir          = args.left_hand_pose_dir,
        left_hand_mask_dir          = args.left_hand_mask_dir,
        right_hand_pose_dir         = args.right_hand_pose_dir,
        right_hand_mask_dir         = args.right_hand_mask_dir,
        refined_left_hand_pose_dir  = args.refined_left_hand_pose_dir,
        refined_right_hand_pose_dir = args.refined_right_hand_pose_dir,
        depth_dir                   = args.depth_dir,
        mano_assets_root            = args.mano_assets_root,
        n_epochs                    = args.n_epochs,
        n_gaussian_only_epochs      = args.n_gaussian_only_epochs,
        batch_size                  = args.batch_size,
        lr_gaussians                = args.lr_gaussians,
        lr_hand_gaussians           = args.lr_hand_gaussians,
        lr_mul_delta_p              = args.lr_mul_delta_p,
        lr_mul_quat                 = args.lr_mul_quat,
        lr_mul_scale                = args.lr_mul_scale,
        lr_mul_opacity              = args.lr_mul_opacity,
        lr_mul_color                = args.lr_mul_color,
        lr_mul_obj_global_scale     = args.lr_mul_obj_global_scale,
        lr_object_pose              = args.lr_object_pose,
        lr_object_rot               = args.lr_object_rot,
        lr_object_trans             = args.lr_object_trans,
        lr_hand_global_orient       = args.lr_hand_global_orient,
        lr_hand_finger              = args.lr_hand_finger,
        lr_hand_trans               = args.lr_hand_trans,
        lr_bg_rot                   = args.lr_bg_rot,
        lr_bg_trans                 = args.lr_bg_trans,
        lr_hand_pose                = args.lr_hand_pose,
        lr_betas                    = args.lr_betas,
        learn_hand_scale            = args.learn_hand_scale,
        lr_hand_scale               = args.lr_hand_scale,
        w_hand_scale_prior          = args.w_hand_scale_prior,
        render_every                = args.render_every,
        progress_dir                = args.progress_dir,
        debug_frame_idx             = args.debug_frame_idx,
        w_photometric               = args.w_photometric,
        w_silhouette                = args.w_silhouette,
        w_silhouette_obj            = args.w_silhouette_obj,
        w_silhouette_hand           = args.w_silhouette_hand,
        w_depth                     = args.w_depth,
        w_log_depth_grad            = args.w_log_depth_grad,
        w_photometric_ssim          = args.w_photometric_ssim,
        w_depth_ssim                = args.w_depth_ssim,
        w_smooth_obj_rot            = args.w_smooth_obj_rot,
        w_smooth_obj_trans          = args.w_smooth_obj_trans,
        w_smooth_hand_rot           = args.w_smooth_hand_rot,
        w_smooth_hand_finger        = args.w_smooth_hand_finger,
        w_smooth_hand_trans         = args.w_smooth_hand_trans,
        w_beta_prior                = args.w_beta_prior,
        w_delta_p_reg               = args.w_delta_p_reg,
        w_delta_p_reg_obj           = args.w_delta_p_reg_obj,
        w_delta_p_reg_hand          = args.w_delta_p_reg_hand,
        w_delta_p_reg_bg            = args.w_delta_p_reg_bg,
        w_obj_scale_prior           = args.w_obj_scale_prior,
        mask_background_to_black    = args.mask_background_to_black,
        balance_photometric_by_mask = args.balance_photometric_by_mask,
        freeze_object_rot           = args.freeze_object_rot,
        freeze_object_trans         = args.freeze_object_trans,
        freeze_object_scale         = args.freeze_object_scale,
        freeze_hand_rot             = args.freeze_hand_rot,
        freeze_hand_trans           = args.freeze_hand_trans,
        with_background             = args.with_background,
        bg_ref_frame                = args.bg_ref_frame,
        lr_bg_gaussians             = args.lr_bg_gaussians,
        lr_bg_pose                  = args.lr_bg_pose,
        bg_max_points               = args.bg_max_points,
        background_pose_init_dir    = args.background_pose_init_dir,
        bg_init_stride              = args.bg_init_stride,
        bg_voxel_size               = args.bg_voxel_size,
        w_scale_aniso_bg            = args.w_scale_aniso_bg,
        w_density_bg                = args.w_density_bg,
        n_density_neighbors         = args.n_density_neighbors,
        density_subsample_frac_bg   = args.density_subsample_frac_bg,
        w_sdf_density_bg            = args.w_sdf_density_bg,
        w_normal_consistency_bg     = args.w_normal_consistency_bg,
        n_sdf_samples_bg            = args.n_sdf_samples_bg,
        n_sdf_neighbors_bg          = args.n_sdf_neighbors_bg,
        valid_mask_threshold        = args.valid_mask_threshold,
        valid_mask_erode_iters      = args.valid_mask_erode_iters,
        n_obj_gaussians             = args.n_obj_gaussians,
        n_hand_gaussians            = args.n_hand_gaussians,
        object_anchor_mode           = args.object_anchor_mode,
        hand_anchor_mode             = args.hand_anchor_mode,
        face_normal_thin_factor_obj  = args.face_normal_thin_factor_obj,
        face_normal_thin_factor_hand = args.face_normal_thin_factor_hand,
        w_face_delta_p_tangent_obj         = args.w_face_delta_p_tangent_obj,
        w_face_delta_p_normal_outward_obj  = args.w_face_delta_p_normal_outward_obj,
        w_face_delta_p_normal_inward_obj   = args.w_face_delta_p_normal_inward_obj,
        w_face_delta_p_tangent_hand        = args.w_face_delta_p_tangent_hand,
        w_face_delta_p_normal_outward_hand = args.w_face_delta_p_normal_outward_hand,
        w_face_delta_p_normal_inward_hand  = args.w_face_delta_p_normal_inward_hand,
        use_cosine_lr_schedule      = args.use_cosine_lr_schedule,
        cosine_lr_min_ratio         = args.cosine_lr_min_ratio,
        coarse_init_scale_factor    = args.coarse_init_scale_factor,
        coarse_decay_epochs         = args.coarse_decay_epochs,
        pose_confidence_decay       = args.pose_confidence_decay,
        pose_confidence_ref_frame   = args.pose_confidence_ref_frame,
        pose_confidence_dynamic_tau = args.pose_confidence_dynamic_tau,
        w_pose_init_prior           = args.w_pose_init_prior,
        rotation_search_n_candidates      = args.rotation_search_n_candidates,
        rotation_search_period            = args.rotation_search_period,
        rotation_search_local_frac        = args.rotation_search_local_frac,
        rotation_search_local_max_deg     = args.rotation_search_local_max_deg,
        rotation_search_silhouette_weight = args.rotation_search_silhouette_weight,
        rotation_search_smoothness_weight = args.rotation_search_smoothness_weight,
        use_l2_photometric                = args.use_l2_photometric,
        use_l2_silhouette                 = args.use_l2_silhouette,
        train_resolution_scale            = args.train_resolution_scale,
        multiview_include_background      = args.multiview_include_background,
        ignore_optimizer_state            = args.ignore_optimizer_state,
        freeze_gaussians                  = args.freeze_gaussians,
        random_init_obj_pose              = args.random_init_obj_pose,
        random_init_obj_pose_trans_std    = args.random_init_obj_pose_trans_std,
        freeze_bg_rot               = args.freeze_bg_rot,
        freeze_bg_trans             = args.freeze_bg_trans,
        w_smooth_bg_rot             = args.w_smooth_bg_rot,
        w_smooth_bg_trans           = args.w_smooth_bg_trans,
        seed                        = args.seed,
        dev                         = args.dev,
    )
