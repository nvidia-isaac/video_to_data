from v2d.docker.container import run_in_container
from v2d.gsplat.docker._config import IMAGE_NAME, MODULES_DIR


def run_video_to_gsplat(
    video_path: str,
    depth_folder: str,
    intrinsics_path: str,
    masks_dir: str,
    prompts_path: str,
    output_dir: str,
    weights_dir: str,
    smpl_path: str = None,
    object_meshes_dir: str = None,
    camera_mode: str = 'static',
    num_frames: int = None,
    frame_step: int = 1,
    sh_degree: int = 3,
    alternating: bool = True,
    n_cycles: int = 3,
    iterations_canonical_per_cycle: int = 1000,
    iterations_pose_per_cycle: int = 500,
    iterations_refine: int = 1000,
    n_pose_sweep_passes: int = 1,
    train_scale: float = 0.5,
    entity_mask_interval: int = 5,
    weight_entity_mask: float = 1.0,
    weight_depth: float = 0.1,
    lr_scale: float = 1.0,
    lr_obj_pose: float = 1e-3,
    lr_obj_scale: float = 1e-4,
    weight_obj_scale_reg: float = 0.1,
    lr_decay_schedule: str = 'cosine',
    lr_decay_final: float = 0.1,
    lr_body_joints: float = 0.0,
    batch_size: int = 4,
    initial_opacity_obj: float = 0.05,
    body_subdivisions: int = 0,
    body_mask_outside_weight: float = 0.5,
    weight_obj_pose_smooth: float = 0.0,
    weight_body_pose_smooth: float = 0.0,
    weight_body_anchor: float = 0.0,
    weight_obj_anchor: float = 0.0,
    lr_exposure: float = 1e-2,
    weight_exposure_reg: float = 0.1,
    weight_isotropy: float = 0.0,
    hard_mining_beta: float = 0.0,
    hard_mining_eps: float = 0.1,
    frame_sampling: str = 'hard_negative',
    config_diversity_temperature: float = 1.0,
    min_obj_confidence: float = 0.1,
    weight_obj_slerp_anchor: float = 1.0,
    max_gaussians: int = 500_000,
    prune_opacity_threshold: float = 0.005,
    grad_threshold: float = 0.0002,
    densify_every: int = 100,
    max_scale_factor: float = 0.1,
    reset_opacity_every: int = 500,
    dev: bool = False,
) -> None:
    """
    Run 4D Gaussian Splatting reconstruction inside a Docker container.

    Required inputs:
      video_path        - source video
      depth_folder      - depth PNGs from a monocular depth module (e.g. v2d_moge)
      intrinsics_path   - CameraIntrinsics JSON (frame-0)
      masks_dir         - SAM2 mask directory ({object_id}/{frame:06d}.png)
      prompts_path      - SAM2 prompts JSON (object_id → role mapping)
      output_dir        - results written here
      weights_dir       - model weights; must contain smpl/ subdir with SMPL .pkl files

    Optional inputs:
      smpl_path         - depth-aligned NlfResult NPZ (from run_align_nlf_to_depth)
      object_meshes_dir - directory containing object_{id}.obj files (from run_image_to_mesh)

    Optional parameters:
      camera_mode       - "static" (default) or "joint"
      num_frames        - limit frames processed (useful for debugging)
      sh_degree         - spherical harmonics degree (default 3)
      iterations_phase1/2/3 - per-phase iteration counts
    """
    inputs = {
        'video_path': video_path,
        'depth_folder': depth_folder,
        'intrinsics_path': intrinsics_path,
        'masks_dir': masks_dir,
        'prompts_path': prompts_path,
        'weights_dir': weights_dir,
    }
    if smpl_path is not None:
        inputs['smpl_path'] = smpl_path
    if object_meshes_dir is not None:
        inputs['object_meshes_dir'] = object_meshes_dir

    run_in_container(
        image=IMAGE_NAME,
        module='v2d.gsplat.lib.run_video_to_gsplat',
        inputs=inputs,
        outputs={'output_dir': output_dir},
        extra_args={
            'camera_mode': camera_mode,
            'num_frames': num_frames,
            'frame_step': frame_step,
            'sh_degree': sh_degree,
            'alternating': alternating,
            'n_cycles': n_cycles,
            'iterations_canonical_per_cycle': iterations_canonical_per_cycle,
            'iterations_pose_per_cycle': iterations_pose_per_cycle,
            'iterations_refine': iterations_refine,
            'n_pose_sweep_passes': n_pose_sweep_passes,
            'train_scale': train_scale,
            'entity_mask_interval': entity_mask_interval,
            'weight_entity_mask': weight_entity_mask,
            'weight_depth': weight_depth,
            'lr_scale': lr_scale,
            'lr_obj_pose': lr_obj_pose,
            'lr_obj_scale': lr_obj_scale,
            'weight_obj_scale_reg': weight_obj_scale_reg,
            'lr_decay_schedule': lr_decay_schedule,
            'lr_decay_final': lr_decay_final,
            'lr_body_joints': lr_body_joints,
            'batch_size': batch_size,
            'initial_opacity_obj': initial_opacity_obj,
            'body_subdivisions': body_subdivisions,
            'body_mask_outside_weight': body_mask_outside_weight,
            'weight_obj_pose_smooth': weight_obj_pose_smooth,
            'weight_body_pose_smooth': weight_body_pose_smooth,
            'weight_body_anchor': weight_body_anchor,
            'weight_obj_anchor': weight_obj_anchor,
            'lr_exposure': lr_exposure,
            'weight_exposure_reg': weight_exposure_reg,
            'weight_isotropy': weight_isotropy,
            'hard_mining_beta': hard_mining_beta,
            'hard_mining_eps': hard_mining_eps,
            'frame_sampling': frame_sampling,
            'config_diversity_temperature': config_diversity_temperature,
            'min_obj_confidence': min_obj_confidence,
            'weight_obj_slerp_anchor': weight_obj_slerp_anchor,
            'max_gaussians': max_gaussians,
            'prune_opacity_threshold': prune_opacity_threshold,
            'grad_threshold': grad_threshold,
            'densify_every': densify_every,
            'max_scale_factor': max_scale_factor,
            'reset_opacity_every': reset_opacity_every,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='4D Gaussian Splatting reconstruction (Docker wrapper)')
    parser.add_argument('--video_path',          required=True)
    parser.add_argument('--depth_folder',        required=True)
    parser.add_argument('--intrinsics_path',     required=True)
    parser.add_argument('--masks_dir',           required=True)
    parser.add_argument('--prompts_path',        required=True)
    parser.add_argument('--output_dir',          required=True)
    parser.add_argument('--weights_dir',         required=True)
    parser.add_argument('--smpl_path',           default=None)
    parser.add_argument('--object_meshes_dir',   default=None)
    parser.add_argument('--camera_mode',         default='static')
    parser.add_argument('--num_frames',          type=int, default=None)
    parser.add_argument('--frame_step',          type=int, default=1)
    parser.add_argument('--sh_degree',                      type=int,   default=3)
    parser.add_argument('--alternating',    action='store_true',  default=True)
    parser.add_argument('--no_alternating', dest='alternating', action='store_false')
    parser.add_argument('--n_cycles',                       type=int,   default=3)
    parser.add_argument('--iterations_canonical_per_cycle', type=int,   default=1000)
    parser.add_argument('--iterations_pose_per_cycle',      type=int,   default=500)
    parser.add_argument('--iterations_refine',              type=int,   default=1000)
    parser.add_argument('--n_pose_sweep_passes',            type=int,   default=1)
    parser.add_argument('--train_scale',                    type=float, default=0.5)
    parser.add_argument('--entity_mask_interval', type=int,   default=5)
    parser.add_argument('--weight_entity_mask',   type=float, default=1.0)
    parser.add_argument('--weight_depth',         type=float, default=0.1)
    parser.add_argument('--lr_scale',             type=float, default=1.0)
    parser.add_argument('--lr_obj_pose',          type=float, default=1e-3)
    parser.add_argument('--lr_obj_scale',         type=float, default=1e-4)
    parser.add_argument('--weight_obj_scale_reg', type=float, default=0.1)
    parser.add_argument('--lr_decay_schedule',    type=str,   default='cosine',
                        choices=['none', 'cosine', 'exponential'])
    parser.add_argument('--lr_decay_final',       type=float, default=0.1)
    parser.add_argument('--lr_body_joints',       type=float, default=0.0)
    parser.add_argument('--batch_size',           type=int,   default=4)
    parser.add_argument('--initial_opacity_obj',       type=float, default=0.05)
    parser.add_argument('--body_subdivisions',         type=int,   default=0)
    parser.add_argument('--body_mask_outside_weight',  type=float, default=0.5)
    parser.add_argument('--weight_obj_pose_smooth',    type=float, default=0.0)
    parser.add_argument('--weight_body_pose_smooth', type=float, default=0.0)
    parser.add_argument('--weight_body_anchor',     type=float, default=0.0)
    parser.add_argument('--weight_obj_anchor',      type=float, default=0.0)
    parser.add_argument('--lr_exposure',            type=float, default=1e-2)
    parser.add_argument('--weight_exposure_reg',    type=float, default=0.1)
    parser.add_argument('--weight_isotropy',        type=float, default=0.0)
    parser.add_argument('--hard_mining_beta',       type=float, default=0.0)
    parser.add_argument('--hard_mining_eps',        type=float, default=0.1)
    parser.add_argument('--frame_sampling',         type=str,   default='hard_negative',
                        choices=['uniform', 'hard_negative', 'config_diversity'])
    parser.add_argument('--config_diversity_temperature', type=float, default=1.0)
    parser.add_argument('--min_obj_confidence',           type=float, default=0.1)
    parser.add_argument('--weight_obj_slerp_anchor',      type=float, default=1.0)
    parser.add_argument('--max_gaussians',              type=int,   default=500_000)
    parser.add_argument('--prune_opacity_threshold',    type=float, default=0.005)
    parser.add_argument('--grad_threshold',             type=float, default=0.0002)
    parser.add_argument('--densify_every',              type=int,   default=100)
    parser.add_argument('--max_scale_factor',           type=float, default=0.1)
    parser.add_argument('--reset_opacity_every',        type=int,   default=500)
    parser.add_argument('--dev',                    action='store_true')
    args = parser.parse_args()

    run_video_to_gsplat(
        video_path=args.video_path,
        depth_folder=args.depth_folder,
        intrinsics_path=args.intrinsics_path,
        masks_dir=args.masks_dir,
        prompts_path=args.prompts_path,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        smpl_path=args.smpl_path,
        object_meshes_dir=args.object_meshes_dir,
        camera_mode=args.camera_mode,
        num_frames=args.num_frames,
        sh_degree=args.sh_degree,
        n_cycles=args.n_cycles,
        iterations_canonical_per_cycle=args.iterations_canonical_per_cycle,
        iterations_pose_per_cycle=args.iterations_pose_per_cycle,
        iterations_refine=args.iterations_refine,
        n_pose_sweep_passes=args.n_pose_sweep_passes,
        train_scale=args.train_scale,
        entity_mask_interval=args.entity_mask_interval,
        weight_entity_mask=args.weight_entity_mask,
        weight_depth=args.weight_depth,
        lr_scale=args.lr_scale,
        lr_obj_pose=args.lr_obj_pose,
        lr_obj_scale=args.lr_obj_scale,
        weight_obj_scale_reg=args.weight_obj_scale_reg,
        lr_decay_schedule=args.lr_decay_schedule,
        lr_decay_final=args.lr_decay_final,
        lr_body_joints=args.lr_body_joints,
        batch_size=args.batch_size,
        initial_opacity_obj=args.initial_opacity_obj,
        body_subdivisions=args.body_subdivisions,
        body_mask_outside_weight=args.body_mask_outside_weight,
        weight_obj_pose_smooth=args.weight_obj_pose_smooth,
        weight_body_pose_smooth=args.weight_body_pose_smooth,
        weight_body_anchor=args.weight_body_anchor,
        weight_obj_anchor=args.weight_obj_anchor,
        lr_exposure=args.lr_exposure,
        weight_exposure_reg=args.weight_exposure_reg,
        weight_isotropy=args.weight_isotropy,
        hard_mining_beta=args.hard_mining_beta,
        hard_mining_eps=args.hard_mining_eps,
        frame_sampling=args.frame_sampling,
        config_diversity_temperature=args.config_diversity_temperature,
        min_obj_confidence=args.min_obj_confidence,
        weight_obj_slerp_anchor=args.weight_obj_slerp_anchor,
        max_gaussians=args.max_gaussians,
        prune_opacity_threshold=args.prune_opacity_threshold,
        grad_threshold=args.grad_threshold,
        densify_every=args.densify_every,
        max_scale_factor=args.max_scale_factor,
        reset_opacity_every=args.reset_opacity_every,
        dev=args.dev,
    )
