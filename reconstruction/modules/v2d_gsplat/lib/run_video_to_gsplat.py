"""
Entry point for v2d_gsplat: video → 4D Gaussian Splatting reconstruction.

Inputs (all file-based, matching Docker wrapper convention):
  video_path          - input RGB video
  depth_folder        - pre-computed depth PNGs (one per frame, uint16)
  intrinsics_path     - CameraIntrinsics JSON (frame-0 reference)
  masks_dir           - SAM2 mask directory ({object_id}/{frame:06d}.png)
  prompts_path        - SAM2 prompts JSON (provides role per object_id)
  output_dir          - root output directory
  weights_dir         - model weights; expects smpl/ subdir with SMPL_NEUTRAL.pkl etc.

Optional inputs:
  smpl_path           - depth-aligned NlfResult NPZ (from run_align_nlf_to_depth)
  object_meshes_dir   - directory with object_{id}.obj files (from SAM3D)

Optional parameters:
  camera_mode         - "static" (default) or "joint"
  num_frames          - cap the number of video frames used
  sh_degree           - spherical harmonics degree (default 3)
  n_cycles / iterations_canonical_per_cycle / iterations_pose_per_cycle / iterations_refine

Outputs written to output_dir/:
  gaussians.ply          - optimised Gaussian scene
  entities.json          - entity metadata
  smpl/smpl_refined.npz  - refined SMPL parameters (if SMPL was used)
  renders/comparison.mp4 - original vs. rendered side-by-side video
  renders/*.png          - checkpoint renders during optimisation
"""

import os
import json
import argparse

import numpy as np
import torch

from v2d.common.datatypes import CameraIntrinsics
from v2d.gsplat.lib.scene import GaussianScene
from v2d.gsplat.lib.deformation import SmplDeformer, BodyPoseParams, ObjectPoseParams
from v2d.gsplat.lib.initialization import build_scene
from v2d.gsplat.lib.optimization import run_optimization, OptimConfig
from v2d.gsplat.lib.extraction import (
    save_gaussians_ply, save_entities_json, save_smpl_results,
    save_object_poses, save_renders_video,
    save_scene_checkpoint, save_entity_renders, save_orbit_renders,
)


def _parse_entity_role_map(prompts_path: str) -> dict:
    """
    Read SAM2 prompts JSON and return {object_id: role}.
    Roles in prompts JSON: "human" → body entity, "object" → rigid object.
    Anything else defaults to "object".
    """
    with open(prompts_path) as f:
        data = json.load(f)
    role_map = {}
    for p in data.get('prompts', []):
        oid = int(p['object_id'])
        role = p.get('role', 'object').lower()
        if role not in ('human', 'object'):
            role = 'object'
        role_map[oid] = role
    return role_map


def _get_num_frames(video_path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _get_object_asset_paths(object_meshes_dir: str, entity_role_map: dict) -> tuple:
    """
    Scan object_meshes_dir for object_{id}.obj, object_{id}_transform.json, and
    object_{id}_fp_poses/ directories.
    Returns ({object_id: mesh_path}, {object_id: transform_path}, {object_id: fp_poses_dir}).
    """
    if not object_meshes_dir or not os.path.isdir(object_meshes_dir):
        return {}, {}, {}
    mesh_paths = {}
    transform_paths = {}
    fp_poses_dirs = {}
    for oid, role in entity_role_map.items():
        if role != 'object':
            continue
        mesh_cand = os.path.join(object_meshes_dir, f"object_{oid}.obj")
        xform_cand = os.path.join(object_meshes_dir, f"object_{oid}_transform.json")
        poses_cand = os.path.join(object_meshes_dir, f"object_{oid}_fp_poses")
        if os.path.exists(mesh_cand):
            mesh_paths[oid] = mesh_cand
        if os.path.exists(xform_cand):
            transform_paths[oid] = xform_cand
            print(f"[gsplat] Found SAM3D transform for object {oid}: {xform_cand}")
        if os.path.isdir(poses_cand):
            fp_poses_dirs[oid] = poses_cand
            n = len([f for f in os.listdir(poses_cand) if f.endswith('.json')])
            print(f"[gsplat] Found FP poses for object {oid}: {n} frames at {poses_cand}")
    return mesh_paths, transform_paths, fp_poses_dirs


def video_to_gsplat(
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
    max_gaussians: int = 500_000,
    prune_opacity_threshold: float = 0.005,
    grad_threshold: float = 0.0002,
    densify_every: int = 100,
    max_scale_factor: float = 0.1,
    reset_opacity_every: int = 500,
    device: str = 'cuda',
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Parse inputs
    # ------------------------------------------------------------------ #
    intrinsics = CameraIntrinsics.load(intrinsics_path)
    entity_role_map = _parse_entity_role_map(prompts_path)
    total_frames = _get_num_frames(video_path)
    if num_frames is not None:
        total_frames = min(num_frames, total_frames)
    frame_indices = list(range(0, total_frames, max(1, frame_step)))
    all_frame_indices = list(range(total_frames))

    object_mesh_paths, object_transform_paths, object_fp_poses_dirs = _get_object_asset_paths(object_meshes_dir, entity_role_map)

    print(f"[gsplat] video={os.path.basename(video_path)}  frames={total_frames}  step={frame_step}  sampled={len(frame_indices)}")
    print(f"[gsplat] entities: {entity_role_map}")
    print(f"[gsplat] object meshes:     {list(object_mesh_paths.keys())}")
    print(f"[gsplat] object transforms: {list(object_transform_paths.keys())}")
    print(f"[gsplat] object FP poses:   {list(object_fp_poses_dirs.keys())}")

    # ------------------------------------------------------------------ #
    # 2. Body model setup
    # ------------------------------------------------------------------ #
    smpl_deformer = None
    body_pose_params = None
    smpl_model_type = 'smpl'
    smpl_gender = 'neutral'

    has_human = any(r == 'human' for r in entity_role_map.values())
    # smplx.create(model_path, model_type='smpl') looks for {model_path}/smpl/*.pkl
    # so pass weights_dir directly (not weights_dir/smpl)
    smpl_model_dir = weights_dir

    if has_human and os.path.isdir(smpl_model_dir):
        # Detect model type from result file (HDF5 or NPZ)
        if smpl_path and os.path.exists(smpl_path):
            import h5py, numpy as np
            if h5py.is_hdf5(smpl_path):
                with h5py.File(smpl_path, 'r') as f:
                    _decode = lambda v: v.decode('utf-8') if isinstance(v, bytes) else str(v)
                    smpl_model_type = _decode(f['model_type'][()])
                    smpl_gender     = _decode(f['gender'][()])
            else:
                meta = np.load(smpl_path, allow_pickle=True)
                smpl_model_type = str(meta.get('model_type', 'smpl'))
                smpl_gender     = str(meta.get('gender', 'neutral'))

        try:
            smpl_deformer = SmplDeformer(
                smpl_model_dir,
                gender=smpl_gender,
                model_type=smpl_model_type,
                device=device,
            )
            print(f"[gsplat] SMPL deformer loaded ({smpl_model_type}, {smpl_gender})")
        except Exception as e:
            print(f"[gsplat] WARNING: Could not load SMPL model: {e}")
            smpl_deformer = None
    elif has_human:
        print(f"[gsplat] WARNING: No SMPL model found at {smpl_model_dir} — body entity will have static Gaussians")

    # ------------------------------------------------------------------ #
    # 3. Pose params
    # ------------------------------------------------------------------ #
    if smpl_deformer is not None:
        n_body_joints = smpl_deformer.body_model.NUM_BODY_JOINTS
        body_pose_params = BodyPoseParams(total_frames, n_body_joints, device=device)
        if smpl_path and os.path.exists(smpl_path):
            body_pose_params.load_from_npz(smpl_path)
            print(f"[gsplat] Initialised body pose from {smpl_path}")
        body_pose_params = body_pose_params.to(device)

    # Build ordered list of object oids (matches rid indexing in ObjectPoseParams)
    object_oids = sorted(oid for oid, r in entity_role_map.items() if r == 'object')
    n_objects = len(object_oids)
    obj_pose_params = None
    if n_objects > 0:
        obj_pose_params = ObjectPoseParams(total_frames, n_objects, device=device)
        for rid, oid in enumerate(object_oids):
            poses_dir = object_fp_poses_dirs.get(oid)
            transform_path = object_transform_paths.get(oid)
            if poses_dir and transform_path:
                obj_pose_params.load_from_fp_poses_dir(poses_dir, rid, transform_path)
            elif poses_dir:
                print(f"[gsplat] WARNING: FP poses found for object {oid} but no frame-0 transform — skipping pose init")

    # ------------------------------------------------------------------ #
    # 4. Scene initialisation
    # ------------------------------------------------------------------ #
    print("[gsplat] Initialising scene…")
    smpl_betas = body_pose_params.betas.detach() if body_pose_params is not None else None

    scene = build_scene(
        video_path=video_path,
        depth_folder=depth_folder,
        intrinsics=intrinsics,
        masks_dir=masks_dir,
        entity_role_map=entity_role_map,
        smpl_deformer=smpl_deformer,
        smpl_betas=smpl_betas,
        object_mesh_paths=object_mesh_paths,
        object_transform_paths=object_transform_paths,
        initial_opacity_obj=initial_opacity_obj,
        body_subdivisions=body_subdivisions,
        device=device,
    ).to(device)

    # ------------------------------------------------------------------ #
    # 4b. Debug: dump initial body Gaussian positions vs NLF vertices
    # ------------------------------------------------------------------ #
    if body_pose_params is not None and smpl_deformer is not None:
        _debug_dump_body_init(
            video_path=video_path,
            intrinsics=intrinsics,
            scene=scene,
            body_pose_params=body_pose_params,
            smpl_deformer=smpl_deformer,
            output_dir=output_dir,
            frame_t=0,
            device=device,
        )

    # ------------------------------------------------------------------ #
    # 5. Optimise
    # ------------------------------------------------------------------ #
    loss_weights = OptimConfig.__dataclass_fields__['loss_weights'].default_factory()
    loss_weights['entity_mask'] = weight_entity_mask
    loss_weights['depth'] = weight_depth
    cfg = OptimConfig(
        alternating=alternating,
        n_cycles=n_cycles,
        iterations_canonical_per_cycle=iterations_canonical_per_cycle,
        iterations_pose_per_cycle=iterations_pose_per_cycle,
        iterations_refine=iterations_refine,
        n_pose_sweep_passes=n_pose_sweep_passes,
        max_gaussians=max_gaussians,
        prune_opacity_threshold=prune_opacity_threshold,
        grad_threshold=grad_threshold,
        densify_every=densify_every,
        max_scale_factor=max_scale_factor,
        reset_opacity_every=reset_opacity_every,
        sh_degree=sh_degree,
        train_scale=train_scale,
        entity_mask_interval=entity_mask_interval,
        loss_weights=loss_weights,
        lr_scale=lr_scale,
        lr_obj_pose=lr_obj_pose,
        lr_body_joints=lr_body_joints,
        batch_size=batch_size,
        body_mask_outside_weight=body_mask_outside_weight,
        weight_obj_pose_smooth=weight_obj_pose_smooth,
        weight_body_pose_smooth=weight_body_pose_smooth,
        weight_body_anchor=weight_body_anchor,
        weight_obj_anchor=weight_obj_anchor,
        lr_exposure=lr_exposure,
        weight_exposure_reg=weight_exposure_reg,
        weight_isotropy=weight_isotropy,
        device=device,
    )

    scene = run_optimization(
        scene=scene,
        video_path=video_path,
        depth_folder=depth_folder,
        intrinsics=intrinsics,
        masks_dir=masks_dir,
        entity_role_map=entity_role_map,
        frame_indices=frame_indices,
        all_frame_indices=all_frame_indices,
        body_pose_params=body_pose_params,
        obj_pose_params=obj_pose_params,
        smpl_deformer=smpl_deformer,
        cfg=cfg,
        output_dir=output_dir,
        total_frames=total_frames,
    )

    # ------------------------------------------------------------------ #
    # 6. Save outputs
    # ------------------------------------------------------------------ #
    print("[gsplat] Saving outputs…")
    save_gaussians_ply(scene, os.path.join(output_dir, 'gaussians.ply'))
    save_entities_json(entity_role_map, object_mesh_paths, os.path.join(output_dir, 'entities.json'))

    if body_pose_params is not None:
        save_smpl_results(
            body_pose_params,
            os.path.join(output_dir, 'poses'),
            model_type=smpl_model_type,
            gender=smpl_gender,
        )

    if obj_pose_params is not None:
        save_object_poses(obj_pose_params, os.path.join(output_dir, 'poses'))

    save_renders_video(
        scene=scene,
        video_path=video_path,
        depth_folder=depth_folder,
        intrinsics=intrinsics,
        body_pose_params=body_pose_params,
        obj_pose_params=obj_pose_params,
        smpl_deformer=smpl_deformer,
        frame_indices=all_frame_indices,
        output_path=os.path.join(output_dir, 'renders', 'comparison.mp4'),
        sh_degree=sh_degree,
        device=device,
    )

    save_scene_checkpoint(
        scene, body_pose_params, obj_pose_params,
        output_dir=output_dir,
    )

    save_entity_renders(
        scene=scene,
        body_pose_params=body_pose_params,
        obj_pose_params=obj_pose_params,
        smpl_deformer=smpl_deformer,
        intrinsics=intrinsics,
        frame_indices=all_frame_indices,
        output_dir=os.path.join(output_dir, 'debug'),
        device=device,
    )

    save_orbit_renders(
        scene=scene,
        body_pose_params=body_pose_params,
        obj_pose_params=obj_pose_params,
        smpl_deformer=smpl_deformer,
        intrinsics=intrinsics,
        frame_t=all_frame_indices[len(all_frame_indices) // 2],  # middle frame
        output_dir=os.path.join(output_dir, 'debug', 'orbit'),
        device=device,
    )

    print(f"[gsplat] Done. Outputs at: {output_dir}")


def _debug_dump_body_init(
    video_path: str,
    intrinsics: CameraIntrinsics,
    scene,
    body_pose_params,
    smpl_deformer,
    output_dir: str,
    frame_t: int = 0,
    device: str = 'cuda',
) -> None:
    """
    Save a debug image showing, for frame_t:
      - Left:  frame RGB with NLF posed SMPL vertices projected (green dots)
      - Right: frame RGB with initial body Gaussian world positions projected (red dots)
    Saved to {output_dir}/debug/body_init_debug_frame{frame_t:06d}.png
    """
    import cv2
    from PIL import Image
    from v2d.gsplat.lib.optimization import compute_world_positions
    from v2d.gsplat.lib.scene import ENTITY_BODY

    os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)

    # Load frame RGB
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_t)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  [debug] Could not read frame {frame_t}")
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
    H, W = rgb.shape[:2]

    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    def project(pts_3d):
        """Project (N, 3) world points to (N, 2) pixel coords."""
        x = pts_3d[:, 0] / pts_3d[:, 2].clamp(min=1e-3)
        y = pts_3d[:, 1] / pts_3d[:, 2].clamp(min=1e-3)
        u = (x * fx + cx).long()
        v = (y * fy + cy).long()
        return u.cpu().numpy(), v.cpu().numpy()

    def draw_dots(img, us, vs, color, radius=2):
        out = img.copy()
        for u, v in zip(us, vs):
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(out, (int(u), int(v)), radius, color, -1)
        return out

    with torch.no_grad():
        go, bp, betas, transl = body_pose_params.frame(frame_t)

        # --- NLF posed vertices (green) ---
        nlf_verts = smpl_deformer.get_posed_vertices(go, bp, betas, transl)  # (1, V, 3)
        nlf_verts = nlf_verts.squeeze(0)  # (V, 3)
        # Subsample for speed
        step = max(1, nlf_verts.shape[0] // 500)
        nlf_sub = nlf_verts[::step]
        u_nlf, v_nlf = project(nlf_sub)

        # --- Body Gaussian world positions (red) ---
        world_pos = compute_world_positions(scene, body_pose_params, None, smpl_deformer, frame_t)
        body_mask = scene.body_mask()
        body_world = world_pos[body_mask]  # (N_body, 3)
        step2 = max(1, body_world.shape[0] // 500)
        body_sub = body_world[::step2]
        u_body, v_body = project(body_sub)

    left  = draw_dots(rgb, u_nlf,  v_nlf,  color=(0, 200, 0),   radius=3)   # green = NLF
    right = draw_dots(rgb, u_body, v_body, color=(200, 0, 0),   radius=3)   # red   = Gaussians

    # Annotate
    cv2.putText(left,  'NLF vertices (green)',        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0),   2)
    cv2.putText(right, 'Body Gaussians (red)',         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0),   2)
    cv2.putText(left,  f'frame {frame_t}',            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(right, f'frame {frame_t}',            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    combined = np.concatenate([left, right], axis=1)
    out_path = os.path.join(output_dir, 'debug', f'body_init_debug_frame{frame_t:06d}.png')
    Image.fromarray(combined).save(out_path)
    print(f"  [debug] Body init dump saved to {out_path}")
    print(f"  [debug] NLF transl frame {frame_t}: {transl.squeeze().tolist()}")
    print(f"  [debug] NLF body_pose norm frame {frame_t}: {bp.norm().item():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4D Gaussian Splatting reconstruction')
    parser.add_argument('--video_path',        required=True)
    parser.add_argument('--depth_folder',      required=True)
    parser.add_argument('--intrinsics_path',   required=True)
    parser.add_argument('--masks_dir',         required=True)
    parser.add_argument('--prompts_path',      required=True)
    parser.add_argument('--output_dir',        required=True)
    parser.add_argument('--weights_dir',       required=True)
    parser.add_argument('--smpl_path',         default=None)
    parser.add_argument('--object_meshes_dir', default=None)
    parser.add_argument('--camera_mode',       default='static')
    parser.add_argument('--num_frames',        type=int, default=None)
    parser.add_argument('--frame_step',        type=int, default=1)
    parser.add_argument('--sh_degree',                      type=int,   default=3)
    parser.add_argument('--alternating',                    action='store_true', default=True)
    parser.add_argument('--no_alternating',                 dest='alternating', action='store_false')
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
    parser.add_argument('--max_gaussians',              type=int,   default=500_000)
    parser.add_argument('--prune_opacity_threshold',    type=float, default=0.005)
    parser.add_argument('--grad_threshold',             type=float, default=0.0002)
    parser.add_argument('--densify_every',              type=int,   default=100)
    parser.add_argument('--max_scale_factor',           type=float, default=0.1)
    parser.add_argument('--reset_opacity_every',        type=int,   default=500)
    args = parser.parse_args()

    video_to_gsplat(
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
        frame_step=args.frame_step,
        sh_degree=args.sh_degree,
        alternating=args.alternating,
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
        max_gaussians=args.max_gaussians,
        prune_opacity_threshold=args.prune_opacity_threshold,
        grad_threshold=args.grad_threshold,
        densify_every=args.densify_every,
        max_scale_factor=args.max_scale_factor,
        reset_opacity_every=args.reset_opacity_every,
    )
