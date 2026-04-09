"""
End-to-end Gaussian Splatting pipeline using v2d_sam2 test assets.

Prerequisites (weights must be downloaded before running):
  data/weights/moge/          - MoGe model weights
  data/weights/sam2/          - SAM2 model weights
  data/weights/sam3d/         - SAM3D model weights
  data/weights/nlf/           - NLF model weights
  data/weights/smpl/          - SMPL model files (SMPL_NEUTRAL.pkl etc.)
                                Available from https://smpl.is.tue.mpg.de/
                                (Same files as used by v2d_nlf; place/symlink here)

Test assets used:
  modules/v2d_sam2/assets/test_video.mp4      - video with person + object
  modules/v2d_sam2/assets/test_prompts.json   - bounding boxes for person (id=0) and object (id=1)

Run from reconstruction/:
  python experiments/run_gsplat_example.py

To test with a quick run (few frames, fewer iterations):
  python experiments/run_gsplat_example.py --quick
"""

import os
import argparse

from v2d.common.utils import extract_images
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses as run_fp_video_to_poses
from v2d.nlf.docker.run_video_to_smpl import run_video_to_smpl
from v2d.nlf.docker.run_align_nlf_to_depth import run_align_nlf_to_depth
from v2d.gsplat.docker.run_video_to_gsplat import run_video_to_gsplat


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
VIDEO   = 'modules/v2d_sam2/assets/test_video.mp4'
PROMPTS = 'modules/v2d_sam2/assets/test_prompts.json'
WEIGHTS     = 'data/weights'
WEIGHTS_NLF = 'data/weights/nlf'   # gsplat expects smpl/ subdir here
OUT     = 'data/outputs/gsplat_example'

# Intermediate outputs
FRAMES_DIR     = f'{OUT}/frames'
DEPTH_DIR      = f'{OUT}/depth'
INTRINSICS_DIR = f'{OUT}/intrinsics'
MASKS_DIR      = f'{OUT}/masks'          # SAM2 writes: masks/{object_id}/{frame:06d}.png
OBJECTS_DIR    = f'{OUT}/objects'
SMPL_RAW       = f'{OUT}/smpl/smpl_raw.npz'
SMPL_ALIGNED   = f'{OUT}/smpl/smpl_aligned.npz'
GSPLAT_DIR     = f'{OUT}/gsplat'

# SAM2 object IDs from test_prompts.json:
#   object_id=0 → role="human"  → human body
#   object_id=1 → role="object" → rigid object (the hand-held item)
HUMAN_OBJ_ID  = 0
OBJECT_OBJ_ID = 1


def run_prerequisites(dev: bool = False, quick: bool = False):
    """
    Run all upstream modules to produce the inputs needed by v2d_gsplat.
    Each step writes to disk; re-running is idempotent (files are overwritten).
    """

    # ------------------------------------------------------------------ #
    # Step 1: Depth + intrinsics (MoGe)
    # ------------------------------------------------------------------ #
    print('\n[1/8] Running MoGe depth estimation…')
    run_video_to_depth(
        video_path=VIDEO,
        depth_folder=DEPTH_DIR,
        intrinsics_folder=INTRINSICS_DIR,
        weights_path=f'{WEIGHTS}/moge',
        dev=dev,
    )
    # Reference intrinsics: all frames share the same camera model from MoGe
    intrinsics_path = f'{INTRINSICS_DIR}/000000.json'

    # ------------------------------------------------------------------ #
    # Step 2: Video segmentation (SAM2)
    # ------------------------------------------------------------------ #
    print('\n[2/8] Running SAM2 video segmentation…')
    run_video_to_masks(
        video_path=VIDEO,
        prompts_path=PROMPTS,
        masks_dir=MASKS_DIR,
        weights_dir=f'{WEIGHTS}/sam2',
        dev=dev,
    )
    # Output layout: MASKS_DIR/0/{frame:06d}.png  (human)
    #                MASKS_DIR/1/{frame:06d}.png  (object)

    # ------------------------------------------------------------------ #
    # Step 3: Object 3D mesh (SAM3D, frame 0)
    # ------------------------------------------------------------------ #
    print('\n[3/8] Running SAM3D object reconstruction…')
    # Extract frame 0 as PNG for SAM3D input
    extract_images(VIDEO, FRAMES_DIR)

    os.makedirs(OBJECTS_DIR, exist_ok=True)
    run_image_to_mesh(
        image_path=f'{FRAMES_DIR}/000000.png',
        mask_path=f'{MASKS_DIR}/{OBJECT_OBJ_ID}/000000.png',
        mesh_path=f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}.obj',
        transform_path=f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}_transform.json',
        intrinsics_path=f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}_intrinsics.json',
        weights_dir=f'{WEIGHTS}/sam3d',
        with_texture_baking=True,
        dev=dev,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Scale + pose alignment (FoundationPose, frame 0)
    # Refines the SAM3D mesh scale and estimates a 6-DoF object-to-camera
    # pose aligned to the monocular depth at frame 0. Overwrites
    # object_{id}.obj and object_{id}_transform.json with the FP outputs
    # so the gsplat initialisation uses the better-aligned mesh/pose.
    # ------------------------------------------------------------------ #
    print('\n[4/8] Running FoundationPose mesh scale + pose estimation…')
    run_estimate_mesh_scale(
        mesh_path=f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}.obj',
        rgb_path=f'{FRAMES_DIR}/000000.png',
        depth_path=f'{DEPTH_DIR}/000000.png',
        mask_path=f'{MASKS_DIR}/{OBJECT_OBJ_ID}/000000.png',
        intrinsics_path=intrinsics_path,
        weights_dir=f'{WEIGHTS}/foundation_pose',
        scale_path=f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}_fp_scale.json',
        rescaled_mesh_path=f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}.obj',   # overwrite SAM3D mesh
        pose_path=f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}_transform.json', # overwrite SAM3D transform
        dev=dev,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Per-frame object pose tracking (FoundationPose)
    # Tracks the object through the full video using the scale-corrected
    # mesh from step 4.  Saves {frame:06d}.json per-frame Transform3d
    # files to OBJECTS_DIR/object_{id}_fp_poses/, which gsplat loads to
    # initialise ObjectPoseParams — analogous to NLF poses for the body.
    # ------------------------------------------------------------------ #
    print('\n[5/8] Running FoundationPose object pose tracking…')
    fp_poses_dir = f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}_fp_poses'
    run_fp_video_to_poses(
        video_path=VIDEO,
        depth_folder=DEPTH_DIR,
        masks_folder=f'{MASKS_DIR}/{OBJECT_OBJ_ID}',
        camera_intrinsics_path=intrinsics_path,
        mesh_path=f'{OBJECTS_DIR}/object_{OBJECT_OBJ_ID}.obj',
        poses_dir=fp_poses_dir,
        weights_dir=f'{WEIGHTS}/foundation_pose',
        dev=dev,
    )

    # ------------------------------------------------------------------ #
    # Step 6: SMPL body estimation (NLF)
    # NLF expects masks at {masks_dir}/{frame:06d}.png — pass the human subdir
    # ------------------------------------------------------------------ #
    print('\n[6/8] Running NLF SMPL body estimation…')
    run_video_to_smpl(
        video_path=VIDEO,
        masks_dir=f'{MASKS_DIR}/{HUMAN_OBJ_ID}',
        intrinsics_path=intrinsics_path,
        gender='neutral',
        output_path=SMPL_RAW,
        weights_dir=f'{WEIGHTS}/nlf',
        model_type='smpl',
        dev=dev,
    )

    # ------------------------------------------------------------------ #
    # Step 7: Align SMPL to depth-space
    # ------------------------------------------------------------------ #
    print('\n[7/8] Aligning SMPL to depth-space…')
    run_align_nlf_to_depth(
        smpl_results_path=SMPL_RAW,
        depth_folder=DEPTH_DIR,
        masks_dir=f'{MASKS_DIR}/{HUMAN_OBJ_ID}',
        intrinsics_path=intrinsics_path,
        output_path=SMPL_ALIGNED,
        weights_dir=f'{WEIGHTS}/nlf',
        dev=dev,
    )

    return intrinsics_path


def run_gsplat(intrinsics_path: str, dev: bool = False, quick: bool = False, frame_step: int = 1):
    """Run the Gaussian Splatting optimization."""

    # Quick mode: fewer iterations, quarter-res training
    n_cycles                       = 10    if quick else 3
    iterations_canonical_per_cycle = 400  if quick else 1000
    iterations_pose_per_cycle      = 400  if quick else 500
    iterations_refine              = 800  if quick else 1000
    train_scale                    = 0.5 if quick else 0.5
    num_frames                     = None

    # --------------------------------------------------------------------------- #
    # Optimization parameters — tune these to improve object quality
    # --------------------------------------------------------------------------- #
    # Entity mask loss weight: drives object opacity up and pose to match SAM2 mask.
    # Higher = stronger mask adherence. Default 1.0; 3.0 works well for objects.
    weight_entity_mask  = 3.0

    # Depth loss weight: L1 between rendered and monocular depth (MoGe).
    # Acts as a per-frame geometric prior. 0.0 = disabled; 0.1 = default.
    weight_depth        = 1.0

    # How often to compute per-entity silhouette losses (every N iters).
    # 1 = every iteration (best quality, slower); 5 = default (faster).
    entity_mask_interval = 1

    # Global learning rate multiplier applied to all parameter groups.
    # 1.0 = default; 0.5 = half speed (more stable, slower); 2.0 = double (faster, may diverge).
    lr_scale            = 1.0

    # Object pose learning rate (scaled independently by lr_obj_pose, then lr_scale).
    # Lower = smoother convergence, less oscillation.
    lr_obj_pose         = 1e-4

    # Body joint angle LR. 0 = lock to NLF initialization (recommended).
    # The rendering loss gradient through LBS is too noisy to reliably improve joint
    # angles — NLF was trained specifically for this and provides much better priors.
    # Set > 0 only if NLF quality is poor and you want rendering to override it.
    lr_body_joints      = 1e-4

    # Frames sampled per iteration. Higher = smoother pose gradients, less noise.
    batch_size          = 4

    # Initial opacity for object Gaussians (sigmoid-space: 0–1).
    # Higher starting opacity gives the mask loss more signal early in training.
    initial_opacity_obj = 0.3

    # Body mesh subdivision rounds before Gaussian init.
    # 0 = ~6890 Gaussians (one per SMPL vertex)
    # 1 = ~27K  (~4× via edge midpoints — recommended for better hand/foot coverage)
    # 2 = ~110K (~16× — high quality, slower training)
    body_subdivisions = 1

    # Weight on body entity-mask loss *outside* the SAM2 body silhouette.
    # Low = gentle (avoids penalising occluded parts).
    # High = aggressively removes hand/foot ghosts. 0.5 is a good starting point.
    body_mask_outside_weight = 0.0

    # Temporal smoothness on object SE(3) poses.
    # Penalises frame-to-frame delta in translation and rotation.
    # 0 = disabled; ~0.1 gentle; ~1.0 strong. Start at 0.2 and tune up if still shaky.
    weight_obj_pose_smooth  = 1e-1

    # Temporal smoothness on body SMPL pose (global orient, body joints, root translation).
    # Same idea. Body pose has many more dimensions so needs a higher weight for equivalent
    # regularisation strength. Start at 1.0 and tune up if body still skips.
    weight_body_pose_smooth = 1e-1

    # Maximum number of Gaussians across the whole scene.
    # Lower = faster training and rendering; higher = more detail.
    max_gaussians       = 1_000_000

    # Opacity below which a Gaussian is pruned during densification.
    # Lower = keep more Gaussians (less aggressive pruning); higher = more aggressive.
    # Default 0.005 is very aggressive early in training — try 0.001 if count is too low.
    prune_opacity_threshold = 0.0010

    # Position gradient norm threshold for clone/split candidates.
    # Lower = more Gaussians densified; higher = fewer, only where error is large.
    grad_threshold      = 0.0001

    # Densify every N canonical-phase iterations.
    densify_every       = 50

    # Passes over the full video in the final pose-only sweep.
    # Each pass does one backward per frame (canonical frozen).
    # 0 = skip sweep; 1 = one clean pass (default); 2+ = more refinement.
    n_pose_sweep_passes = 1

    # If True (default): alternate canonical and pose phases each cycle.
    # If False: optimize all parameters jointly each cycle (canonical+pose simultaneously).
    alternating = False

    print('\n[8/8] Running Gaussian Splatting optimization…')
    run_video_to_gsplat(
        video_path=VIDEO,
        depth_folder=DEPTH_DIR,
        intrinsics_path=intrinsics_path,
        masks_dir=MASKS_DIR,
        prompts_path=PROMPTS,
        output_dir=GSPLAT_DIR,
        weights_dir=WEIGHTS_NLF,
        smpl_path=SMPL_ALIGNED,
        object_meshes_dir=OBJECTS_DIR,
        camera_mode='static',
        num_frames=num_frames,
        frame_step=frame_step,
        n_cycles=n_cycles,
        iterations_canonical_per_cycle=iterations_canonical_per_cycle,
        iterations_pose_per_cycle=iterations_pose_per_cycle,
        iterations_refine=iterations_refine,
        train_scale=train_scale,
        weight_entity_mask=weight_entity_mask,
        weight_depth=weight_depth,
        entity_mask_interval=entity_mask_interval,
        lr_scale=lr_scale,
        lr_obj_pose=lr_obj_pose,
        lr_body_joints=lr_body_joints,
        batch_size=batch_size,
        initial_opacity_obj=initial_opacity_obj,
        body_subdivisions=body_subdivisions,
        body_mask_outside_weight=body_mask_outside_weight,
        weight_obj_pose_smooth=weight_obj_pose_smooth,
        weight_body_pose_smooth=weight_body_pose_smooth,
        weight_body_anchor=0.0,
        weight_obj_anchor=0.0,
        max_gaussians=max_gaussians,
        prune_opacity_threshold=prune_opacity_threshold,
        grad_threshold=grad_threshold,
        densify_every=densify_every,
        n_pose_sweep_passes=n_pose_sweep_passes,
        alternating=alternating,
        dev=dev,
    )


def main():
    parser = argparse.ArgumentParser(description='Gaussian Splatting e2e example')
    parser.add_argument('--dev',        action='store_true', help='Mount local modules into containers')
    parser.add_argument('--quick',      action='store_true', help='Short run for testing (20 frames, few iters)')
    parser.add_argument('--frame-step', type=int, default=1, help='Use every Nth frame (e.g. 5 = full coverage at 1/5 density)')
    parser.add_argument('--gsplat-only', action='store_true',
                        help='Skip prerequisites and run gsplat directly (requires prior outputs)')
    args = parser.parse_args()

    intrinsics_path = f'{INTRINSICS_DIR}/000000.json'

    if not args.gsplat_only:
        intrinsics_path = run_prerequisites(dev=args.dev, quick=args.quick)

    run_gsplat(intrinsics_path, dev=args.dev, quick=args.quick, frame_step=args.frame_step)

    print(f'\nDone. Outputs:\n'
          f'  Canonical Gaussians:   {GSPLAT_DIR}/gaussians.ply\n'
          f'  Entity metadata:       {GSPLAT_DIR}/entities.json\n'
          f'  Body pose trajectory:  {GSPLAT_DIR}/poses/smpl_refined.npz\n'
          f'  Object pose trajectory:{GSPLAT_DIR}/poses/object_poses.npz\n'
          f'  Comparison video:      {GSPLAT_DIR}/renders/comparison.mp4\n'
          f'  Checkpoint renders:    {GSPLAT_DIR}/renders/*.png\n'
          f'\n'
          f'The canonical Gaussians + pose trajectories together describe the full\n'
          f'4D scene. Load gaussians.ply, then apply poses from the NPZ files per\n'
          f'frame to reconstruct any timepoint.')


if __name__ == '__main__':
    main()
