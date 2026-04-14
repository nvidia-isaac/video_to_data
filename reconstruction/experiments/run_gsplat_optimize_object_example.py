"""
Standalone rigid-object Gaussian optimization — gsplat_example_2 scene.

Optimizes canonical Gaussians and per-frame SE(3) poses for the hand-held
object (object_id=1) from the gsplat_example_2 dataset.  All prerequisite
data (frames, depth, masks, mesh, FP poses) must already exist under
data/outputs/gsplat_example_2/ — run run_gsplat_example_2.py first if not.

Run from reconstruction/:
  python experiments/run_gsplat_optimize_object_example.py

Optional flags:
  --dev         Mount local modules into the container (live code editing)
  --frame-step  Use every Nth frame, e.g. --frame-step 5 for a quick test
"""

import argparse

from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.gsplat.docker.run_gsplat_optimize_object import run_gsplat_optimize_object


OUT        = 'data/outputs/gsplat_example_2'
CONFIG     = 'experiments/configs/gsplat_optimize_object.yaml'
WEIGHTS_FP = 'data/weights/foundation_pose'

# Pre-computed inputs (produced by run_gsplat_example_2.py prerequisites)
FRAMES_DIR      = f'{OUT}/frames'
MASKS_DIR       = f'{OUT}/masks/1'          # flat mask dir for object_id=1
DEPTH_DIR       = f'{OUT}/depth'
INTRINSICS_PATH = f'{OUT}/intrinsics/000000.json'
MESH_PATH           = f'{OUT}/objects/object_1.obj'
FP_POSES_DIR        = f'{OUT}/objects/object_1_fp_poses'
EKF_POSES_DIR       = f'{OUT}/objects/object_1_ekf_poses'
PERSON_MASKS_DIR    = f'{OUT}/masks/0'

# Output
OBJECT_GSPLAT_DIR = f'{OUT}/gsplat_object'


def main():
    parser = argparse.ArgumentParser(description='Object-only Gaussian optimization example')
    parser.add_argument('--dev',        action='store_true', help='Mount local modules into container')
    parser.add_argument('--frame-step', type=int, default=1, help='Subsample every Nth frame')
    parser.add_argument('--skip-ekf',   action='store_true', help='Skip EKF smoothing, use raw FP poses')
    args = parser.parse_args()

    if not args.skip_ekf:
        run_ekf_smoothing(
            poses_dir=FP_POSES_DIR,
            mesh_path=MESH_PATH,
            intrinsics_path=INTRINSICS_PATH,
            weights_dir=WEIGHTS_FP,
            output_dir=EKF_POSES_DIR,
            masks_folder=MASKS_DIR,
            dev=args.dev,
            process_noise_xy=0.01,
            process_noise_z=0.01,
            process_noise_r=0.02,
            measurement_noise_xy=0.01,
            measurement_noise_z=0.04,
            measurement_noise_r=0.02,
        )
        poses_dir = EKF_POSES_DIR
    else:
        poses_dir = FP_POSES_DIR
    run_gsplat_optimize_object(
        images_dir=FRAMES_DIR,
        masks_dir=MASKS_DIR,
        intrinsics_path=INTRINSICS_PATH,
        output_dir=OBJECT_GSPLAT_DIR,
        config_path=CONFIG,
        mesh_path=MESH_PATH,
        depth_dir=DEPTH_DIR,
        poses_dir=poses_dir,
        person_masks_dir=PERSON_MASKS_DIR,
        frame_step=args.frame_step,
        dev=args.dev,
    )

    print(f'\nDone.  Outputs at: {OBJECT_GSPLAT_DIR}')
    print(f'  Canonical Gaussians:  {OBJECT_GSPLAT_DIR}/gaussians.ply')
    print(f'  Refined poses:        {OBJECT_GSPLAT_DIR}/poses/')
    print(f'  Debug renders:        {OBJECT_GSPLAT_DIR}/debug/batch_*/  (every {args.frame_step * 200} batches)')


if __name__ == '__main__':
    main()
