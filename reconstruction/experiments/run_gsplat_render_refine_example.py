"""
Render-and-compare Gaussian object tracking — gsplat_example_2 scene.

Two-phase pipeline:
  Phase 1: select quality keyframes from FP poses, train canonical Gaussians
           with poses fixed (eliminates canonical/pose ambiguity).
  Phase 2: freeze canonical, refine every frame's pose via render-and-compare.

All prerequisite data must already exist under data/outputs/gsplat_example_2/ —
run run_gsplat_example_2.py first if not.

Run from reconstruction/:
  python experiments/run_gsplat_render_refine_example.py

Optional flags:
  --dev         Mount local modules into the container (live code editing)
  --frame-step  Use every Nth frame, e.g. --frame-step 5 for a quick test
  --skip-ekf    Use raw FP poses instead of EKF-smoothed poses as Phase 1 prior
"""

import argparse

from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.gsplat.docker.run_gsplat_render_refine import run_gsplat_render_refine


OUT        = 'data/outputs/gsplat_example_2'
CONFIG     = 'experiments/configs/gsplat_render_refine.yaml'
WEIGHTS_FP = 'data/weights/foundation_pose'

FRAMES_DIR       = f'{OUT}/frames'
MASKS_DIR        = f'{OUT}/masks/1'
DEPTH_DIR        = f'{OUT}/depth'
INTRINSICS_PATH  = f'{OUT}/intrinsics/000000.json'
MESH_PATH        = f'{OUT}/objects/object_1.obj'
FP_POSES_DIR     = f'{OUT}/objects/object_1_fp_poses'
EKF_POSES_DIR    = f'{OUT}/objects/object_1_ekf_poses'
PERSON_MASKS_DIR = f'{OUT}/masks/0'

OUTPUT_DIR = f'{OUT}/gsplat_render_refine'


def main():
    parser = argparse.ArgumentParser(description='Render-and-compare tracking example')
    parser.add_argument('--dev',        action='store_true', help='Mount local modules')
    parser.add_argument('--frame-step', type=int, default=1, help='Subsample every Nth frame')
    parser.add_argument('--skip-ekf',   action='store_true', help='Use raw FP poses')
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

    import os
    config_path = CONFIG if os.path.exists(CONFIG) else None

    run_gsplat_render_refine(
        images_dir=FRAMES_DIR,
        masks_dir=MASKS_DIR,
        intrinsics_path=INTRINSICS_PATH,
        output_dir=OUTPUT_DIR,
        config_path=config_path,
        mesh_path=MESH_PATH,
        depth_dir=DEPTH_DIR,
        poses_dir=poses_dir,
        person_masks_dir=PERSON_MASKS_DIR,
        frame_step=args.frame_step,
        dev=args.dev,
    )

    print(f'\nDone.  Outputs at: {OUTPUT_DIR}')
    print(f'  Canonical:      {OUTPUT_DIR}/canonical_gaussians.ply')
    print(f'  Refined poses:  {OUTPUT_DIR}/poses/')
    print(f'  Keyframes:      {OUTPUT_DIR}/keyframes.txt')
    print(f'  Phase 1 debug:  {OUTPUT_DIR}/debug_canonical/')
    print(f'  Phase 2 debug:  {OUTPUT_DIR}/debug_track/')


if __name__ == '__main__':
    main()
