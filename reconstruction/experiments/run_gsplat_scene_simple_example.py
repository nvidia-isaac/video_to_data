"""
Simplified joint scene optimization — gsplat_example_2.

Prerequisite data must already exist under data/outputs/gsplat_example_2/ —
run run_gsplat_example_2.py first if not.

Run from reconstruction/:
  python experiments/run_gsplat_scene_simple_example.py [--dev] [--frame-step N]
"""

import argparse

from v2d.gsplat.docker.run_gsplat_scene_simple import run_gsplat_scene_simple


OUT     = 'data/outputs/gsplat_example_2'
WEIGHTS = 'data/weights/nlf'
CONFIG  = 'experiments/configs/gsplat_scene_simple.yaml'

VIDEO           = f'{OUT}/Date03_Sub03_chairblack_lift.2.color_every4.mp4'
DEPTH_DIR       = f'{OUT}/depth'
INTRINSICS_PATH = f'{OUT}/intrinsics/000000.json'
MASKS_DIR       = f'{OUT}/masks'
PROMPTS_PATH    = 'modules/v2d_sam2/assets/test_prompts.json'
SMPL_PATH       = f'{OUT}/smpl/smpl_aligned.npz'
OBJECTS_DIR     = f'{OUT}/objects'

OUTPUT_DIR = f'{OUT}/gsplat_simple'


def main():
    parser = argparse.ArgumentParser(description='Simplified scene optimization example')
    parser.add_argument('--dev',        action='store_true')
    parser.add_argument('--frame-step', type=int, default=1)
    args = parser.parse_args()

    run_gsplat_scene_simple(
        video_path=VIDEO,
        depth_folder=DEPTH_DIR,
        intrinsics_path=INTRINSICS_PATH,
        masks_dir=MASKS_DIR,
        prompts_path=PROMPTS_PATH,
        output_dir=OUTPUT_DIR,
        weights_dir=WEIGHTS,
        config_path=CONFIG,
        smpl_path=SMPL_PATH,
        object_meshes_dir=OBJECTS_DIR,
        frame_step=args.frame_step,
        dev=args.dev,
    )

    print(f'\nDone.  Outputs at: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
