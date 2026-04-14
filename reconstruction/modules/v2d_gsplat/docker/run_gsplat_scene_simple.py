from v2d.docker.container import run_in_container
from v2d.gsplat.docker._config import IMAGE_NAME, MODULES_DIR


def run_gsplat_scene_simple(
    video_path: str,
    depth_folder: str,
    intrinsics_path: str,
    masks_dir: str,
    prompts_path: str,
    output_dir: str,
    weights_dir: str,
    config_path: str = None,
    smpl_path: str = None,
    object_meshes_dir: str = None,
    num_frames: int = None,
    frame_step: int = 1,
    dev: bool = False,
) -> None:
    inputs = {
        'video_path':      video_path,
        'depth_folder':    depth_folder,
        'intrinsics_path': intrinsics_path,
        'masks_dir':       masks_dir,
        'prompts_path':    prompts_path,
        'weights_dir':     weights_dir,
    }
    if config_path is not None:
        inputs['config_path'] = config_path
    if smpl_path is not None:
        inputs['smpl_path'] = smpl_path
    if object_meshes_dir is not None:
        inputs['object_meshes_dir'] = object_meshes_dir

    extra_args = {'frame_step': frame_step}
    if num_frames is not None:
        extra_args['num_frames'] = num_frames

    run_in_container(
        image=IMAGE_NAME,
        module='v2d.gsplat.lib.run_gsplat_scene_simple',
        inputs=inputs,
        outputs={'output_dir': output_dir},
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={'PYTHONUNBUFFERED': '1'},
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simplified joint scene optimization (Docker wrapper)')
    parser.add_argument('--video_path',        required=True)
    parser.add_argument('--depth_folder',      required=True)
    parser.add_argument('--intrinsics_path',   required=True)
    parser.add_argument('--masks_dir',         required=True)
    parser.add_argument('--prompts_path',      required=True)
    parser.add_argument('--output_dir',        required=True)
    parser.add_argument('--weights_dir',       required=True)
    parser.add_argument('--config_path',       default=None)
    parser.add_argument('--smpl_path',         default=None)
    parser.add_argument('--object_meshes_dir', default=None)
    parser.add_argument('--num_frames',        type=int, default=None)
    parser.add_argument('--frame_step',        type=int, default=1)
    parser.add_argument('--dev',               action='store_true')
    args = parser.parse_args()

    run_gsplat_scene_simple(
        video_path=args.video_path,
        depth_folder=args.depth_folder,
        intrinsics_path=args.intrinsics_path,
        masks_dir=args.masks_dir,
        prompts_path=args.prompts_path,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        config_path=args.config_path,
        smpl_path=args.smpl_path,
        object_meshes_dir=args.object_meshes_dir,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
        dev=args.dev,
    )
