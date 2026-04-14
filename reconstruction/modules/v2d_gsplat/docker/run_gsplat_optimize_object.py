from v2d.docker.container import run_in_container
from v2d.gsplat.docker._config import IMAGE_NAME, MODULES_DIR


def run_gsplat_optimize_object(
    images_dir: str,
    masks_dir: str,
    intrinsics_path: str,
    output_dir: str,
    config_path: str = None,
    mesh_path: str = None,
    depth_dir: str = None,
    poses_dir: str = None,
    person_masks_dir: str = None,
    frame_step: int = 1,
    dev: bool = False,
) -> None:
    """
    Run standalone rigid-object Gaussian optimization inside a Docker container.

    Required inputs:
      images_dir      - folder of RGB images ({frame:06d}.png)
      masks_dir       - folder of object masks ({frame:06d}.png, flat layout)
      intrinsics_path - CameraIntrinsics JSON
      output_dir      - results written here

    Optional inputs:
      config_path     - YAML config (see ObjectOptimConfig for all fields + defaults)
      mesh_path       - .obj mesh for Gaussian init (falls back to depth unproject)
      depth_dir       - depth folder, used only for mesh-to-depth alignment at init
      poses_dir       - per-frame initial SE(3) JSONs ({frame:06d}.json, Transform3d format)
    """
    inputs = {
        'images_dir':      images_dir,
        'masks_dir':       masks_dir,
        'intrinsics_path': intrinsics_path,
    }
    if config_path is not None:
        inputs['config_path'] = config_path
    if mesh_path is not None:
        inputs['mesh_path'] = mesh_path
    if depth_dir is not None:
        inputs['depth_dir'] = depth_dir
    if poses_dir is not None:
        inputs['poses_dir'] = poses_dir
    if person_masks_dir is not None:
        inputs['person_masks_dir'] = person_masks_dir

    run_in_container(
        image=IMAGE_NAME,
        module='v2d.gsplat.lib.run_gsplat_optimize_object',
        inputs=inputs,
        outputs={'output_dir': output_dir},
        extra_args={'frame_step': frame_step},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={'PYTHONUNBUFFERED': '1'},
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rigid-object Gaussian optimization (Docker wrapper)')
    parser.add_argument('--images_dir',      required=True)
    parser.add_argument('--masks_dir',       required=True)
    parser.add_argument('--intrinsics_path', required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--config_path',     default=None)
    parser.add_argument('--mesh_path',       default=None)
    parser.add_argument('--depth_dir',       default=None)
    parser.add_argument('--poses_dir',        default=None)
    parser.add_argument('--person_masks_dir', default=None)
    parser.add_argument('--frame_step',       type=int, default=1)
    parser.add_argument('--dev',             action='store_true')
    args = parser.parse_args()

    run_gsplat_optimize_object(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        intrinsics_path=args.intrinsics_path,
        output_dir=args.output_dir,
        config_path=args.config_path,
        mesh_path=args.mesh_path,
        depth_dir=args.depth_dir,
        poses_dir=args.poses_dir,
        person_masks_dir=args.person_masks_dir,
        frame_step=args.frame_step,
        dev=args.dev,
    )
