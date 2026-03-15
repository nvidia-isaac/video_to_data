import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_estimate_scale(
    mesh_path: str,
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    transform_path: str,
    output_transform_path: str,
    weights_dir: str,
    debug_dir: str = None,
    num_levels: int = 3,
    num_samples_per_level: int = 10,
    level_size: float = 2.0,
    dev: bool = False,
) -> None:
    weights_abs = os.path.abspath(weights_dir)
    weights_container = f"/data/weights_dir/{os.path.basename(weights_abs)}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.estimate_scale",
        inputs={"mesh": mesh_path, "rgb": rgb_path, "depth": depth_path, "mask": mask_path, "intrinsics": intrinsics_path, "transform": transform_path, "weights_dir": weights_dir},
        outputs={"output_transform": output_transform_path, "debug_dir": debug_dir},
        extra_args={"num_levels": num_levels, "num_samples_per_level": num_samples_per_level, "level_size": level_size},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"FOUNDATIONPOSE_WEIGHTS_DIR": weights_container},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FoundationPose scale estimation in Docker")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output-transform", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--num-samples-per-level", type=int, default=10)
    parser.add_argument("--level-size", type=float, default=2.0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_estimate_scale(
        args.mesh, args.rgb, args.depth, args.mask, args.intrinsics,
        args.transform, args.output_transform, args.weights_dir,
        debug_dir=args.debug_dir, num_levels=args.num_levels,
        num_samples_per_level=args.num_samples_per_level,
        level_size=args.level_size, dev=args.dev,
    )
