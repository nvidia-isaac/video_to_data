from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_align_mesh_scale(
    mesh_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    transform_path: str,
    output_transform_path: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.align_mesh_scale",
        inputs={"mesh": mesh_path, "depth": depth_path, "mask": mask_path, "intrinsics": intrinsics_path, "transform": transform_path},
        outputs={"output_transform": output_transform_path},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run mesh scale alignment in Docker")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output-transform", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_mesh_scale(
        args.mesh, args.depth, args.mask, args.intrinsics,
        args.transform, args.output_transform, dev=args.dev,
    )
