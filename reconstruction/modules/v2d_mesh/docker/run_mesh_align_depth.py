from v2d.docker.container import run_in_container
from v2d.mesh.docker._config import IMAGE_NAME, MODULES_DIR


def run_mesh_align_depth(
    mesh_path: str,
    depth_path: str,
    intrinsics_path: str,
    output_transform_path: str,
    transform_path: str | None = None,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mesh.lib.run_mesh_align_depth",
        inputs={"mesh": mesh_path, "depth": depth_path, "intrinsics": intrinsics_path, "transform": transform_path},
        outputs={"output_transform": output_transform_path},
        dev=dev,
        modules_dir=MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate a scale Transform3d to align mesh depth to real depth (via Docker)")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output_transform", required=True)
    parser.add_argument("--transform", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_align_depth(
        args.mesh, args.depth, args.intrinsics, args.output_transform,
        transform_path=args.transform, dev=args.dev,
    )
