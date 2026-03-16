from v2d.docker.container import run_in_container
from v2d.mesh.docker._config import IMAGE_NAME, MODULES_DIR


def run_mesh_render_image(
    mesh_path: str,
    intrinsics_path: str,
    output_image_path: str,
    transform_path: str | None = None,
    background_path: str | None = None,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mesh.lib.run_mesh_render_image",
        inputs={"mesh": mesh_path, "intrinsics": intrinsics_path, "transform": transform_path, "background": background_path},
        outputs={"output_image": output_image_path},
        dev=dev,
        modules_dir=MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render an RGB image of a mesh (via Docker)")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output_image", required=True)
    parser.add_argument("--transform", default=None)
    parser.add_argument("--background", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_render_image(args.mesh, args.intrinsics, args.output_image, transform_path=args.transform, background_path=args.background, dev=args.dev)
