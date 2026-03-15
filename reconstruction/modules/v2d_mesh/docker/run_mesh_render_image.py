import os
from v2d.docker.container import run_in_container

IMAGE_NAME = "v2d_mesh"
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_mesh_render_image(
    mesh_path: str,
    intrinsics_path: str,
    output_image_path: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mesh.lib.run_mesh_render_image",
        inputs={"mesh": mesh_path, "intrinsics": intrinsics_path},
        outputs={"output_image": output_image_path},
        dev=dev,
        modules_dir=_MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render an RGB image of a mesh (via Docker)")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output_image", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_render_image(args.mesh, args.intrinsics, args.output_image, dev=args.dev)
