import os
from v2d.docker.container import run_in_container

IMAGE_NAME = "v2d_mesh"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_mesh_get_bounding_box(
    mesh_path: str,
    output_path: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mesh.lib.run_mesh_get_bounding_box",
        inputs={"mesh": mesh_path},
        outputs={"output": output_path},
        dev=dev,
        modules_dir=_MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute mesh bounding box (via Docker)")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True, help="Output JSON file for BoundingBox3d")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_get_bounding_box(args.mesh, args.output, dev=args.dev)
