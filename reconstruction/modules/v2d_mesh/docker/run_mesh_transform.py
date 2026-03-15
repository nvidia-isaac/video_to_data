import os
from v2d.docker.container import run_in_container

IMAGE_NAME = "v2d_mesh"
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_mesh_transform(
    input_mesh_path: str,
    transform_path: str,
    output_mesh_path: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mesh.lib.run_mesh_transform",
        inputs={"input_mesh": input_mesh_path, "transform": transform_path},
        outputs={"output_mesh": output_mesh_path},
        dev=dev,
        modules_dir=_MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply a Transform3d to a mesh (via Docker)")
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output_mesh", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_transform(args.input_mesh, args.transform, args.output_mesh, dev=args.dev)
