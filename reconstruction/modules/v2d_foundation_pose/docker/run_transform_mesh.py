import os
from v2d.docker.container import run_in_container

IMAGE_NAME = "v2d_foundation_pose"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_transform_mesh(
    input_mesh: str,
    output_mesh: str,
    transform_path: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.transform_mesh",
        inputs={"input_mesh": input_mesh, "transform": transform_path},
        outputs={"output_mesh": output_mesh},
        dev=dev,
        modules_dir=_MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run mesh transform in Docker")
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_transform_mesh(
        args.input_mesh, args.output_mesh, args.transform, dev=args.dev,
    )
