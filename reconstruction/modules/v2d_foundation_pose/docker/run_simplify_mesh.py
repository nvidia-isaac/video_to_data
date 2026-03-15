from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_simplify_mesh(
    input_mesh: str,
    output_mesh: str,
    faces: int = None,
    factor: float = None,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.simplify_mesh",
        inputs={"input_mesh": input_mesh},
        outputs={"output_mesh": output_mesh},
        extra_args={"faces": faces, "factor": factor},
        dev=dev,
        modules_dir=MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run mesh simplification in Docker")
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument("--factor", type=float, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_simplify_mesh(
        args.input_mesh, args.output_mesh,
        faces=args.faces, factor=args.factor, dev=args.dev,
    )
