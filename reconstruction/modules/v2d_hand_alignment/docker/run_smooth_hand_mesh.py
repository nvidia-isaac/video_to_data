from v2d.docker.container import run_in_container
from v2d.hand_alignment.docker._config import IMAGE_NAME, MODULES_DIR


def run_smooth_hand_mesh(
    input_path: str,
    output_path: str,
    sigma: float = 5.0,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_alignment.lib.smooth_hand_mesh",
        inputs={"input_path": input_path},
        outputs={"output_path": output_path},
        extra_args={"sigma": sigma},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_smooth_hand_mesh(
        args.input_path, args.output_path,
        sigma=args.sigma,
        dev=args.dev,
    )
