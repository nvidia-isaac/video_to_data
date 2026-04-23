from v2d.docker.container import run_in_container
from v2d.mv.postprocess.docker._config import IMAGE_NAME, MODULES_DIR


def run_export_demo(
    seq_dir: str,
    output_dir: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.postprocess.lib.export_demo",
        inputs={"seq_dir": seq_dir},
        outputs={"output_dir": output_dir},
        gpus=True,
        dev=dev,
        modules_dir=MODULES_DIR,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run export_demo in Docker")
    parser.add_argument("--seq_dir", type=str, required=True,
                        help="Path to the exported sequence directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to write demo output (overlay videos)")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_export_demo(
        seq_dir=args.seq_dir,
        output_dir=args.output_dir,
        dev=args.dev,
    )
