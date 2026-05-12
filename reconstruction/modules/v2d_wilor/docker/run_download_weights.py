from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_download_weights(weights_dir: str, dev: bool = False) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.download_weights",
        inputs={},
        outputs={"weights_dir": weights_dir},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Warm WiLoR HuggingFace cache")
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_download_weights(weights_dir=args.weights_dir, dev=args.dev)
