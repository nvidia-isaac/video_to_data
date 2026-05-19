from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_image_to_hands(
    image_path: str,
    output_path: str,
    weights_dir: str,
    bboxes_path: str | None = None,
    dev: bool = False,
) -> None:
    inputs = {
        "image_path":  image_path,
        "weights_dir": weights_dir,
    }
    if bboxes_path is not None:
        inputs["bboxes_path"] = bboxes_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.image_to_hands",
        inputs=inputs,
        outputs={"output_path": output_path},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WiLoR: bbox + MANO from a single image")
    parser.add_argument("--image_path",  required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--bboxes_path", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_image_to_hands(
        image_path  = args.image_path,
        output_path = args.output_path,
        weights_dir = args.weights_dir,
        bboxes_path = args.bboxes_path,
        dev         = args.dev,
    )
