from v2d.docker.container import run_in_container
from v2d.hand_detector.docker._config import IMAGE_NAME, MODULES_DIR


def run_image_to_hand_bboxes(
    image_path: str,
    output_path: str,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    pad_ratio: float = 0.15,
    selfie: bool = False,
    dev: bool = False,
) -> None:
    extra_args: dict = {
        "max_num_hands":            max_num_hands,
        "min_detection_confidence": min_detection_confidence,
        "pad_ratio":                pad_ratio,
    }
    if selfie:
        extra_args["selfie"] = True
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hand_detector.lib.image_to_hand_bboxes",
        inputs={"image_path": image_path},
        outputs={"output_path": output_path},
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MediaPipe Hands: bbox + handedness from a single image")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--pad_ratio", type=float, default=0.15)
    parser.add_argument("--selfie", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_image_to_hand_bboxes(
        image_path               = args.image_path,
        output_path              = args.output_path,
        max_num_hands            = args.max_num_hands,
        min_detection_confidence = args.min_detection_confidence,
        pad_ratio                = args.pad_ratio,
        selfie                   = args.selfie,
        dev                      = args.dev,
    )
