from v2d.docker.container import run_in_container
from v2d.grounding_dino.docker._config import IMAGE_NAME, MODULES_DIR


def run_image_list_to_object_bboxes(
    image_dir: str,
    output_path: str,
    prompt: str,
    model_dir: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    debug_output: str = None,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.grounding_dino.lib.image_list_to_object_bboxes",
        inputs={"image_dir": image_dir, "model_dir": model_dir},
        outputs={"output_path": output_path, "debug_output": debug_output},
        extra_args={"prompt": prompt, "box_threshold": box_threshold, "text_threshold": text_threshold},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grounding DINO: detect objects in image directory")
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--box_threshold', type=float, default=0.35)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--debug_output', type=str, default=None,
                        help='Directory to save annotated debug images')
    parser.add_argument('--dev', action='store_true', help='Mount local modules for development')
    args = parser.parse_args()
    run_image_list_to_object_bboxes(
        args.image_dir, args.output_path, args.prompt, args.model_dir,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        debug_output=args.debug_output, dev=args.dev,
    )
