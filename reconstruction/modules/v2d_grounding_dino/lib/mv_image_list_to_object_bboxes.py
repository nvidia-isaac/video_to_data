"""Multi-view Grounding DINO: detect objects across cameras using a text prompt.

Reads the object prompt from a plain-text file and runs detection on a subset
of rectified frames for each camera defined by the rig config.
"""

import json
from pathlib import Path

from tqdm import tqdm

from v2d.mv.rig import RigConfig

from .image_list_to_object_bboxes import IMAGE_EXTENSIONS, _detect, _get_model
from .vis import visualize_image_list_bboxes


def mv_image_list_to_object_bboxes_from_config(cfg):
    """Run Grounding DINO object detection across multiple cameras."""
    rig = RigConfig(cfg.rig_config)

    prompt = Path(cfg.prompt_path).read_text().strip()

    frames_slice = slice(cfg.get("start", 0), cfg.get("stop"), cfg.get("step", 10))
    box_threshold = cfg.get("box_threshold", 0.35)
    text_threshold = cfg.get("text_threshold", 0.25)
    debug = cfg.get("debug", 0)

    model = _get_model(cfg.model_dir)

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        image_dir = Path(cfg.image_path_template.format(cam_name=cam.name))
        output_path = Path(cfg.output_path_template.format(cam_name=cam.name))

        image_files = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )[frames_slice]

        results = {}
        for image_path in tqdm(image_files, desc=f"Grounding DINO [{cam.name}]"):
            detections = _detect(model, str(image_path), prompt, box_threshold, text_threshold)
            results[image_path.stem] = detections

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Saved {len(results)} frame detections to {output_path}")

        if debug > 0:
            debug_dir = output_path.parent / f"{cam.name}_vis"
            visualize_image_list_bboxes(str(image_dir), results, str(debug_dir))


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(
        description="Multi-view Grounding DINO object detection"
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root directory containing per-camera image folders")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to plain-text file containing the object prompt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for per-camera bbox JSONs")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with Grounding DINO weights")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(Path(__file__).parent / "mv_image_list_to_object_bboxes.yaml"),
    )
    parser.add_argument("--debug", type=int, default=None, help="Debug level override")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    overrides = {
        "image_dir": args.image_dir,
        "prompt_path": args.prompt_path,
        "output_dir": args.output_dir,
        "model_dir": args.model_dir,
    }
    if args.debug is not None:
        overrides["debug"] = args.debug
    cfg = OmegaConf.merge(cfg, overrides)
    mv_image_list_to_object_bboxes_from_config(cfg)
