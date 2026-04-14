"""Multi-view preprocessing: stereo rectification and HOI metadata remapping."""

import copy
import json
import logging
import shutil
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import yaml

from v2d.common.datatypes import BoundingBox
from v2d.mv.rig import RigConfig
from v2d.mv.io.video import FrameSource

from v2d.mv.preprocess.lib.image_proc import ImagePipeline
from v2d.mv.preprocess.lib.preprocess_stereo import preprocess_stereo


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mv_preprocess(
    rig: RigConfig,
    frame_sources: dict[int, FrameSource],
    output_image_dirs: dict[int, Path],
    camera_params_path: Path,
    output_camera_params_path: Path,
    scale: float = 1.0,
    output_resolution: tuple[int, int] | None = None,
    correction_focal: dict[int, float] | None = None,
    num_workers: int | None = None,
    output_video_paths: dict[int, Path] | None = None,
    extrinsics_camera_params_path: Path | None = None,
    hoi_metadata_path: Path | None = None,
    labeled_bbox_paths: dict[str, Path] | None = None,
    output_hoi_metadata_path: Path | None = None,
    output_prompt_path: Path | None = None,
):
    """Preprocess all stereo pairs defined in a rig config.

    Camera parameters are read from ``rig.get_camera(cam_id).param`` and
    updated in place on the rig after each stereo pair is processed.
    Updated params and frame metadata are saved at the end.

    Args:
        rig: RigConfig with stereo pair definitions and loaded camera params.
        frame_sources: Mapping from cam_id to FrameSource.
        output_image_dirs: Mapping from cam_id to output image directory.
        camera_params_path: Source camera params file (for save merge).
        output_camera_params_path: Where to write updated camera params.
        scale: Scale factor for images.
        output_resolution: (width, height) target after center cropping.
        correction_focal: Per-camera focal length correction factors.
        num_workers: Number of parallel workers.
        output_video_paths: Optional mapping from cam_id to output video path.
        extrinsics_camera_params_path: Optional path to merge extrinsics from.
        hoi_metadata_path: Optional path to hoi_metadata.yaml for bbox remapping.
        labeled_bbox_paths: Mapping from camera name to output labeled bbox JSON path.
        output_hoi_metadata_path: Where to write the copied hoi_metadata.
        output_prompt_path: Where to write the object prompt as plain text.
    """
    if output_video_paths is None:
        output_video_paths = {}

    pipelines: dict[str, ImagePipeline] = {}

    for pair in rig.get_stereo_pairs():
        left_id = pair.left.cam_id
        right_id = pair.right.cam_id
        logger.info(f"Processing stereo pair: {pair.name} (cam {left_id} / {right_id})")

        (left_pipeline, _right_pipeline), (left_param, right_param) = preprocess_stereo(
            left_source=frame_sources[left_id],
            right_source=frame_sources[right_id],
            left_output_image_dir=output_image_dirs[left_id],
            right_output_image_dir=output_image_dirs[right_id],
            left_param=rig.get_camera(left_id).param,
            right_param=rig.get_camera(right_id).param,
            scale=scale,
            output_resolution=output_resolution,
            correction_focal=correction_focal,
            left_cam_id=left_id,
            right_cam_id=right_id,
            num_workers=num_workers,
            left_output_video_path=output_video_paths.get(left_id),
            right_output_video_path=output_video_paths.get(right_id),
        )

        rig.cameras[left_id].param = left_param
        rig.cameras[right_id].param = right_param
        pipelines[pair.left.name] = left_pipeline

    logger.info("All stereo pairs processed")

    if hoi_metadata_path is not None:
        cam_id_to_name = {
            cam.cam_id: cam.name
            for pair in rig.get_stereo_pairs()
            for cam in (pair.left, pair.right)
        }
        output_image_dirs_by_name = {
            cam_id_to_name[cam_id]: path
            for cam_id, path in output_image_dirs.items()
        }
        remap_hoi_bboxes(
            hoi_metadata_path=hoi_metadata_path,
            pipelines=pipelines,
            output_image_dirs=output_image_dirs_by_name,
            labeled_bbox_paths=labeled_bbox_paths or {},
            output_hoi_metadata_path=output_hoi_metadata_path,
            output_prompt_path=output_prompt_path,
        )

    if extrinsics_camera_params_path is not None:
        logger.info(f"Merging extrinsics from {extrinsics_camera_params_path}")
        rig.merge_extrinsics(extrinsics_camera_params_path)

    rig.save_camera_params(
        source_path=camera_params_path,
        output_path=output_camera_params_path,
    )
    logger.info(f"Updated camera params written to {output_camera_params_path}")

    frame_meta = camera_params_path.parent / "frame_metadata.jsonl"
    if frame_meta.exists():
        output_camera_params_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(frame_meta, output_camera_params_path.parent / "frame_metadata.jsonl")


def remap_hoi_bboxes(
    hoi_metadata_path: Path,
    pipelines: dict[str, ImagePipeline],
    output_image_dirs: dict[str, Path],
    labeled_bbox_paths: dict[str, Path],
    output_hoi_metadata_path: Path,
    output_prompt_path: Path | None = None,
):
    """Remap object bboxes from hoi_metadata.yaml through preprocessing pipelines.

    For each left camera listed in the metadata's object.bbox field, remaps the
    raw-image xywh bbox into processed-image xyxy coordinates, saves them as
    grounding-dino-format JSON, and generates a visualization overlay.

    Args:
        hoi_metadata_path: Path to hoi_metadata.yaml (with object.bbox in raw coords).
        pipelines: Mapping from camera name to ImagePipeline (left cameras only).
        output_image_dirs: Mapping from camera name to directory of processed frames.
        labeled_bbox_paths: Mapping from camera name to output JSON path.
        output_hoi_metadata_path: Where to write the copied hoi_metadata (without bbox).
        output_prompt_path: Where to write the object prompt as plain text.
    """
    with open(hoi_metadata_path) as f:
        meta = yaml.safe_load(f)

    obj_bboxes = meta.get("object", {}).get("bbox", {})
    obj_id = meta.get("object", {}).get("id", "object")

    if not obj_bboxes:
        logger.warning("No object bboxes found in hoi_metadata.yaml, skipping remap")
        return

    for cam_name, xywh in obj_bboxes.items():
        if cam_name not in pipelines:
            logger.warning(f"No pipeline for camera '{cam_name}', skipping bbox remap")
            continue

        x, y, w, h = xywh
        corners = np.array([[x, y], [x + w, y + h]], dtype=np.float64)
        remapped = pipelines[cam_name].map_points(corners)
        x0, y0 = remapped[0]
        x1, y1 = remapped[1]

        bbox = BoundingBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))
        detection = [{"label": obj_id, "box": bbox.to_dict()}]

        first_frame = next(
            (p for p in sorted(output_image_dirs[cam_name].iterdir()) if p.suffix == ".png"),
            None,
        )
        frame_stem = first_frame.stem if first_frame is not None else "000000"
        results = {frame_stem: detection}

        json_path = labeled_bbox_paths[cam_name]
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved remapped bbox to {json_path}")

        if first_frame is not None:
            img = iio.imread(first_frame, plugin="pillow")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pt1 = (int(round(x0)), int(round(y0)))
            pt2 = (int(round(x1)), int(round(y1)))
            cv2.rectangle(img_bgr, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(
                img_bgr, obj_id, (pt1[0], max(pt1[1] - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            vis_path = json_path.with_name(json_path.stem + "_vis.png")
            cv2.imwrite(str(vis_path), img_bgr)
            logger.info(f"Saved bbox visualization to {vis_path}")
        else:
            logger.warning(f"No frames found in {output_image_dirs[cam_name]}, skipping visualization")

    meta_copy = copy.deepcopy(meta)
    if "object" in meta_copy and "bbox" in meta_copy["object"]:
        del meta_copy["object"]["bbox"]

    output_hoi_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_hoi_metadata_path, "w") as f:
        yaml.dump(meta_copy, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Copied hoi_metadata (without bbox) to {output_hoi_metadata_path}")

    if output_prompt_path is not None:
        prompt = meta.get("object", {}).get("prompt", "")
        output_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        output_prompt_path.write_text(prompt)
        logger.info(f"Wrote object prompt to {output_prompt_path}")


def mv_preprocess_from_config(cfg):
    """Resolve config fields into explicit arguments for mv_preprocess."""
    camera_params_path = Path(cfg.camera_params_path)
    rig = RigConfig(cfg.rig_name, camera_params_path=camera_params_path)

    frames_slice = slice(cfg.get("start", 0), cfg.get("stop"), cfg.get("step", 1))

    frame_sources: dict[int, FrameSource] = {}
    output_image_dirs: dict[int, Path] = {}
    output_video_paths: dict[int, Path] = {}
    labeled_bbox_paths: dict[str, Path] = {}

    for pair in rig.get_stereo_pairs():
        for cam in (pair.left, pair.right):
            frame_sources[cam.cam_id] = FrameSource(
                image_dir=Path(cfg.image_dir) / cam.image_path,
                frames_slice=frames_slice,
            )

            output_image_dirs[cam.cam_id] = Path(
                cfg.output_image_path_template.format(cam_name=cam.name)
            )

            if cfg.get("output_video_path_template"):
                output_video_paths[cam.cam_id] = Path(
                    cfg.output_video_path_template.format(cam_name=cam.name)
                )

            if cfg.get("labeled_bbox_path_template"):
                labeled_bbox_paths[cam.name] = Path(
                    cfg.labeled_bbox_path_template.format(cam_name=cam.name)
                )

    correction_focal_raw = cfg.get("correction_focal", {})
    correction_focal = {int(k): float(v) for k, v in correction_focal_raw.items()} if correction_focal_raw else None

    hoi_metadata_path = cfg.get("hoi_metadata_path")
    extrinsics_camera_params_path = (
        Path(cfg.extrinsics_camera_params_path)
        if cfg.get("extrinsics_camera_params_path")
        else None
    )

    mv_preprocess(
        rig=rig,
        frame_sources=frame_sources,
        output_image_dirs=output_image_dirs,
        camera_params_path=camera_params_path,
        output_camera_params_path=Path(cfg.output_camera_params_path),
        scale=cfg.get("scale", 1.0),
        output_resolution=tuple(cfg.output_resolution) if cfg.get("output_resolution") else None,
        correction_focal=correction_focal,
        num_workers=cfg.get("num_workers"),
        output_video_paths=output_video_paths or None,
        extrinsics_camera_params_path=extrinsics_camera_params_path,
        hoi_metadata_path=Path(hoi_metadata_path) if hoi_metadata_path else None,
        labeled_bbox_paths=labeled_bbox_paths or None,
        output_hoi_metadata_path=Path(cfg.output_hoi_metadata_path) if hoi_metadata_path else None,
        output_prompt_path=Path(cfg.output_prompt_path) if hoi_metadata_path else None,
    )


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Multi-view preprocessing")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing per-camera image subdirectories")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--extrinsics_camera_params_path", type=str, default=None)
    parser.add_argument("--hoi_metadata_path", type=str, default=None)
    parser.add_argument("--config_path", type=str,
                        default=str(Path(__file__).parent / "mv_preprocess.yaml"))
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    overrides: dict = {
        "image_dir": args.image_dir,
        "output_dir": args.output_dir,
        "camera_params_path": args.camera_params_path,
    }
    if args.extrinsics_camera_params_path is not None:
        overrides["extrinsics_camera_params_path"] = args.extrinsics_camera_params_path
    if args.hoi_metadata_path is not None:
        overrides["hoi_metadata_path"] = args.hoi_metadata_path

    cfg = OmegaConf.merge(cfg, overrides)
    mv_preprocess_from_config(cfg)
