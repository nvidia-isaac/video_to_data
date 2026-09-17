# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-view preprocessing: stereo rectification and HOI metadata forwarding."""

import copy
import json
import logging
import shutil
from pathlib import Path

import yaml

from v2d.mv.rig import RigConfig

from v2d.mv.preprocess.lib.config import resolve_calibration_camera_params_path
from v2d.mv.preprocess.lib.preprocess_stereo import preprocess_stereo


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_frame_count_from_edex(camera_params_path: Path) -> int | None:
    try:
        with open(camera_params_path) as f:
            edex = json.load(f)
        header = edex[0] if isinstance(edex, list) and edex else edex
        frame_start = int(header["frame_start"])
        frame_end = int(header["frame_end"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read frame count from {camera_params_path}: {exc}")
        return None

    frame_count = frame_end - frame_start
    if frame_count <= 0:
        logger.warning(
            f"Invalid frame range in {camera_params_path}: "
            f"frame_start={frame_start}, frame_end={frame_end}"
        )
        return None
    return frame_count


def mv_preprocess(
    rig: RigConfig,
    rgb_paths: dict[int, Path],
    output_image_dirs: dict[int, Path],
    camera_params_path: Path,
    output_camera_params_path: Path,
    scale: float = 1.0,
    output_resolution: tuple[int, int] | None = None,
    correction_focal: dict[int, float] | None = None,
    num_workers: int | None = None,
    frames_slice: slice | None = None,
    output_video_paths: dict[int, Path] | None = None,
    calibration_camera_params_path: Path | None = None,
    hoi_metadata_path: Path | None = None,
    output_hoi_metadata_path: Path | None = None,
    output_prompt_path: Path | None = None,
    mesh_path: Path | None = None,
    output_mesh_dir: Path | None = None,
    extrinsics_camera_params_path: Path | None = None,
):
    """Preprocess all stereo pairs defined in a rig config.

    Camera parameters are read from ``rig.get_camera(cam_id).param`` and
    updated in place on the rig after each stereo pair is processed.
    Updated params and frame metadata are saved at the end.

    Args:
        rig: RigConfig with stereo pair definitions and loaded camera params.
        rgb_paths: Mapping from cam_id to path for RGB frames.
        output_image_dirs: Mapping from cam_id to output image directory.
        camera_params_path: Source camera params file (for save merge).
        output_camera_params_path: Where to write updated camera params.
        scale: Scale factor for images.
        output_resolution: (width, height) target after center cropping.
        correction_focal: Per-camera focal correction factors applied during
            preprocessing. The current stereo-4 dataset config corrects
            cameras 6 and 7 here rather than during calibration.
        num_workers: Number of parallel workers.
        frames_slice: Optional slice to limit frame range.
        output_video_paths: Optional mapping from cam_id to output video path.
        calibration_camera_params_path: Optional calibration EDEX supplying
            corrected ``K``/``P`` and calibrated ``T``.
        hoi_metadata_path: Optional path to ``hoi_metadata.yaml`` to forward.
            Any legacy ``object.bbox`` field is ignored; preprocessing never
            reads or writes a ``labeled_bboxes`` directory.
        output_hoi_metadata_path: Where to write the copied hoi_metadata.
        output_prompt_path: Where to write the object prompt as plain text.
        mesh_path: Optional path to object mesh file. Used to locate the
            source mesh directory; the entire directory is copied into
            ``output_mesh_dir`` so that sibling files (alternate mesh
            variants like ``output_aligned.glb``, the symmetry annotation
            ``output_symmetry.json``, etc.) travel with the mesh.
        output_mesh_dir: Directory to copy the mesh template into.
        extrinsics_camera_params_path: Deprecated alias for
            ``calibration_camera_params_path``.
    """
    resolved_calibration_path = resolve_calibration_camera_params_path(
        calibration_camera_params_path,
        extrinsics_camera_params_path,
    )
    calibration_camera_params_path = (
        Path(resolved_calibration_path)
        if resolved_calibration_path is not None
        else None
    )

    if output_video_paths is None:
        output_video_paths = {}

    if calibration_camera_params_path is not None:
        logger.info(
            "Merging calibration intrinsics from "
            f"{calibration_camera_params_path}"
        )
        rig.merge_intrinsics(calibration_camera_params_path)
    else:
        logger.info(
            "No calibration_camera_params_path provided; "
            "skipping calibration intrinsics and extrinsics merge"
        )

    for pair in rig.get_stereo_pairs():
        left_id = pair.left.cam_id
        right_id = pair.right.cam_id
        logger.info(f"Processing stereo pair: {pair.name} (cam {left_id} / {right_id})")

        (_left_pipeline, _right_pipeline), (left_param, right_param) = preprocess_stereo(
            left_path=rgb_paths[left_id],
            right_path=rgb_paths[right_id],
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
            frames_slice=frames_slice,
            left_output_video_path=output_video_paths.get(left_id),
            right_output_video_path=output_video_paths.get(right_id),
        )

        rig.cameras[left_id].param = left_param
        rig.cameras[right_id].param = right_param

    logger.info("All stereo pairs processed")

    if hoi_metadata_path is not None:
        frame_count = _read_frame_count_from_edex(camera_params_path)
        forward_hoi_metadata(
            hoi_metadata_path=hoi_metadata_path,
            output_hoi_metadata_path=output_hoi_metadata_path,
            output_prompt_path=output_prompt_path,
            frame_count=frame_count,
        )
    else:
        logger.info("No hoi_metadata_path provided; skipping metadata and prompt outputs")

    if calibration_camera_params_path is not None:
        logger.info(f"Merging calibration extrinsics from {calibration_camera_params_path}")
        rig.merge_extrinsics(calibration_camera_params_path)

    rig.save_camera_params(
        source_path=camera_params_path,
        output_path=output_camera_params_path,
    )
    logger.info(f"Updated camera params written to {output_camera_params_path}")

    if mesh_path is not None and output_mesh_dir is not None:
        mesh_path = Path(mesh_path)
        output_mesh_dir = Path(output_mesh_dir)
        shutil.copytree(
            mesh_path.parent,
            output_mesh_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("output.glb"),
        )
        logger.info(f"Pinned object template from {mesh_path.parent} to {output_mesh_dir}")
    else:
        logger.info("No mesh_path provided; skipping object mesh pinning")

    frame_meta = camera_params_path.parent / "frame_metadata.jsonl"
    if frame_meta.exists():
        output_camera_params_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(frame_meta, output_camera_params_path.parent / "frame_metadata.jsonl")


def forward_hoi_metadata(
    hoi_metadata_path: Path,
    output_hoi_metadata_path: Path,
    output_prompt_path: Path | None = None,
    frame_count: int | None = None,
):
    """Forward HOI metadata and prompt without producing labeled bboxes.

    Args:
        hoi_metadata_path: Source ``hoi_metadata.yaml``. A legacy
            ``object.bbox`` field is deliberately ignored.
        output_hoi_metadata_path: Where to write copied metadata without the
            legacy bbox field.
        output_prompt_path: Where to write the object prompt as plain text.
        frame_count: Optional total frame count to add to the copied metadata.
    """
    with open(hoi_metadata_path) as f:
        meta = yaml.safe_load(f) or {}

    object_meta = meta.get("object") or {}
    if not isinstance(object_meta, dict):
        object_meta = {}

    meta_copy = copy.deepcopy(meta)
    if isinstance(meta_copy.get("object"), dict) and "bbox" in meta_copy["object"]:
        del meta_copy["object"]["bbox"]
    if frame_count is not None:
        meta_copy["frame_count"] = frame_count

    output_hoi_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_hoi_metadata_path, "w") as f:
        yaml.dump(meta_copy, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Copied hoi_metadata (without bbox) to {output_hoi_metadata_path}")

    if output_prompt_path is not None:
        prompt = object_meta.get("prompt", "")
        if prompt:
            output_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            output_prompt_path.write_text(prompt)
            logger.info(f"Wrote object prompt to {output_prompt_path}")


def mv_preprocess_from_config(cfg):
    """Resolve config fields into explicit arguments for mv_preprocess."""
    camera_params_path = Path(cfg.camera_params_path)
    rig = RigConfig(cfg.rig_name, camera_params_path=camera_params_path)

    frames_slice = slice(cfg.get("start", 0), cfg.get("stop"), cfg.get("step", 1))

    rgb_paths: dict[int, Path] = {}
    output_image_dirs: dict[int, Path] = {}
    output_video_paths: dict[int, Path] = {}

    for pair in rig.get_stereo_pairs():
        for cam in (pair.left, pair.right):
            raw_path = str(Path(cfg.rgb_dir) / cam.image_path) + cfg.get("input_suffix", "")
            rgb_paths[cam.cam_id] = Path(raw_path)

            output_image_dirs[cam.cam_id] = Path(
                cfg.output_image_path_template.format(cam_name=cam.name)
            )

            if cfg.get("output_video_path_template"):
                output_video_paths[cam.cam_id] = Path(
                    cfg.output_video_path_template.format(cam_name=cam.name)
                )

    correction_focal_raw = cfg.get("correction_focal", {})
    correction_focal = {int(k): float(v) for k, v in correction_focal_raw.items()} if correction_focal_raw else None

    hoi_metadata_path = cfg.get("hoi_metadata_path")
    resolved_calibration_path = resolve_calibration_camera_params_path(
        cfg.get("calibration_camera_params_path"),
        cfg.get("extrinsics_camera_params_path"),
    )
    calibration_camera_params_path = (
        Path(resolved_calibration_path)
        if resolved_calibration_path
        else None
    )

    mv_preprocess(
        rig=rig,
        rgb_paths=rgb_paths,
        output_image_dirs=output_image_dirs,
        camera_params_path=camera_params_path,
        output_camera_params_path=Path(cfg.output_camera_params_path),
        scale=cfg.get("scale", 1.0),
        output_resolution=tuple(cfg.output_resolution) if cfg.get("output_resolution") else None,
        correction_focal=correction_focal,
        num_workers=cfg.get("num_workers"),
        frames_slice=frames_slice,
        output_video_paths=output_video_paths or None,
        calibration_camera_params_path=calibration_camera_params_path,
        hoi_metadata_path=Path(hoi_metadata_path) if hoi_metadata_path else None,
        output_hoi_metadata_path=Path(cfg.output_hoi_metadata_path) if hoi_metadata_path else None,
        output_prompt_path=Path(cfg.output_prompt_path) if hoi_metadata_path else None,
        mesh_path=Path(cfg.mesh_path) if cfg.get("mesh_path") else None,
        output_mesh_dir=Path(cfg.output_mesh_dir) if cfg.get("mesh_path") else None,
    )


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Multi-view preprocessing")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Directory containing per-camera input frames")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--calibration_camera_params_path", type=str, default=None,
                        help="Calibration EDEX supplying K/P and calibrated T")
    parser.add_argument("--extrinsics_camera_params_path", type=str, default=None,
                        help="Deprecated alias for --calibration_camera_params_path")
    parser.add_argument("--hoi_metadata_path", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, default=None,
                        help="Path to object mesh file to pin in the output")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_preprocess.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides: dict = {
        "rgb_dir": args.rgb_dir,
        "output_dir": args.output_dir,
        "camera_params_path": args.camera_params_path,
    }
    if args.calibration_camera_params_path is not None:
        overrides["calibration_camera_params_path"] = args.calibration_camera_params_path
    if args.extrinsics_camera_params_path is not None:
        overrides["extrinsics_camera_params_path"] = args.extrinsics_camera_params_path
    if args.hoi_metadata_path is not None:
        overrides["hoi_metadata_path"] = args.hoi_metadata_path
    if args.mesh_path is not None:
        overrides["mesh_path"] = args.mesh_path

    cfg = OmegaConf.merge(cfg, overrides)
    mv_preprocess_from_config(cfg)
