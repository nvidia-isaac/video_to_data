from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import imageio.v3 as iio
import numpy as np
import pyglet
pyglet.options['headless'] = True
import torch
from tqdm import tqdm

from v2d.common.datatypes import CameraIntrinsics
from v2d.common.video import FrameSource, get_video_writer

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
import trimesh
from v2d.mv.vis.renderer import Renderer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_SOURCE_MASK = "mask"
PROMPT_SOURCE_BBOX_FALLBACK = "bbox_fallback"
PROMPT_SOURCE_FULL_IMAGE_FALLBACK = "full_image_fallback"
PROMPT_METADATA_VERSION = 1


def mhr_estimation_metadata_path(params_path: Path) -> Path:
    """Return the sidecar metadata path for cached per-camera MHR params."""
    return Path(str(params_path) + ".meta.json")


def _metadata_path_value(path: Path | None) -> str | None:
    return str(Path(path).resolve()) if path is not None else None


def _prompt_mode(mask_path: Path | None) -> str:
    return "mask" if mask_path is not None else "bbox"


def build_mhr_estimation_metadata(
    rgb_path: Path,
    bbox_path: Path | None,
    mask_path: Path | None,
    n_frames: int,
    prompt_counts: dict[str, int] | None = None,
) -> dict:
    """Describe the prompt inputs used to produce cached per-camera MHR params.

    The sidecar lets multiview estimation tell whether an existing
    ``mhr_params.pt`` was produced from bbox prompts or mask prompts, and from
    which source files, before reusing it.
    """
    return {
        "version": PROMPT_METADATA_VERSION,
        "prompt_mode": _prompt_mode(mask_path),
        "source_paths": {
            "rgb_path": _metadata_path_value(rgb_path),
            "bbox_path": _metadata_path_value(bbox_path),
            "mask_path": _metadata_path_value(mask_path),
        },
        "n_frames": int(n_frames),
        "prompt_counts": {
            PROMPT_SOURCE_MASK: 0,
            PROMPT_SOURCE_BBOX_FALLBACK: 0,
            PROMPT_SOURCE_FULL_IMAGE_FALLBACK: 0,
            **(prompt_counts or {}),
        },
    }


def mhr_estimation_cache_matches(
    params_path: Path,
    rgb_path: Path,
    bbox_path: Path | None,
    mask_path: Path | None,
    n_frames: int,
) -> bool:
    """Return True when an existing per-camera MHR cache can be reused.

    Without this check, a mask-driven rerun could accidentally load an older
    bbox-only ``mhr_params.pt`` and skip the new SAM2 mask prompts entirely.
    """
    params_path = Path(params_path)
    meta_path = mhr_estimation_metadata_path(params_path)
    if not params_path.exists() or not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    expected = build_mhr_estimation_metadata(
        rgb_path=rgb_path,
        bbox_path=bbox_path,
        mask_path=mask_path,
        n_frames=n_frames,
    )
    return (
        metadata.get("version") == expected["version"]
        and metadata.get("prompt_mode") == expected["prompt_mode"]
        and metadata.get("source_paths") == expected["source_paths"]
        and metadata.get("n_frames") == expected["n_frames"]
    )


def _full_image_bbox(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.array([0, 0, w, h], dtype=np.float32)


def _is_valid_bbox(bbox: np.ndarray | None) -> bool:
    if bbox is None:
        return False
    box = np.asarray(bbox, dtype=np.float32).reshape(-1, 4)[0]
    return bool(
        np.isfinite(box).all()
        and box[2] - box[0] > 1.0
        and box[3] - box[1] > 1.0
    )


def _normalize_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Normalize a mask image to uint8 (H, W, 1), preserving 0/positive values."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask.max(axis=-1)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    h, w = image_shape
    if mask.shape[:2] != (h, w):
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {(h, w)}"
        )

    mask = (mask > 0).astype(np.uint8) * 255
    return mask[..., None]


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Return xyxy bbox around positive mask pixels, or None for empty masks."""
    mask_2d = mask[..., 0] if mask.ndim == 3 else mask
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=np.float32,
    )


def _select_frame_prompt(
    image: np.ndarray,
    stem: str,
    frame_idx: int,
    bbox_track: np.ndarray | None,
    mask_source: FrameSource | None,
    mask_stem_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Select bbox/mask prompt for one frame, preferring non-empty masks."""
    if mask_source is not None and stem in mask_stem_to_idx:
        mask = _normalize_mask(
            mask_source[mask_stem_to_idx[stem]],
            image_shape=image.shape[:2],
        )
        mask_bbox = _bbox_from_mask(mask)
        if _is_valid_bbox(mask_bbox):
            return mask_bbox, mask, PROMPT_SOURCE_MASK

    if bbox_track is not None and _is_valid_bbox(bbox_track[frame_idx]):
        return (
            np.asarray(bbox_track[frame_idx], dtype=np.float32).reshape(4),
            None,
            PROMPT_SOURCE_BBOX_FALLBACK,
        )

    return _full_image_bbox(image), None, PROMPT_SOURCE_FULL_IMAGE_FALLBACK


def coalesce_mhr_outputs_dict(
    all_mhr_outputs_dicts: list[dict],
) -> dict:
    """Stack per-frame MHR output dicts into a single dict of (N, ...) tensors."""
    def _stack(key):
        return torch.stack([frame[key] for frame in all_mhr_outputs_dicts])

    return {
        "global_rot": _stack("global_rot"),
        "body_pose_params": _stack("body_pose_params"),
        "hand_pose_params": _stack("hand_pose_params"),
        "scale_params": _stack("scale_params"),
        "shape_params": _stack("shape_params"),
        "expr_params": _stack("expr_params"),
        "pred_cam_t": _stack("pred_cam_t"),
        "focal_length": _stack("focal_length"),
        "pred_vertices": _stack("pred_vertices"),
        "pred_keypoints_3d": _stack("pred_keypoints_3d"),
        "pred_keypoints_2d": _stack("pred_keypoints_2d"),
        "pred_joint_coords": _stack("pred_joint_coords"),
        "pred_pose_raw": _stack("pred_pose_raw"),
        "pred_global_rots": _stack("pred_global_rots"),
        "mhr_model_params": _stack("mhr_model_params"),
    }


def export_mhr_outputs(
    mhr_outputs: dict,
    faces: torch.Tensor,
) -> tuple[dict, dict]:
    """Separately export MHR params and mesh.

    Patches all 3D outputs to include translation (pred_cam_t) and writes
    global_trans into mhr_model_params[0:3] so the saved format is consistent
    with mv_optimize_mhr_params output.
    """
    cam_t = mhr_outputs["pred_cam_t"]  # (N, 3) Y/Z-flipped camera coords

    pred_vertices = mhr_outputs["pred_vertices"] + cam_t[:, None, :]
    pred_keypoints_3d = mhr_outputs["pred_keypoints_3d"] + cam_t[:, None, :]
    pred_joint_coords = mhr_outputs["pred_joint_coords"] + cam_t[:, None, :]

    native_cam_t = cam_t.clone()
    native_cam_t[..., [1, 2]] *= -1  # undo Y/Z flip -> MHR-native
    mhr_model_params = mhr_outputs["mhr_model_params"].clone()
    mhr_model_params[:, 0:3] = native_cam_t * 10

    mhr_params = {
        "global_rot": mhr_outputs["global_rot"],
        "body_pose_params": mhr_outputs["body_pose_params"],
        "hand_pose_params": mhr_outputs["hand_pose_params"],
        "scale_params": mhr_outputs["scale_params"],
        "shape_params": mhr_outputs["shape_params"],
        "expr_params": mhr_outputs["expr_params"],
        "pred_cam_t": mhr_outputs["pred_cam_t"],
        "focal_length": mhr_outputs["focal_length"],
        "pred_keypoints_3d": pred_keypoints_3d,
        "pred_keypoints_2d": mhr_outputs["pred_keypoints_2d"],
        "pred_joint_coords": pred_joint_coords,
        "pred_pose_raw": mhr_outputs["pred_pose_raw"],
        "pred_global_rots": mhr_outputs["pred_global_rots"],
        "mhr_model_params": mhr_model_params,
    }
    mhr_mesh = {
        "faces": faces,
        "pred_vertices": pred_vertices,
        "pred_cam_t": mhr_outputs["pred_cam_t"],
    }
    return mhr_params, mhr_mesh


def estimate_mhr_params(
    rgb_path: Path,
    output_params_path: Path,
    cam_intrinsics: np.ndarray | None = None,
    bbox_path: Path | None = None,
    mask_path: Path | None = None,
    output_mesh_path: Path | None = None,
    estimator: SAM3DBodyEstimator | None = None,
    weights_dir: Path | None = None,
    batch_size: int = 1,
    debug: int = 0,
) -> dict:
    """Run SAM3D-Body inference on a single camera's frames.

    If *cam_intrinsics* is None, the model uses a default FOV.
    Either *bbox_path* or *mask_path* is required. If masks are provided,
    valid non-empty masks drive both crop bboxes and mask conditioning. Missing
    or empty masks fall back to bboxes, then to full-image crops with no mask
    conditioning.
    If *batch_size* > 1, frames are processed in batches via process_batch,
    which amortizes Python/PyTorch dispatcher overhead — meaningful win on
    cluster nodes with slower per-core CPUs. batch_size=1 uses process_one_image.

    Returns the coalesced mhr_params dict and saves it to *output_params_path*.
    """
    output_params_path = Path(output_params_path).resolve()
    output_params_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path = Path(rgb_path)
    bbox_path = Path(bbox_path) if bbox_path is not None else None
    mask_path = Path(mask_path) if mask_path is not None else None

    if bbox_path is None and mask_path is None:
        raise ValueError("Either bbox_path or mask_path is required")

    frame_source = FrameSource.from_path(rgb_path)
    n_frames = frame_source.n_frames
    image_size = frame_source.image_size

    bbox_track = None
    if bbox_path is not None:
        if not bbox_path.exists():
            raise FileNotFoundError(f"BBox track not found: {bbox_path}")
        bbox_data = torch.load(bbox_path, weights_only=False)
        bbox_track = bbox_data["bbox_track"]
        assert bbox_track.shape[0] == n_frames, (
            f"BBox track has {bbox_track.shape[0]} frames, but source has {n_frames}"
        )

    mask_source = None
    mask_stem_to_idx: dict[str, int] = {}
    if mask_path is not None:
        mask_source = FrameSource.from_path(mask_path)
        mask_stem_to_idx = {stem: i for i, stem in enumerate(mask_source.stems)}

    if estimator is None:
        if weights_dir is None:
            raise ValueError("weights_dir is required")
        body_model_path = weights_dir / "sam-3d-body-dinov3/model.ckpt"
        mhr_path = weights_dir / "sam-3d-body-dinov3/assets/mhr_model.pt"
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=str(body_model_path),
            device=DEVICE,
            mhr_path=str(mhr_path),
        )
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
        )

    if debug > 0:
        renderer = Renderer(image_size=image_size)
        (output_params_path.parent / "mhr_overlay").mkdir(parents=True, exist_ok=True)
        if debug > 1:
            writer = get_video_writer(output_params_path.parent / "mhr_overlay.mp4", fps=30, crf=23)

    # Disable per-frame empty_cache() inside SAM3DBodyEstimator.process_one_image:
    # it forces a full CUDA sync each frame and stalls the pipeline.
    torch.cuda.empty_cache = lambda: None

    cam_int_tensor = (
        torch.from_numpy(cam_intrinsics) if cam_intrinsics is not None else None
    )

    all_outputs: list[dict] = []
    prompt_counts = {
        PROMPT_SOURCE_MASK: 0,
        PROMPT_SOURCE_BBOX_FALLBACK: 0,
        PROMPT_SOURCE_FULL_IMAGE_FALLBACK: 0,
    }
    frame_iter = frame_source.iter_frames()
    pbar = tqdm(total=n_frames, desc="Running SAM3D-Body estimation")
    for batch_idx, start in enumerate(range(0, n_frames, batch_size)):
        end = min(start + batch_size, n_frames)
        bs = end - start

        images = [next(frame_iter) for _ in range(bs)]
        bbox_list = []
        mask_list = []
        prompt_source_list = []
        for j, image in enumerate(images):
            frame_idx = start + j
            box, mask, prompt_source = _select_frame_prompt(
                image=image,
                stem=frame_source.stems[frame_idx],
                frame_idx=frame_idx,
                bbox_track=bbox_track,
                mask_source=mask_source,
                mask_stem_to_idx=mask_stem_to_idx,
            )
            bbox_list.append(box)
            mask_list.append(mask)
            prompt_source_list.append(prompt_source)
            prompt_counts[prompt_source] += 1

        batch_has_masks = any(mask is not None for mask in mask_list)
        cam_int_list = (
            [cam_int_tensor for _ in range(bs)]
            if cam_int_tensor is not None else None
        )

        # Set to True (or `batch_idx == 2`, etc.) to dump a torch.profiler
        # table for one batch — useful when investigating CPU dispatch hotspots.
        do_profile = False

        def _run():
            if bs == 1:
                # Keep single-frame path going through process_one_image
                # for parity with batch_size=1 callers.
                ci = cam_int_list[0].unsqueeze(0) if cam_int_list is not None else None
                bb = bbox_list[0]
                mm = mask_list[0] if batch_has_masks else None
                return estimator.process_one_image(
                    img=images[0], bboxes=bb, masks=mm, cam_int=ci, inference_type="body",
                )
            return estimator.process_batch(
                images=images, bboxes=bbox_list,
                masks=mask_list if batch_has_masks else None,
                cam_ints=cam_int_list,
                inference_type="body",
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()

        if do_profile:
            from torch.profiler import profile, ProfilerActivity
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                outputs = _run()
            tqdm.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            tqdm.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        else:
            outputs = _run()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_time = time.time() - t_start

        assert len(outputs) == bs, f"Expected {bs} outputs, got {len(outputs)}"
        for output, prompt_source in zip(outputs, prompt_source_list):
            output["prompt_source"] = prompt_source
        all_outputs.extend(outputs)

        if batch_idx % 10 == 0:
            tqdm.write(
                f"Frames {start}-{end - 1}: infer={1000 * infer_time:.1f}ms "
                f"({1000 * infer_time / bs:.1f}ms/frame, batch={bs})"
            )

        if debug > 0:
            for j, (image, frame_output) in enumerate(zip(images, outputs)):
                i = start + j
                should_render = debug > 1 or i % 30 == 0
                if not should_render:
                    continue
                pred_cam_t = frame_output["pred_cam_t"].cpu().numpy()
                cam_pose = np.eye(4)
                cam_pose[:3, 3] = -pred_cam_t
                verts = frame_output["pred_vertices"].cpu().numpy()
                mesh = trimesh.Trimesh(
                    vertices=verts,
                    faces=estimator.faces,
                    process=False,
                )
                mesh.visual.vertex_colors = np.full((len(verts), 4), [102, 230, 179, 255], dtype=np.uint8)
                if cam_intrinsics is not None:
                    K = cam_intrinsics
                else:
                    f = float(frame_output["focal_length"].cpu().numpy().reshape(-1)[0])
                    h, w = image.shape[:2]
                    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
                rendered_image = renderer.render_overlay(
                    meshes=[mesh], K=K, T=cam_pose, image=image,
                ) * 255.0
                rendered_image = rendered_image.astype(np.uint8)
                label = f"Frame {i}"
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(rendered_image, label, (rendered_image.shape[1] - tw - 10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if debug > 1:
                    writer.write_frame(rendered_image)
                if i % 30 == 0:
                    iio.imwrite(output_params_path.parent / "mhr_overlay" / f"{i:06d}.png",
                                rendered_image)

        pbar.update(bs)
        batch_idx += 1
    pbar.close()

    mhr_outputs = coalesce_mhr_outputs_dict(all_mhr_outputs_dicts=all_outputs)
    mhr_params, mhr_mesh = export_mhr_outputs(
        mhr_outputs=mhr_outputs,
        faces=estimator.model.head_pose.faces,
    )

    torch.save(mhr_params, output_params_path)
    print(f"Saved MHR params for {n_frames} frames to {output_params_path}")
    metadata = build_mhr_estimation_metadata(
        rgb_path=rgb_path,
        bbox_path=bbox_path,
        mask_path=mask_path,
        n_frames=n_frames,
        prompt_counts=prompt_counts,
    )
    with open(mhr_estimation_metadata_path(output_params_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved MHR metadata to {mhr_estimation_metadata_path(output_params_path)}")

    if output_mesh_path is not None:
        torch.save(mhr_mesh, output_mesh_path)
        print(f"Saved MHR mesh for {n_frames} frames to {output_mesh_path}")

    if debug > 0:
        renderer.close()
        if debug > 1:
            writer.close()
            print(f"Saved debug video to {output_params_path.parent / 'mhr_overlay.mp4'}")

    return mhr_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3D-Body estimation on a single camera")
    parser.add_argument("--rgb_path", type=Path, required=True,
                        help="Path to input frames (image dir, .h5, or video file)")
    parser.add_argument("--cam_intrinsics_path", type=Path, default=None,
                        help="Path to camera intrinsics JSON. If omitted, the model uses a default FOV.")
    parser.add_argument("--weights_dir", type=Path, required=True, help="Directory containing model weights")
    parser.add_argument("--bbox_path", type=Path, default=None,
                        help="Optional path to bbox track .pt file.")
    parser.add_argument("--mask_path", type=Path, default=None,
                        help="Optional path to SAM2 mask directory or .h5 file.")
    parser.add_argument("--output_params_path", type=Path, default=Path("mhr_params.pt"))
    parser.add_argument("--output_mesh_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Frames per inference call (>1 amortizes Python dispatcher overhead).")
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    cam_intrinsics = (
        CameraIntrinsics.load(str(args.cam_intrinsics_path)).to_matrix()
        if args.cam_intrinsics_path is not None else None
    )

    estimate_mhr_params(
        cam_intrinsics=cam_intrinsics,
        rgb_path=args.rgb_path,
        bbox_path=args.bbox_path,
        mask_path=args.mask_path,
        output_params_path=args.output_params_path,
        output_mesh_path=args.output_mesh_path,
        weights_dir=args.weights_dir,
        batch_size=args.batch_size,
        debug=args.debug,
    )
