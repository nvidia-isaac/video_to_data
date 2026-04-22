from __future__ import annotations

import argparse
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
from v2d.mv.io.video import FrameSource, get_video_writer

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
import trimesh
from v2d.mv.vis.renderer import Renderer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    cam_intrinsics: np.ndarray,
    frame_source: FrameSource,
    bbox_path: Path,
    output_params_path: Path,
    output_mesh_path: Path | None = None,
    estimator: SAM3DBodyEstimator | None = None,
    weights_dir: Path | None = None,
    debug: int = 0,
) -> dict:
    """Run SAM3D-Body inference on a single camera's frames.

    Provide exactly one of *image_dir* (folder of PNGs) or *video_path*.
    Returns the coalesced mhr_params dict and saves it to *output_params_path*.
    """
    output_params_path = Path(output_params_path).resolve()
    output_params_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = frame_source.n_frames
    image_size = frame_source.image_size

    bbox_track = None
    if bbox_path is not None:
        bbox_path = Path(bbox_path)
        if not bbox_path.exists():
            raise FileNotFoundError(f"BBox track not found: {bbox_path}")
        bbox_data = torch.load(bbox_path, weights_only=False)
        bbox_track = bbox_data["bbox_track"]
        assert bbox_track.shape[0] == n_frames, (
            f"BBox track has {bbox_track.shape[0]} frames, but source has {n_frames}"
        )

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

    all_outputs: list[dict] = []
    for i, image in tqdm(enumerate(frame_source.iter_frames()), total=n_frames,
                         desc="Running SAM3D-Body estimation"):
        bbox = bbox_track[i] if bbox_track is not None else None

        start_time = time.time()
        outputs = estimator.process_one_image(
            img=image,
            bboxes=bbox,
            cam_int=torch.from_numpy(cam_intrinsics).unsqueeze(0),
            inference_type="body",
        )
        inference_time = time.time() - start_time

        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        frame_output = outputs[0]
        all_outputs.append(frame_output)

        if debug > 0:
            should_render = debug > 1 or i % 30 == 0
            if should_render:
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
                rendered_image = renderer.render_overlay(
                    meshes=[mesh],
                    K=cam_intrinsics,
                    T=cam_pose,
                    image=image,
                ) * 255.0
                rendered_image = rendered_image.astype(np.uint8)
                label = f"Frame {i}"
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(rendered_image, label, (rendered_image.shape[1] - tw - 10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if debug > 1:
                    writer.write_frame(rendered_image)
                if i % 30 == 0:
                    tqdm.write(f"Frame {i} inference time: {inference_time:.3f}s")
                    iio.imwrite(output_params_path.parent / "mhr_overlay" / f"{i:06d}.png",
                                rendered_image)

    mhr_outputs = coalesce_mhr_outputs_dict(all_mhr_outputs_dicts=all_outputs)
    mhr_params, mhr_mesh = export_mhr_outputs(
        mhr_outputs=mhr_outputs,
        faces=estimator.model.head_pose.faces,
    )

    torch.save(mhr_params, output_params_path)
    print(f"Saved MHR params for {n_frames} frames to {output_params_path}")

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

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", type=Path, help="Directory of PNG images")
    input_group.add_argument("--video_path", type=Path, help="Path to video file")

    parser.add_argument("--cam_intrinsics_path", type=Path, required=True,
                        help="Path to camera intrinsics JSON (CameraIntrinsics format)")
    parser.add_argument("--weights_dir", type=Path, required=True, help="Directory containing model weights")
    parser.add_argument("--bbox_path", type=Path, required=True, help="Path to bbox track .npy file")
    parser.add_argument("--output_params_path", type=Path, default=Path("mhr_params.pt"))
    parser.add_argument("--output_mesh_path", type=Path, default=None)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    cam_intrinsics = CameraIntrinsics.load(str(args.cam_intrinsics_path)).to_matrix()

    estimate_mhr_params(
        cam_intrinsics=cam_intrinsics,
        frame_source=FrameSource(image_dir=args.image_dir, video_path=args.video_path),
        bbox_path=args.bbox_path,
        output_params_path=args.output_params_path,
        output_mesh_path=args.output_mesh_path,
        weights_dir=args.weights_dir,
        debug=args.debug,
    )
