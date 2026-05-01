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
from v2d.common.video import FrameSource, get_video_writer

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
    rgb_path: Path,
    output_params_path: Path,
    cam_intrinsics: np.ndarray | None = None,
    bbox_path: Path | None = None,
    output_mesh_path: Path | None = None,
    estimator: SAM3DBodyEstimator | None = None,
    weights_dir: Path | None = None,
    batch_size: int = 1,
    debug: int = 0,
) -> dict:
    """Run SAM3D-Body inference on a single camera's frames.

    If *cam_intrinsics* is None, the model uses a default FOV.
    If *bbox_path* is None, the full image is used as the bbox each frame.
    If *batch_size* > 1, frames are processed in batches via process_batch,
    which amortizes Python/PyTorch dispatcher overhead — meaningful win on
    cluster nodes with slower per-core CPUs. batch_size=1 uses process_one_image.

    Returns the coalesced mhr_params dict and saves it to *output_params_path*.
    """
    output_params_path = Path(output_params_path).resolve()
    output_params_path.parent.mkdir(parents=True, exist_ok=True)

    frame_source = FrameSource.from_path(rgb_path)
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

    # Disable per-frame empty_cache() inside SAM3DBodyEstimator.process_one_image:
    # it forces a full CUDA sync each frame and stalls the pipeline.
    torch.cuda.empty_cache = lambda: None

    cam_int_tensor = (
        torch.from_numpy(cam_intrinsics) if cam_intrinsics is not None else None
    )

    all_outputs: list[dict] = []
    frame_iter = frame_source.iter_frames()
    pbar = tqdm(total=n_frames, desc="Running SAM3D-Body estimation")
    for batch_idx, start in enumerate(range(0, n_frames, batch_size)):
        end = min(start + batch_size, n_frames)
        bs = end - start

        images = [next(frame_iter) for _ in range(bs)]
        bbox_list = (
            [bbox_track[start + j] for j in range(bs)]
            if bbox_track is not None else None
        )
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
                bb = bbox_list[0] if bbox_list is not None else None
                return estimator.process_one_image(
                    img=images[0], bboxes=bb, cam_int=ci, inference_type="body",
                )
            return estimator.process_batch(
                images=images, bboxes=bbox_list, cam_ints=cam_int_list,
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
                        help="Path to bbox track .pt file. If omitted, the full image is used as the bbox.")
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
        output_params_path=args.output_params_path,
        output_mesh_path=args.output_mesh_path,
        weights_dir=args.weights_dir,
        batch_size=args.batch_size,
        debug=args.debug,
    )
