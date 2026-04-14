"""
Depth Anything 3 video to depth processing function.
Can be called directly from command line or imported as a function.

DA3 (nested model) is a multi-view model: frames are passed in chunks so the
model can reason across views within each chunk and produce consistent
metric-scale depth. Consecutive chunks share an overlap region; the median
depth of the overlap is used to scale-align each new chunk to the previous
one, giving globally consistent depth across the full video.

DA3 metric model is a monocular model: each frame is processed independently.
The raw network output requires focal-length post-processing to recover
real-world metric scale: metric_depth = focal * net_output / 300.
The focal length is taken from predicted intrinsics (if available) or from
a provided input_intrinsics_path.
"""
import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from depth_anything_3.api import DepthAnything3
from v2d.common.datatypes import DepthImage, CameraIntrinsics

_models: dict[str, DepthAnything3] = {}


def _get_model(weights_path: str) -> DepthAnything3:
    if weights_path not in _models:
        print(f"Initializing Depth Anything 3 model from {weights_path}...")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Depth Anything 3 checkpoint not found at {weights_path}")
        model = DepthAnything3.from_pretrained(weights_path)
        model = model.to("cuda")
        model.eval()
        _models[weights_path] = model
    return _models[weights_path]


def _infer_chunk(model, images, known_intrinsics, process_res, process_res_method,
                 use_ray_pose, ref_view_strategy):
    """Run inference on a list of PIL images, return prediction at model resolution."""
    infer_kwargs = {
        "process_res": process_res,
        "process_res_method": process_res_method,
        "use_ray_pose": use_ray_pose,
        "ref_view_strategy": ref_view_strategy,
    }
    if known_intrinsics is not None:
        K = np.array([[known_intrinsics.fx, 0, known_intrinsics.cx],
                      [0, known_intrinsics.fy, known_intrinsics.cy],
                      [0, 0, 1]], dtype=np.float32)
        infer_kwargs["intrinsics"] = np.tile(K[None], (len(images), 1, 1))  # (N, 3, 3)
    return model.inference(images, **infer_kwargs)


def _apply_metric_scaling(raw_depth: np.ndarray, focal_at_model_res: float) -> np.ndarray:
    """Apply DA3 metric post-processing: metric_depth = focal * net_output / 300."""
    return focal_at_model_res * raw_depth / 300.0


def video_to_depth(
    video_path: str,
    depth_folder: str,
    intrinsics_folder: str,
    weights_path: str,
    model: str = "nested",
    input_intrinsics_path: str = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    chunk_size: int = 80,
    chunk_overlap: int = 10,
):
    """Process video to depth frames using Depth Anything 3.

    Args:
        video_path:              Path to input video.
        depth_folder:            Output folder for depth images.
        intrinsics_folder:       Output folder for camera intrinsics JSON files.
        weights_path:            Path to Depth Anything 3 model weights.
        model:                   Model variant: "nested" (default) or "metric".
                                 "nested" uses DA3NESTED-GIANT-LARGE for multi-view
                                 consistent depth. "metric" uses DA3METRIC-LARGE for
                                 monocular metric depth via focal-length post-processing.
        input_intrinsics_path:   Optional path to a CameraIntrinsics JSON with known
                                 calibrated intrinsics. When provided, the intrinsics
                                 are passed to DA3 as conditioning and written to the
                                 output folder. Required for "metric" model if the model
                                 does not predict intrinsics.
        process_res:             Resolution cap for inference (default 504).
        process_res_method:      Resize method: "upper_bound_resize" or "pad_to_square"
                                 (default "upper_bound_resize").
        use_ray_pose:            Use ray-based pose estimation instead of the camera
                                 decoder. Only used for "nested" model (default False).
        ref_view_strategy:       Reference view selection strategy for multi-view
                                 inference. Only used for "nested" model.
        chunk_size:              Frames per inference chunk (nested model only).
                                 Set to 0 to process all frames in a single call.
        chunk_overlap:           Overlap frames for scale alignment between chunks
                                 (nested model only, default 10).
    """
    da3_model = _get_model(weights_path)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(intrinsics_folder, exist_ok=True)

    known_intrinsics: CameraIntrinsics | None = None
    if input_intrinsics_path is not None:
        known_intrinsics = CameraIntrinsics.load(input_intrinsics_path)

    print(f"Loading frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    n_frames = len(images)
    print(f"Loaded {n_frames} frames.")

    h, w = images[0].height, images[0].width

    if model == "metric":
        _video_to_depth_metric(
            da3_model, images, h, w, n_frames,
            depth_folder, intrinsics_folder,
            known_intrinsics, process_res, process_res_method,
        )
    else:
        _video_to_depth_nested(
            da3_model, images, h, w, n_frames,
            depth_folder, intrinsics_folder,
            known_intrinsics, process_res, process_res_method,
            use_ray_pose, ref_view_strategy, chunk_size, chunk_overlap,
        )

    print(f"Done. Wrote {n_frames} depth frames.")


def _video_to_depth_metric(
    da3_model, images, h, w, n_frames,
    depth_folder, intrinsics_folder,
    known_intrinsics, process_res, process_res_method,
):
    """Per-frame monocular metric inference with focal-length post-processing."""
    infer_kwargs = {
        "process_res": process_res,
        "process_res_method": process_res_method,
    }
    if known_intrinsics is not None:
        K = np.array([[known_intrinsics.fx, 0, known_intrinsics.cx],
                      [0, known_intrinsics.fy, known_intrinsics.cy],
                      [0, 0, 1]], dtype=np.float32)
        infer_kwargs["intrinsics"] = K[None]  # (1, 3, 3); reused each frame

    for frame_idx, img in enumerate(images):
        print(f"Metric frame {frame_idx + 1}/{n_frames}...")
        prediction = da3_model.inference([img], **infer_kwargs)

        raw_depth = np.array(prediction.depth[0])  # (H', W')
        depth_h, depth_w = raw_depth.shape

        # Resolve focal length and output intrinsics
        if known_intrinsics is not None:
            # Scale known focal to model processing resolution
            focal_at_model_res = known_intrinsics.fx * (depth_w / w)
            cam_intr = known_intrinsics
        elif prediction.intrinsics is not None:
            K = np.array(prediction.intrinsics[0])  # (3, 3) at model resolution
            focal_at_model_res = float(K[0, 0])
            scale_x = w / depth_w
            scale_y = h / depth_h
            cam_intr = CameraIntrinsics(
                fx=float(K[0, 0]) * scale_x,
                fy=float(K[1, 1]) * scale_y,
                cx=float(K[0, 2]) * scale_x,
                cy=float(K[1, 2]) * scale_y,
                width=w,
                height=h,
            )
        else:
            raise RuntimeError(
                "DA3 metric model requires focal length for metric post-processing. "
                "Provide --input_intrinsics_path with calibrated camera intrinsics."
            )

        depth = _apply_metric_scaling(raw_depth, focal_at_model_res)

        if depth.shape != (h, w):
            depth = F.interpolate(
                torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().numpy()

        DepthImage(depth=depth).to_pil_image().save(
            os.path.join(depth_folder, f"{frame_idx:06d}.png")
        )
        cam_intr.save(
            os.path.join(intrinsics_folder, f"{frame_idx:06d}.json")
        )


def _video_to_depth_nested(
    da3_model, images, h, w, n_frames,
    depth_folder, intrinsics_folder,
    known_intrinsics, process_res, process_res_method,
    use_ray_pose, ref_view_strategy, chunk_size, chunk_overlap,
):
    """Chunked multi-view inference for the nested model."""
    effective_chunk = chunk_size if chunk_size > 0 else n_frames
    stride = n_frames if effective_chunk == n_frames else max(1, effective_chunk - chunk_overlap)
    chunk_starts = list(range(0, n_frames, stride))

    prev_overlap_depths = None

    for chunk_idx, start in enumerate(chunk_starts):
        end = min(start + effective_chunk, n_frames)
        chunk_images = images[start:end]
        print(f"Chunk {chunk_idx + 1}/{len(chunk_starts)}: frames {start}–{end - 1} ({len(chunk_images)} frames)...")

        prediction = _infer_chunk(
            da3_model, chunk_images, known_intrinsics,
            process_res, process_res_method, use_ray_pose, ref_view_strategy,
        )
        depth_h, depth_w = prediction.depth[0].shape

        chunk_depths = [np.array(d) for d in prediction.depth]

        # Scale-align this chunk to the previous one using the overlap region
        if prev_overlap_depths is not None:
            n_ov = min(chunk_overlap, len(chunk_depths))
            cur_overlap = np.stack(chunk_depths[:n_ov])
            prev_median = np.median(prev_overlap_depths)
            cur_median = np.median(cur_overlap)
            scale = prev_median / cur_median if cur_median > 0 else 1.0
            print(f"  Scale alignment: {scale:.4f} (prev_median={prev_median:.4f}, cur_median={cur_median:.4f})")
            chunk_depths = [d * scale for d in chunk_depths]

        n_ov = min(chunk_overlap, len(chunk_depths))
        prev_overlap_depths = np.stack(chunk_depths[-n_ov:])

        write_start = 0 if chunk_idx == 0 else chunk_overlap
        for local_idx in range(write_start, len(chunk_depths)):
            global_idx = start + local_idx
            if global_idx >= n_frames:
                break

            depth = chunk_depths[local_idx]
            if depth.shape != (h, w):
                depth = F.interpolate(
                    torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().numpy()

            if known_intrinsics is not None:
                cam_intr = known_intrinsics
            else:
                K = np.array(prediction.intrinsics[local_idx])
                scale_x = w / depth_w
                scale_y = h / depth_h
                cam_intr = CameraIntrinsics(
                    fx=float(K[0, 0]) * scale_x,
                    fy=float(K[1, 1]) * scale_y,
                    cx=float(K[0, 2]) * scale_x,
                    cy=float(K[1, 2]) * scale_y,
                    width=w,
                    height=h,
                )

            DepthImage(depth=depth).to_pil_image().save(
                os.path.join(depth_folder, f"{global_idx:06d}.png")
            )
            cam_intr.save(
                os.path.join(intrinsics_folder, f"{global_idx:06d}.json")
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to depth frames with Depth Anything 3")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Output folder for depth images")
    parser.add_argument("--intrinsics_folder", type=str, required=True, help="Output folder for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--model", type=str, default="nested", choices=["nested", "metric"],
                        help="Model variant: 'nested' (default) or 'metric'")
    parser.add_argument("--input_intrinsics_path", type=str, default=None, help="Optional known camera intrinsics JSON")
    parser.add_argument("--process_res", type=int, default=504, help="Resolution cap for inference")
    parser.add_argument("--process_res_method", type=str, default="upper_bound_resize", help="Resize method for processing")
    parser.add_argument("--use_ray_pose", action="store_true", help="Use ray-based pose estimation (nested only)")
    parser.add_argument("--ref_view_strategy", type=str, default="saddle_balanced", help="Reference view selection strategy (nested only)")
    parser.add_argument("--chunk_size", type=int, default=0, help="Frames per inference chunk, 0=all at once (nested only)")
    parser.add_argument("--chunk_overlap", type=int, default=10, help="Overlap frames between chunks (nested only)")

    args = parser.parse_args()
    video_to_depth(
        args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path,
        model=args.model,
        input_intrinsics_path=args.input_intrinsics_path,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
