"""
Depth Anything 3 video to depth processing function.
Can be called directly from command line or imported as a function.

DA3 is a multi-view model: frames are passed in chunks so the model can reason
across views within each chunk and produce consistent metric-scale depth.
Consecutive chunks share an overlap region; the median depth of the overlap is
used to scale-align each new chunk to the previous one, giving globally
consistent depth across the full video.
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

_model = None


def _get_model(weights_path: str) -> DepthAnything3:
    global _model
    if _model is None:
        print(f"Initializing Depth Anything 3 model from {weights_path}...")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Depth Anything 3 checkpoint not found at {weights_path}")
        _model = DepthAnything3.from_pretrained(weights_path)
        _model = _model.to("cuda")
        _model.eval()
    return _model


def _infer_chunk(model, images, known_intrinsics, process_res, process_res_method,
                 use_ray_pose, ref_view_strategy):
    """Run inference on a list of PIL images, return (depths, intrinsics_list) at model resolution."""
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
    prediction = model.inference(images, **infer_kwargs)
    return prediction


def video_to_depth(
    video_path: str,
    depth_folder: str,
    intrinsics_folder: str,
    weights_path: str,
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
        input_intrinsics_path:   Optional path to a CameraIntrinsics JSON with
                                 known calibrated intrinsics. When provided, the
                                 intrinsics are passed to DA3 as conditioning and
                                 written to the output folder instead of DA3's
                                 estimates.
        process_res:             Resolution cap for inference (default 504).
        process_res_method:      Resize method: "upper_bound_resize" or "pad_to_square"
                                 (default "upper_bound_resize").
        use_ray_pose:            Use ray-based pose estimation instead of the camera
                                 decoder (default False).
        ref_view_strategy:       Reference view selection strategy for multi-view
                                 inference: "saddle_balanced", "first", "middle", or
                                 "saddle_sim_range" (default "saddle_balanced").
        chunk_size:              Number of frames per inference chunk. Set to 0
                                 (default) to process all frames in a single call,
                                 which gives globally consistent depth and intrinsics.
                                 Use a positive value (e.g. 80) only if you hit OOM.
        chunk_overlap:           Frames shared between consecutive chunks for scale
                                 alignment (only used when chunk_size > 0, default 10).
    """
    model = _get_model(weights_path)
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

    # chunk_size=0 means process all frames in one call (preferred for scale consistency)
    effective_chunk = chunk_size if chunk_size > 0 else n_frames
    # When processing all frames at once, stride equals n_frames so there is exactly one chunk.
    stride = n_frames if effective_chunk == n_frames else max(1, effective_chunk - chunk_overlap)
    chunk_starts = list(range(0, n_frames, stride))

    all_depths = [None] * n_frames         # final depth arrays (h, w) at full resolution
    all_cam_intrinsics = [None] * n_frames
    prev_overlap_depths = None             # depths of the overlap region from the prior chunk

    for chunk_idx, start in enumerate(chunk_starts):
        end = min(start + effective_chunk, n_frames)
        chunk_images = images[start:end]
        print(f"Chunk {chunk_idx + 1}/{len(chunk_starts)}: frames {start}–{end - 1} ({len(chunk_images)} frames)...")

        prediction = _infer_chunk(
            model, chunk_images, known_intrinsics,
            process_res, process_res_method, use_ray_pose, ref_view_strategy,
        )
        depth_h, depth_w = prediction.depth[0].shape

        # Decode all depths in this chunk at model resolution
        chunk_depths = []
        for d in prediction.depth:
            chunk_depths.append(np.array(d))  # (H', W')

        # Scale-align this chunk to the previous one using the overlap region
        if prev_overlap_depths is not None:
            # Current chunk's first `chunk_overlap` frames correspond to the overlap
            n_ov = min(chunk_overlap, len(chunk_depths))
            cur_overlap = np.stack(chunk_depths[:n_ov])     # (ov, H', W')
            prev_median = np.median(prev_overlap_depths)
            cur_median  = np.median(cur_overlap)
            scale = prev_median / cur_median if cur_median > 0 else 1.0
            print(f"  Scale alignment: {scale:.4f} (prev_median={prev_median:.4f}, cur_median={cur_median:.4f})")
            chunk_depths = [d * scale for d in chunk_depths]

        # Save the overlap region of this chunk for the next chunk's alignment
        n_ov = min(chunk_overlap, len(chunk_depths))
        # The overlap for the NEXT chunk is the last `chunk_overlap` frames of this chunk
        prev_overlap_depths = np.stack(chunk_depths[-n_ov:])

        # Write frames — only write the non-overlap portion for all but the first chunk,
        # to avoid overwriting already-written frames with potentially mis-scaled data.
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

    print(f"Done. Wrote {n_frames} depth frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to depth frames with Depth Anything 3")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Output folder for depth images")
    parser.add_argument("--intrinsics_folder", type=str, required=True, help="Output folder for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--input_intrinsics_path", type=str, default=None, help="Optional known camera intrinsics JSON")
    parser.add_argument("--process_res", type=int, default=504, help="Resolution cap for inference")
    parser.add_argument("--process_res_method", type=str, default="upper_bound_resize", help="Resize method for processing")
    parser.add_argument("--use_ray_pose", action="store_true", help="Use ray-based pose estimation")
    parser.add_argument("--ref_view_strategy", type=str, default="saddle_balanced", help="Reference view selection strategy")
    parser.add_argument("--chunk_size", type=int, default=0, help="Frames per inference chunk (0 = all at once)")
    parser.add_argument("--chunk_overlap", type=int, default=10, help="Overlap frames between chunks for scale alignment")

    args = parser.parse_args()
    video_to_depth(
        args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path,
        input_intrinsics_path=args.input_intrinsics_path,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
