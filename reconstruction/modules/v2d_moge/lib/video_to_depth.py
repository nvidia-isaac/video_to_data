"""
MoGe video to depth processing function.
Can be called directly from command line or imported as a function.
"""
from moge.model.v2 import MoGeModel
import os
import argparse
import torch
import numpy as np
import cv2
import json
from PIL import Image
from v2d.common.datatypes import DepthImage, CameraIntrinsics

_model = None

def _get_model(weights_path: str):
    global _model
    if _model is None:
        print(f"Initializing MoGeV2 model from {weights_path}...")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"MoGeV2 checkpoint not found at {weights_path}")

        possible_paths = [
            os.path.join(weights_path, "model.pt"),
            os.path.join(weights_path, "pytorch_model.bin"),
            os.path.join(weights_path, "model.safetensors"),
            weights_path,
        ]

        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break

        if checkpoint_path is None:
            files_in_dir = os.listdir(weights_path) if os.path.isdir(weights_path) else []
            raise FileNotFoundError(
                f"MoGeV2 checkpoint not found in {weights_path}\n"
                f"Files found: {files_in_dir}"
            )

        print(f"Loading MoGeV2 from local checkpoint: {checkpoint_path}")
        _model = MoGeModel.from_pretrained(checkpoint_path)
        _model.to("cuda")
        _model.eval()
    return _model

def _preprocess_numpy(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

def video_to_depth(video_path: str, depth_folder: str, intrinsics_folder: str, weights_path: str, batch_size: int = 8):
    """Process video to depth frames."""
    model = _get_model(weights_path)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(intrinsics_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_batch = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_batch.append(image)

        if len(image_batch) == batch_size or frame_index == total_frames - 1:
            img_arrays = [np.asarray(img) for img in image_batch]
            input_tensor = torch.stack([
                _preprocess_numpy(arr) for arr in img_arrays
            ], dim=0).to("cuda")

            with torch.no_grad():
                predictions = model.infer(input_tensor)

            for i, img in enumerate(image_batch):
                depth = predictions["depth"][i].cpu().numpy()
                intrinsics = predictions["intrinsics"][i].cpu().numpy()

                depth_img = DepthImage(depth=depth)
                camera_intrinsics = CameraIntrinsics(
                    fx=float(intrinsics[0, 0] * img.width),
                    fy=float(intrinsics[1, 1] * img.height),
                    cx=float(intrinsics[0, 2] * img.width),
                    cy=float(intrinsics[1, 2] * img.height),
                    width=img.width,
                    height=img.height
                )

                depth_img.to_pil_image().save(
                    os.path.join(depth_folder, f"{frame_index - len(image_batch) + i + 1:06d}.png")
                )
                with open(
                    os.path.join(intrinsics_folder, f"{frame_index - len(image_batch) + i + 1:06d}.json"), "w"
                ) as f:
                    json.dump(camera_intrinsics.to_dict(), f, indent=4)

            image_batch = []

        frame_index += 1

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to depth frames")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Output folder for depth images")
    parser.add_argument("--intrinsics_folder", type=str, required=True, help="Output folder for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")

    args = parser.parse_args()
    video_to_depth(args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path, args.batch_size)
