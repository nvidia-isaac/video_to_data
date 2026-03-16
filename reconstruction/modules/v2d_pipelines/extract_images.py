"""Extract video frames to numbered PNG images."""
import os
import cv2


def extract_images(video_path: str, output_folder: str) -> int:
    """Extract all frames from a video to a folder as numbered PNGs.

    Output files are named 000000.png, 000001.png, ...

    Returns:
        Number of frames extracted.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"{frame_idx:06d}.png"), frame)
        frame_idx += 1
    cap.release()
    return frame_idx
