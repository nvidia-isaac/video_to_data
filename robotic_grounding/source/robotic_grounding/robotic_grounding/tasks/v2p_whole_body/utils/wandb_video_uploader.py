# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background video uploader for wandb integration."""

import glob
import os
import re
import threading
import time


class WandbVideoUploader:
    """Background thread that monitors a folder and uploads new videos to wandb."""

    def __init__(
        self,
        video_folder: str,
        check_interval: float = 30.0,
        num_steps_per_env: int = 24,
        wandb_key: str = "train/video",
    ) -> None:
        """Initialize the video uploader.

        Args:
            video_folder: Path to the folder containing videos.
            check_interval: How often to check for new videos (in seconds).
            num_steps_per_env: Number of env steps per training iteration (for step conversion).
            wandb_key: W&B metric key to log videos under (e.g. "train/video" or "eval/video").
        """
        self.video_folder = video_folder
        self.check_interval = check_interval
        self.num_steps_per_env = num_steps_per_env
        self.wandb_key = wandb_key
        self.uploaded_videos: set = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._metric_defined = False

    def start(self) -> None:
        """Start the background upload thread."""
        self._thread = threading.Thread(target=self._upload_loop, daemon=True)
        self._thread.start()
        print(f"[INFO] Started wandb video uploader for: {self.video_folder}")

    def stop(self) -> None:
        """Stop the background upload thread and upload any remaining videos."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Final upload of any remaining videos
        self._upload_new_videos()
        print(
            f"[INFO] Stopped wandb video uploader. Uploaded {len(self.uploaded_videos)} videos total."
        )

    def _upload_loop(self) -> None:
        """Main loop that checks for and uploads new videos."""
        while not self._stop_event.is_set():
            self._upload_new_videos()
            # Wait for the check interval, but check stop event more frequently
            for _ in range(int(self.check_interval)):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

    def _upload_new_videos(self) -> None:
        """Find and upload any new videos."""
        try:
            import wandb  # noqa: PLC0415

            if wandb.run is None:
                return

            # Define custom metric for videos (allows out-of-order step logging)
            if not self._metric_defined:
                step_key = self.wandb_key.replace("/video", "/video_step")
                wandb.define_metric(step_key)
                wandb.define_metric(self.wandb_key, step_metric=step_key)
                self._metric_defined = True

            # Find all mp4 files
            video_files = glob.glob(os.path.join(self.video_folder, "*.mp4"))

            for video_path in video_files:
                if video_path in self.uploaded_videos:
                    continue

                # Check if the file is still being written (size changing)
                try:
                    size1 = os.path.getsize(video_path)
                    time.sleep(0.5)
                    size2 = os.path.getsize(video_path)
                    if size1 != size2:
                        # File is still being written, skip for now
                        continue
                except OSError:
                    continue

                video_name = os.path.basename(video_path)

                # Extract env step from filename (e.g., rl-video-step-4800.mp4)
                match = re.search(r"step-(\d+)", video_name)
                if match:
                    env_step = int(match.group(1))
                    # Convert env step to training iteration
                    training_iter = env_step // self.num_steps_per_env
                else:
                    training_iter = None

                # Rename to video_train_{training_iter}.mp4 convention
                if training_iter is not None:
                    new_name = f"video_train_{training_iter}.mp4"
                    new_path = os.path.join(self.video_folder, new_name)
                    if new_path != video_path:
                        os.rename(video_path, new_path)
                        video_path = new_path  # noqa: PLW2901
                        video_name = new_name

                # Upload to wandb with the correct training iteration
                try:
                    step_key = self.wandb_key.replace("/video", "/video_step")
                    log_data = {self.wandb_key: wandb.Video(video_path, format="mp4")}
                    if training_iter is not None:
                        log_data[step_key] = training_iter
                    wandb.log(log_data, commit=False)
                    self.uploaded_videos.add(video_path)
                    print(
                        f"[INFO] Uploaded video to wandb: {video_name} (iter={training_iter})"
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to upload video {video_name}: {e}")

        except Exception as e:
            print(f"[WARNING] Error in video upload loop: {e}")
