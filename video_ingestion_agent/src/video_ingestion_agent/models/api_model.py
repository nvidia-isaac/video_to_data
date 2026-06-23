# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""API-based model wrapper.

This module provides the API model wrapper for external LLM services
like ChatGPT, Gemini, Claude, etc. via NVIDIA Inference API.

The NVIDIA Inference API supports multiple model providers through the
model name (e.g., "openai/gpt-4o", "google/gemini-1.5-pro", "anthropic/claude-3").
"""

import logging
import os
import time

import requests

from video_ingestion_agent.utils.video_utils import (
    extract_frames_base64 as _extract_frames_base64_shared,
)
from video_ingestion_agent.utils.video_utils import get_video_info as _get_video_info_shared

logger = logging.getLogger(__name__)


class APIModel:
    """API-based model backend using NVIDIA Inference API.

    This class provides a unified interface for making API calls to various
    LLM models via NVIDIA's Inference API, with support for both text and
    video (via frame extraction) inputs.

    Args:
        model_name: Model identifier (e.g., "openai/gpt-4o", "google/gemini-1.5-pro")
        api_key: API key. If None, reads from NIM_API_KEY environment variable.
        api_url: API endpoint URL. If None, uses NVIDIA default.
        fps: Frames per second for video extraction

    Example:
        model = APIModel("openai/gpt-4o")
        response = model.generate_text([
            {"role": "user", "content": "Hello!"}
        ])

        response = model.generate_from_video("video.mp4", "Describe this video")
    """

    # NVIDIA Inference API endpoint
    DEFAULT_API_URL = "https://inference-api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        model_name: str = "openai/openai/gpt-5.2",
        api_key: str | None = None,
        api_url: str | None = None,
        fps: int = 4,
    ):
        self.model_name = model_name
        self.fps = fps

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("NIM_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key not provided and NIM_API_KEY environment variable not set"
                )

        self.api_key = api_key
        self.api_url = api_url or self.DEFAULT_API_URL

        logger.info(f"[APIModel] Initialized: {model_name}")
        # Log initialization info (mask API key for security)
        masked_key = (
            f"{self.api_key[:8]}...{self.api_key[-4:]}"
            if self.api_key and len(self.api_key) > 12
            else "***"
        )
        logger.info(f"[APIModel] API URL: {self.api_url}")
        logger.info(f"[APIModel] API Key: {masked_key}")

    def _get_video_info(self, video_path: str) -> dict:
        """Get video information using OpenCV."""
        return _get_video_info_shared(video_path)

    def _make_request(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Make API request and return response text.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response

        Raises:
            RuntimeError: If API request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.debug(f"[APIModel] Making request to {self.api_url}")

        max_retries = 5
        backoff = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    logger.error(f"[APIModel] API request failed after {max_retries} attempts: {e}")
                    raise RuntimeError(
                        f"API request failed after {max_retries} attempts: {e}"
                    ) from e

                logger.warning(
                    f"[APIModel] Attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2

    def _extract_frames_base64(
        self,
        video_path: str,
    ) -> list[str]:
        """Extract frames from video and encode as base64 images.

        Uses self.fps to determine sampling rate. Extracts frames at the
        specified FPS rate from the video.

        Args:
            video_path: Path to video file

        Returns:
            List of base64-encoded JPEG images
        """
        return _extract_frames_base64_shared(video_path, fps=self.fps)

    def generate_text(
        self,
        conversation: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text response from conversation.

        Converts conversation format to API-compatible format and makes
        the API request.

        Args:
            conversation: List of message dicts with 'role' and 'content'.
                         Content can be a string or a list of content items
                         (for multimodal format compatibility).
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        # Convert conversation to API format
        messages = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle content that's a list (multimodal format)
            if isinstance(content, list):
                # Extract text content
                text_parts = [
                    item.get("text", "") for item in content if item.get("type") == "text"
                ]
                content = " ".join(text_parts)

            messages.append({"role": role, "content": content})

        return self._make_request(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

    def generate_from_video(
        self,
        video_path: str,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from video input.

        Extracts frames from the video, encodes them as base64 images,
        and sends them to the API along with the prompt.

        Args:
            video_path: Path to video file
            prompt: User prompt describing what to analyze
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        # Get video duration so we can tell the model about temporal context
        video_info = self._get_video_info(video_path)
        video_duration = video_info.get("duration", 0.0)

        # Extract frames as base64 images (uses self.fps for sampling)
        frames_base64 = self._extract_frames_base64(video_path)

        # Build messages with images
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        # Build user message with images
        user_content = []

        # Add frame images
        for frame_b64 in frames_base64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                    },
                }
            )

        # Include FPS and duration so the model can map frames to timestamps
        interval = 1.0 / self.fps if self.fps > 0 else 0.0
        video_context = (
            f"[Video context: {len(frames_base64)} frames extracted at {self.fps} fps "
            f"from a {video_duration:.1f}s video. "
            f"Frame 1 = 0.0s, each subsequent frame is {interval:.2f}s later.]"
        )

        # Add text prompt
        user_content.append(
            {
                "type": "text",
                "text": f"{video_context}\n\n{prompt}",
            }
        )

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        return self._make_request(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
