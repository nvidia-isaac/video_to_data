# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM-based model wrapper.

This module provides a model wrapper that connects to a vLLM server
via its OpenAI-compatible API. vLLM provides optimized inference with
PagedAttention, continuous batching, and optimized CUDA kernels,
giving 2-5x speedup over raw HuggingFace transformers.

Supports two modes for video input:
- Local file path mode (default): Sends file:// URLs to vLLM, which
  reads the video directly from disk. Requires vLLM to be started
  with --allowed-local-media-path. This is the fastest option.
- Base64 fallback: Extracts frames client-side and sends as base64
  image_url entries. Works with remote vLLM servers.

Usage:
    # Start vLLM server first:
    # vllm serve nvidia/Cosmos-Reason2-8B \
    #   --allowed-local-media-path / \
    #   --max-model-len 32768 \
    #   --media-io-kwargs '{"video": {"num_frames": -1}}' \
    #   --mm-processor-kwargs '{"min_pixels": 262144, "max_pixels": 8388608}' \
    #   --port 8000

    model = VLLMModel(api_url="http://localhost:8000/v1")
    result = model.generate_from_video("video.mp4", "Describe this video")
"""

import base64
import io
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests as _requests

from video_ingestion_agent.utils.video_utils import (
    extract_frames_base64 as _extract_frames_base64_shared,
)

logger = logging.getLogger(__name__)


class VLLMModel:
    """vLLM model backend using the OpenAI-compatible API.

    Connects to a running vLLM server for fast inference. All preprocessing
    (video frame extraction, resizing, tokenization) is handled server-side
    when using local file path mode.

    Args:
        model_name: Model name as served by vLLM (must match the model
            loaded in the vLLM server).
        api_url: Base URL for the vLLM OpenAI-compatible API
            (default: "http://localhost:8000/v1").
        api_key: API key. Not required for local vLLM servers.
        fps: Frames per second for video extraction (only used in
            base64 fallback mode).
        use_local_media: If True (default), send video as file:// URL
            for server-side processing. If False, extract frames
            client-side and send as base64.
    """

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-8B",
        api_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        fps: int = 4,
        use_local_media: bool = True,
    ):
        self.model_name = model_name
        self.fps = fps
        self.use_local_media = use_local_media
        self.api_url = api_url

        # Resolve API key: explicit > env var > "EMPTY" for local vLLM
        if api_key is None:
            api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        self.api_key = api_key

        # Import and initialize OpenAI client
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "openai package is required for the vLLM backend. "
                "Install it with: pip install 'openai>=1.0.0'"
            ) from err

        self._client = OpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=600.0,  # 10 min per request (video processing can be slow)
        )

        # Pre-flight health check
        if not self._check_health():
            raise ConnectionError(
                f"\nvLLM server is not reachable at {api_url}\n"
                f"\n"
                f"Start it with:\n"
                f"  python scripts/serve.py -c <your_config.yaml>\n"
                f"\n"
                f"Or manually:\n"
                f"  vllm serve {model_name} "
                f"--allowed-local-media-path / --port 8000\n"
            )

        logger.info(f"[VLLMModel] Initialized: {model_name}")
        logger.info(f"[VLLMModel] API URL: {api_url}")
        logger.info(f"[VLLMModel] Local media mode: {use_local_media}")
        logger.info("[VLLMModel] Server health check: OK")

    def generate_text(
        self,
        conversation: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text response from conversation.

        Args:
            conversation: List of message dicts with 'role' and 'content'.
                Content can be a string or a list of content items
                (for multimodal format compatibility).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        messages = self._format_conversation(conversation)

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0,
        )

        return response.choices[0].message.content

    def generate_from_video(
        self,
        video_path: str,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from video input.

        In local media mode, sends the video as a file:// URL for
        server-side processing (no client-side frame extraction).
        In base64 mode, extracts frames and sends as image_url entries.

        Args:
            video_path: Path to video file.
            prompt: User prompt.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        if self.use_local_media:
            # Local file path mode: vLLM reads the video directly
            abs_path = str(Path(video_path).resolve())
            user_content = [
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{abs_path}"},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            # Base64 fallback: extract frames client-side
            user_content = self._build_base64_content(video_path, prompt)

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0,
        )

        if not response.choices:
            logger.error(f"[VLLMModel] Empty choices from server. video={video_path}")
            raise RuntimeError(
                f"vLLM returned no choices for {video_path}. "
                f"Check server logs (~/.video_ingestion_agent/vllm.log) for details."
            )

        return response.choices[0].message.content

    def generate_from_frames(
        self,
        frames: list,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from a list of frames (PIL Images).

        Encodes frames as base64 JPEG and sends via the API.

        Args:
            frames: List of PIL Image objects.
            prompt: User prompt.
            system_prompt: Optional system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        user_content = []
        for frame in frames:
            # Encode PIL Image to base64 JPEG
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=85)
            frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                }
            )

        user_content.append({"type": "text", "text": prompt})

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0,
        )

        return response.choices[0].message.content

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _check_health(self, timeout: float = 3.0) -> bool:
        """Check if the vLLM server is reachable.

        Pings the /health endpoint with a short timeout.

        Returns:
            True if server responds with 200, False otherwise.
        """
        parsed = urlparse(self.api_url)
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"

        try:
            resp = _requests.get(health_url, timeout=timeout)
            return resp.status_code == 200
        except (_requests.ConnectionError, _requests.Timeout):
            return False

    def _format_conversation(self, conversation: list[dict]) -> list[dict]:
        """Convert internal conversation format to OpenAI API format.

        Handles both string content and multimodal list-of-dicts content.
        Strips non-text items (images/video) since generate_text is
        text-only.
        """
        messages = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Extract text parts only
                text_parts = [
                    item.get("text", "") for item in content if item.get("type") == "text"
                ]
                content = " ".join(text_parts)

            messages.append({"role": role, "content": content})

        return messages

    def _build_base64_content(self, video_path: str, prompt: str) -> list[dict]:
        """Extract frames from video and build base64 image_url content.

        Used as fallback when local media mode is not available.
        """
        frames_base64 = self._extract_frames_base64(video_path)

        content = []
        for frame_b64 in frames_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                }
            )

        content.append(
            {
                "type": "text",
                "text": (
                    f"[These are {len(frames_base64)} frames extracted from "
                    f"a video at regular intervals]\n\n{prompt}"
                ),
            }
        )

        return content

    def _extract_frames_base64(self, video_path: str) -> list[str]:
        """Extract frames from video and encode as base64 JPEG strings."""
        return _extract_frames_base64_shared(video_path, fps=self.fps)
