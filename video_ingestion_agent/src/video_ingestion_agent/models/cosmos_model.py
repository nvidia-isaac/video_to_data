# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core Cosmos Reason 2 model wrapper.

This module provides the base model wrapper for Cosmos Reason 2,
handling model loading, inference, and video processing.
"""

import logging

import torch
import transformers

logger = logging.getLogger(__name__)


class CosmosReasonModel:
    """Base wrapper for Cosmos Reason 2 model.

    This class provides the core functionality for loading and running
    Cosmos Reason 2 for video understanding tasks.

    Args:
        model_name: Hugging Face model name (e.g., "nvidia/Cosmos-Reason2-8B")
        device: Device to run the model on
        fps: Frames per second for video processing
        cache_dir: Optional cache directory for model weights
    """

    PIXELS_PER_TOKEN = 32**2

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-8B",
        device: str = "cuda",
        fps: int = 4,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.fps = fps

        # Load model and processor
        logger.info(f"Loading Cosmos Reason 2 model: {model_name}")

        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
        self.processor = transformers.Qwen3VLProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )

        # Configure vision tokens
        min_vision_tokens = 256
        max_vision_tokens = 8192
        self.processor.image_processor.size = {
            "shortest_edge": min_vision_tokens * self.PIXELS_PER_TOKEN,
            "longest_edge": max_vision_tokens * self.PIXELS_PER_TOKEN,
        }
        self.processor.video_processor.size = {
            "shortest_edge": min_vision_tokens * self.PIXELS_PER_TOKEN,
            "longest_edge": max_vision_tokens * self.PIXELS_PER_TOKEN,
        }

        self.model.eval()
        logger.info("Cosmos Reason 2 model loaded successfully")

    def _run_inference(self, inputs, max_new_tokens: int, temperature: float) -> str:
        """Run model inference and decode output.

        This is the common generate -> trim -> decode pipeline shared by
        all generation methods.

        Args:
            inputs: Tokenized inputs (already on device).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Decoded output text.
        """
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return output_text

    def generate_text(
        self,
        conversation: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text response from conversation.

        Args:
            conversation: List of conversation messages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        # Convert string content to multimodal format if needed
        # Qwen3VLProcessor expects content as list of dicts
        formatted_conversation = []
        for msg in conversation:
            if isinstance(msg.get("content"), str):
                formatted_msg = {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                }
            else:
                formatted_msg = msg
            formatted_conversation.append(formatted_msg)

        inputs = self.processor.apply_chat_template(
            formatted_conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        return self._run_inference(inputs, max_new_tokens, temperature)

    def generate_from_video(
        self,
        video_path: str,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from video input.

        Args:
            video_path: Path to video file
            prompt: User prompt
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        conversation = []
        if system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=self.fps,
        )
        inputs = inputs.to(self.model.device)

        return self._run_inference(inputs, max_new_tokens, temperature)

    def generate_from_frames(
        self,
        frames: list,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from a list of frames (PIL Images).

        Args:
            frames: List of PIL Image objects
            prompt: User prompt
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        conversation = []
        if system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": prompt})

        conversation.append(
            {
                "role": "user",
                "content": content,
            }
        )

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        return self._run_inference(inputs, max_new_tokens, temperature)
