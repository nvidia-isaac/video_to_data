# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Model Manager - Unified interface for local, API, and vLLM-based models.

Supports:
- Local models (COSMOS Reason via transformers, LLMs via transformers)
- API models (ChatGPT, Gemini, Claude via NVIDIA Inference API or direct APIs)
- vLLM models (local vLLM server with OpenAI-compatible API for fast inference)

Usage:
    # Local model
    manager = ModelManager()
    model = manager.get_model("nvidia/Cosmos-Reason2-8B", backend="local")

    # API model (ChatGPT via NVIDIA Inference API)
    model = manager.get_model("openai/gpt-4o", backend="api")

    # vLLM model (requires running vLLM server)
    model = manager.get_model("nvidia/Cosmos-Reason2-8B", backend="vllm",
                              api_url="http://localhost:8000/v1")

    # All have the same interface:
    result = model.generate_from_video(video_path, prompt)
    result = model.generate_text(conversation)
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Model
# =============================================================================


class BaseModel(ABC):
    """Abstract base class for all model backends.

    Defines the common interface that both LocalModelWrapper and APIModelWrapper
    must implement. This allows seamless switching between local and API models.
    """

    def __init__(self, model_name: str, fps: int = 4):
        self.model_name = model_name
        self.fps = fps

    @abstractmethod
    def generate_text(
        self,
        conversation: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a conversation.

        Args:
            conversation: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
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
        pass

    @property
    def backend_type(self) -> str:
        """Return the backend type ('local' or 'api')."""
        return "unknown"


# =============================================================================
# Local Model Wrapper
# =============================================================================


class LocalModelWrapper(BaseModel):
    """Local model backend using transformers (COSMOS Reason).

    Wraps CosmosReasonModel for local GPU inference.

    Args:
        model_name: Hugging Face model name (e.g., "nvidia/Cosmos-Reason2-8B")
        device: Device to run the model on (default: "cuda")
        fps: Frames per second for video processing
        cache_dir: Optional cache directory for model weights
    """

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-8B",
        device: str = "cuda",
        fps: int = 4,
        cache_dir: str | None = None,
    ):
        super().__init__(model_name, fps)
        self.device = device

        # Import and initialize the actual model
        from .cosmos_model import CosmosReasonModel

        logger.info(f"[LocalModelWrapper] Loading model: {model_name}")
        self._model = CosmosReasonModel(
            model_name=model_name,
            device=device,
            fps=fps,
            cache_dir=cache_dir,
        )

    @property
    def backend_type(self) -> str:
        return "local"

    def generate_text(
        self,
        conversation: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from conversation using local model."""
        return self._model.generate_text(
            conversation=conversation,
            max_new_tokens=max_new_tokens,
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
        """Generate text from video using local model."""
        return self._model.generate_from_video(
            video_path=video_path,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


# Alias for backward compatibility
LocalModel = LocalModelWrapper


# =============================================================================
# API Model Wrapper
# =============================================================================


class APIModelWrapper(BaseModel):
    """API model backend wrapping the APIModel implementation.

    Wraps APIModel (from api_model.py) for API-based inference via NVIDIA
    Inference API, providing the same BaseModel interface as LocalModelWrapper.

    Args:
        model_name: Model identifier (e.g., "openai/gpt-4o")
        api_key: API key. If None, reads from NIM_API_KEY environment variable.
        api_url: API endpoint URL. If None, uses NVIDIA default.
        fps: Frames per second for video extraction
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        api_key: str | None = None,
        api_url: str | None = None,
        fps: int = 4,
    ):
        super().__init__(model_name, fps)

        # Import and initialize the actual API model
        from .api_model import APIModel as APIModelImpl

        logger.info(f"[APIModelWrapper] Initializing model: {model_name}")
        self._model = APIModelImpl(
            model_name=model_name,
            api_key=api_key,
            api_url=api_url,
            fps=fps,
        )

    @property
    def backend_type(self) -> str:
        return "api"

    def generate_text(
        self,
        conversation: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from conversation using API model."""
        return self._model.generate_text(
            conversation=conversation,
            max_new_tokens=max_new_tokens,
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
        """Generate text from video using API model."""
        return self._model.generate_from_video(
            video_path=video_path,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


# Alias for backward compatibility
APIModel = APIModelWrapper


# =============================================================================
# vLLM Model Wrapper
# =============================================================================


class VLLMModelWrapper(BaseModel):
    """vLLM model backend using OpenAI-compatible API.

    Connects to a running vLLM server for fast inference with optimized
    CUDA kernels, PagedAttention, and continuous batching. All preprocessing
    (video decoding, frame extraction, tokenization) happens server-side
    when using local file path mode.

    Args:
        model_name: Model name as served by vLLM
        api_url: vLLM server URL (default: "http://localhost:8000/v1")
        api_key: API key (default: "EMPTY" for local vLLM)
        fps: Frames per second for video (used in base64 fallback mode)
        use_local_media: Use file:// URLs for server-side video loading
    """

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-8B",
        api_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        fps: int = 4,
        use_local_media: bool = True,
    ):
        super().__init__(model_name, fps)

        from .vllm_model import VLLMModel

        logger.info(f"[VLLMModelWrapper] Connecting to vLLM: {model_name} @ {api_url}")
        self._model = VLLMModel(
            model_name=model_name,
            api_url=api_url,
            api_key=api_key,
            fps=fps,
            use_local_media=use_local_media,
        )

    @property
    def backend_type(self) -> str:
        return "vllm"

    def generate_text(
        self,
        conversation: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from conversation using vLLM server."""
        return self._model.generate_text(
            conversation=conversation,
            max_new_tokens=max_new_tokens,
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
        """Generate text from video using vLLM server."""
        return self._model.generate_from_video(
            video_path=video_path,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


# =============================================================================
# Model Manager (Singleton Factory)
# =============================================================================


class ModelManager:
    """
    Singleton model manager supporting both local and API models.

    Caches model instances by configuration to ensure models are loaded only
    once and shared across components. This is critical for GPU memory efficiency.

    Usage:
        manager = ModelManager()

        # Local model (default)
        local_model = manager.get_model("nvidia/Cosmos-Reason2-8B")

        # API model
        api_model = manager.get_model("openai/gpt-4o", backend="api")

        # Both have same interface
        result = model.generate_from_video(video_path, prompt)
    """

    _instance: "ModelManager | None" = None
    _models: dict[str, BaseModel] = {}

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._models = {}
        return cls._instance

    def get_model(
        self,
        model_name: str,
        backend: str = "local",
        device: str = "cuda",
        fps: int = 4,
        cache_dir: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        use_local_media: bool = True,
    ) -> BaseModel:
        """
        Get or create a model instance.

        Args:
            model_name: Model name/identifier
            backend: "local" for transformers, "api" for API calls,
                "vllm" for vLLM server
            device: Device for local models (default: "cuda")
            fps: Frames per second for video processing
            cache_dir: Cache directory for local model weights
            api_key: API key for API/vLLM models
            api_url: Custom API URL for API/vLLM models
            use_local_media: For vLLM backend, use file:// URLs for
                server-side video loading (default: True)

        Returns:
            Model instance (LocalModelWrapper, APIModelWrapper,
            or VLLMModelWrapper)
        """
        # Create cache key
        cache_key = f"{backend}::{model_name}::{device if backend == 'local' else 'api'}"

        if cache_key in self._models:
            logger.info(f"[ModelManager] Reusing cached model: {model_name} ({backend})")
            model = self._models[cache_key]
            # Update FPS if needed
            if model.fps != fps:
                model.fps = fps
            return model

        # Create new model
        logger.info(f"[ModelManager] Creating new model: {model_name} ({backend})")
        logger.info(f"[ModelManager] Backend: {backend}")
        if backend == "local":
            logger.info(f"[ModelManager] Device: {device}")
        elif backend == "vllm":
            logger.info(f"[ModelManager] vLLM URL: {api_url or 'http://localhost:8000/v1'}")
            logger.info(f"[ModelManager] Local media: {use_local_media}")
        logger.info(f"[ModelManager] FPS: {fps}")

        if backend == "local":
            model = LocalModelWrapper(
                model_name=model_name,
                device=device,
                fps=fps,
                cache_dir=cache_dir,
            )
        elif backend == "api":
            model = APIModelWrapper(
                model_name=model_name,
                api_key=api_key,
                api_url=api_url,
                fps=fps,
            )
        elif backend == "vllm":
            model = VLLMModelWrapper(
                model_name=model_name,
                api_url=api_url or "http://localhost:8000/v1",
                api_key=api_key,
                fps=fps,
                use_local_media=use_local_media,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'local', 'api', or 'vllm'.")

        # Cache the model
        self._models[cache_key] = model

        logger.info("[ModelManager] Model ready and cached")
        logger.info(f"[ModelManager] Total models loaded: {len(self._models)}")

        return model

    def is_loaded(
        self,
        model_name: str,
        backend: str = "local",
        device: str = "cuda",
    ) -> bool:
        """Check if a model is already loaded."""
        cache_key = f"{backend}::{model_name}::{device if backend == 'local' else 'api'}"
        return cache_key in self._models

    def get_loaded_models(self) -> dict[str, BaseModel]:
        """Get all currently loaded models."""
        return dict(self._models)

    def clear_cache(self) -> None:
        """Clear all cached models."""
        logger.warning("[ModelManager] Clearing all cached models")
        self._models.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        if cls._instance is not None:
            logger.warning("[ModelManager] Resetting singleton instance")
            cls._instance._models.clear()
            cls._instance = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    return ModelManager()


# =============================================================================
# Convenience functions
# =============================================================================


def get_local_model(
    model_name: str = "nvidia/Cosmos-Reason2-8B",
    device: str = "cuda",
    fps: int = 4,
) -> BaseModel:
    """Convenience function to get a local model.

    Args:
        model_name: Hugging Face model name
        device: Device to run on
        fps: Frames per second for video

    Returns:
        LocalModelWrapper instance (implements BaseModel interface)
    """
    return get_model_manager().get_model(
        model_name=model_name,
        backend="local",
        device=device,
        fps=fps,
    )


def get_api_model(
    model_name: str = "openai/gpt-4o",
    api_key: str | None = None,
) -> BaseModel:
    """Convenience function to get an API model.

    Args:
        model_name: Model identifier (e.g., "openai/gpt-4o")
        api_key: API key (or set via NIM_API_KEY environment variable)

    Returns:
        APIModelWrapper instance (implements BaseModel interface)
    """
    return get_model_manager().get_model(
        model_name=model_name,
        backend="api",
        api_key=api_key,
    )
