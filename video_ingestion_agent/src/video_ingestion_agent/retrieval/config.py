# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Configuration models for the retrieval agent.

Provides a single Pydantic-based source of truth for all retrieval
agent settings, replacing scattered raw-dict ``.get()`` fallbacks.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RetrievalModelConfig(BaseModel):
    """Model configuration for the retrieval agent."""

    llm_model: str = Field(
        default="openai/openai/gpt-5.2",
        description="LLM model for agent reasoning",
    )
    llm_backend: str = Field(
        default="api",
        description="LLM backend: 'local' or 'api'",
    )
    embedding_model: str = Field(
        default="google/siglip2-base-patch16-256",
        description="Embedding model for semantic frame search",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for API backends (or set NIM_API_KEY env var)",
    )
    device: str = Field(default="cuda", description="Device for local models")

    vllm_url: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM OpenAI-compatible endpoint URL",
    )
    vllm_local_media: bool = Field(
        default=True,
        description="Use file:// URLs for media (fastest when co-located with vLLM)",
    )
    vllm_tp_size: int = Field(
        default=1,
        description="Tensor-parallel size for vLLM server",
    )
    vllm_gpu_memory_utilization: float = Field(
        default=0.8,
        description="Fraction of GPU memory for vLLM KV-cache",
    )


class AgentConfig(BaseModel):
    """Agent behaviour configuration."""

    max_steps: int = Field(default=10, description="Maximum reasoning steps before giving up")
    temperature: float = Field(
        default=0.0, description="Temperature for LLM generation (0.0 = deterministic)"
    )
    max_sub_tasks: int = Field(default=5, description="Maximum sub-tasks to decompose into")
    max_relaxation_levels: int = Field(
        default=3, description="Maximum search relaxation levels (0-3)"
    )
    max_search_attempts: int = Field(
        default=9, description="Maximum search attempts per task before giving up"
    )
    parallel_tasks: bool = Field(
        default=True,
        description="Run sub-tasks in parallel via LangGraph Send API (False = sequential)",
    )


class RetrievalDatabaseConfig(BaseModel):
    """Database paths for the retrieval agent."""

    directory: str = Field(
        default="outputs/",
        description="Directory containing graph.db and vector.db",
    )


class OutputConfig(BaseModel):
    """Output configuration for extracted clips."""

    clips_dir: str = Field(default="outputs/clips", description="Directory for extracted clips")
    clip_padding: float = Field(default=0.5, description="Padding around clip boundaries (seconds)")


class RetrievalLoggingConfig(BaseModel):
    """Logging configuration for the retrieval agent."""

    level: str = Field(default="INFO", description="Log level")
    save_traces: bool = Field(default=True, description="Save agent reasoning traces")
    traces_dir: str = Field(default="outputs/traces", description="Directory for saved traces")


class RetrievalConfig(BaseModel):
    """Top-level retrieval agent configuration."""

    models: RetrievalModelConfig = Field(default_factory=RetrievalModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    database: RetrievalDatabaseConfig = Field(default_factory=RetrievalDatabaseConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: RetrievalLoggingConfig = Field(default_factory=RetrievalLoggingConfig)


def load_retrieval_config(config_path: str | Path) -> RetrievalConfig:
    """Load retrieval configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        RetrievalConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        return RetrievalConfig(**raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}") from e
