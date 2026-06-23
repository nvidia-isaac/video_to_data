# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Configuration models for the unified ingestion + entity graph pipeline.

Pydantic configuration models covering segmentation, verification, refinement,
and entity graph building.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SegmentationConfig(BaseModel):
    """Segmentation configuration (hybrid: chunk-based + VLM prompts)."""

    # Chunking parameters (from video_ingestion_agent)
    chunk_size: float = Field(default=15.0, description="Chunk size in seconds for VLM processing")
    chunk_overlap: float = Field(default=1.5, description="Overlap between chunks in seconds")

    # Segment constraints
    min_clip_s: float = Field(default=1.0, description="Minimum clip duration in seconds")
    max_clip_s: float = Field(default=30.0, description="Maximum clip duration in seconds")

    # VLM prompts
    system_prompt: str = Field(default="", description="System prompt for segmentation VLM")
    user_prompt: str = Field(default="", description="User prompt for segmentation VLM")

    # Overlap-based merge dedup
    dedup_method: Literal["heuristic", "llm"] = Field(
        default="llm",
        description=(
            "Dedup merge strategy: 'heuristic' always merges overlapping clips "
            "keeping the longer clip's annotations; 'llm' asks a language model "
            "whether overlapping clips describe the same action before merging."
        ),
    )
    dedup_overlap_threshold: float | None = Field(
        default=-0.1,
        description=(
            "Minimum temporal overlap in seconds to trigger merging of two clips. "
            "Positive values require actual overlap; negative values also consider "
            "clips separated by a small gap (e.g. -0.1 merges clips up to 0.1s "
            "apart). Set to None to disable merging."
        ),
    )

    # Video scanning
    video_extensions: list[str] = Field(
        default=[".mp4", ".mov", ".mkv"], description="Supported video file extensions"
    )


class VerificationConfig(BaseModel):
    """Verification (critic) configuration."""

    system_prompt: str = Field(
        default="",
        description="System prompt for critic VLM (empty = use default from prompts.py)",
    )
    user_prompt: str = Field(
        default="",
        description=(
            "User prompt template for critic (empty = use default from prompts.py). "
            "Can include {object}, {action}, {description}, {duration} placeholders."
        ),
    )
    max_iterations: int = Field(default=3, description="Maximum number of verify-refine iterations")


class ModelConfig(BaseModel):
    """Model configuration."""

    vlm_model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct", description="VLM model for segmentation/verification"
    )
    vlm_backend: str = Field(default="local", description="VLM backend: 'local', 'api', or 'vllm'")
    vlm_fps: int = Field(default=4, description="Frame sampling rate for VLM")

    llm_model: str | None = Field(
        default=None,
        description="LLM model for entity extraction (defaults to vlm_model if None)",
    )
    llm_backend: str = Field(default="local", description="LLM backend: 'local' or 'api'")

    embedding_model: str = Field(
        default="google/siglip2-base-patch16-256",
        description="Embedding model for frame features (SigLIP-2)",
    )
    embedding_batch_size: int = Field(
        default=16,
        description=(
            "Batch size for frame embedding extraction. Lower values reduce VRAM "
            "use (e.g. 8 or 4) when running parallel workers or on smaller GPUs."
        ),
    )

    device: str = Field(default="cuda", description="Device for local models")
    api_key: str | None = Field(default=None, description="API key for API backends")

    # vLLM-specific settings
    vllm_url: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM server URL (OpenAI-compatible endpoint)",
    )
    vllm_tp_size: int = Field(
        default=1,
        description=(
            "Tensor parallel size for vLLM server. Set to the number of GPUs "
            "to shard the model across (e.g. 8 for 8xH100). Only used by "
            "scripts/serve.py when starting the server."
        ),
    )
    vllm_gpu_memory_utilization: float = Field(
        default=0.8,
        description=("Fraction of GPU memory vLLM may use (0.0-1.0). Default 0.8 "),
    )
    vllm_local_media: bool = Field(
        default=True,
        description=(
            "When using vLLM backend, send video as file:// URL for server-side "
            "processing (fastest). Set to False to extract frames client-side "
            "and send as base64 (for remote vLLM servers)."
        ),
    )


class EntityExtractionConfig(BaseModel):
    """Entity extraction configuration."""

    max_time_gap: float = Field(
        default=30.0, description="Maximum time gap for entity merging in seconds"
    )
    min_entity_confidence: float = Field(default=0.5, description="Minimum entity confidence")
    min_relationship_confidence: float = Field(
        default=0.5, description="Minimum relationship confidence"
    )


class PathsConfig(BaseModel):
    """Paths configuration."""

    input_videos_dir: str = Field(default="data/videos", description="Input videos directory")
    runs_dir: str = Field(default="runs", description="Directory for run outputs")


class DatabaseConfig(BaseModel):
    """Database configuration."""

    directory: str = Field(default="outputs/", description="Output directory for databases")
    embedding_dim: int = Field(default=768, description="Embedding dimension for vector DB")
    graph_db_path: str | None = Field(
        default=None,
        description="Explicit graph DB path (overrides directory/graph.db). Used by batch ingestion for shared DB.",
    )
    vector_db_path: str | None = Field(
        default=None,
        description="Explicit vector DB path (overrides directory/vector.db). Used by batch ingestion for shared DB.",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    save_responses: bool = Field(default=True, description="Save model responses for debugging")
    response_dir: str = Field(
        default="outputs/debug/entity_extraction",
        description="Directory for saved responses",
    )


class ProcessingConfig(BaseModel):
    """Processing configuration for frame extraction."""

    fps: float = Field(default=1.0, description="Frame sampling rate for embeddings")


class PipelineConfig(BaseModel):
    """Unified pipeline configuration for ingestion + entity graph."""

    experiment_name: str = Field(
        default="default",
        description="Experiment name for tracking (used as W&B run name, output subdirectory, etc.)",
    )

    models: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    entity_extraction: EntityExtractionConfig = Field(default_factory=EntityExtractionConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Pipeline feature toggles
    enable_verification: bool = Field(default=True, description="Enable verification step")
    enable_refinement: bool = Field(default=True, description="Enable refinement loop")
    enable_entity_graph: bool = Field(default=True, description="Enable entity graph building")
    enable_reporting: bool = Field(default=True, description="Enable HTML report generation")


def load_config(config_path: str | Path) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        PipelineConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        return PipelineConfig(**raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}") from e
