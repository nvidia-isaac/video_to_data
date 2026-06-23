# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for retrieval agent configuration."""

from pathlib import Path

import pytest


class TestRetrievalConfigDefaults:
    """Test RetrievalConfig default values match retrieval.yaml expectations."""

    def test_default_model_config(self):
        from video_ingestion_agent.retrieval.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.models.llm_model == "openai/openai/gpt-5.2"
        assert config.models.llm_backend == "api"
        assert config.models.embedding_model == "google/siglip2-base-patch16-256"
        assert config.models.api_key is None
        assert config.models.device == "cuda"

    def test_default_agent_config(self):
        from video_ingestion_agent.retrieval.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.agent.max_steps == 10
        assert config.agent.temperature == 0.0
        assert config.agent.max_sub_tasks == 5
        assert config.agent.max_relaxation_levels == 3
        assert config.agent.max_search_attempts == 9
        assert config.agent.parallel_tasks is True

    def test_default_database_config(self):
        from video_ingestion_agent.retrieval.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.database.directory == "outputs/"

    def test_default_output_config(self):
        from video_ingestion_agent.retrieval.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.output.clips_dir == "outputs/clips"
        assert config.output.clip_padding == 0.5

    def test_default_logging_config(self):
        from video_ingestion_agent.retrieval.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.logging.level == "INFO"
        assert config.logging.save_traces is True
        assert config.logging.traces_dir == "outputs/traces"


class TestRetrievalConfigLoading:
    """Test loading RetrievalConfig from YAML files."""

    def test_load_from_yaml(self):
        from video_ingestion_agent.retrieval.config import load_retrieval_config

        config_path = Path(__file__).parent.parent / "configs" / "retrieval.yaml"
        if not config_path.exists():
            pytest.skip("retrieval.yaml not found")

        config = load_retrieval_config(config_path)
        assert config.models.llm_model == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.models.llm_backend == "vllm"
        assert config.database.directory == "outputs/"
        assert config.output.clips_dir == "outputs/clips"

    def test_load_missing_file(self):
        from video_ingestion_agent.retrieval.config import load_retrieval_config

        with pytest.raises(FileNotFoundError):
            load_retrieval_config("/nonexistent/config.yaml")


class TestRetrievalConfigOverrides:
    """Test that config overrides work correctly."""

    def test_override_model(self):
        from video_ingestion_agent.retrieval.config import RetrievalConfig, RetrievalModelConfig

        config = RetrievalConfig(
            models=RetrievalModelConfig(
                llm_model="nvidia/Cosmos-Reason2-8B",
                llm_backend="local",
            )
        )
        assert config.models.llm_model == "nvidia/Cosmos-Reason2-8B"
        assert config.models.llm_backend == "local"
        assert config.models.embedding_model == "google/siglip2-base-patch16-256"

    def test_model_copy_update(self):
        from video_ingestion_agent.retrieval.config import RetrievalConfig

        config = RetrievalConfig()
        updated = config.model_copy(
            update={"agent": config.agent.model_copy(update={"max_sub_tasks": 10})}
        )
        assert updated.agent.max_sub_tasks == 10
        assert config.agent.max_sub_tasks == 5  # original unchanged

    def test_model_dump_roundtrip(self):
        from video_ingestion_agent.retrieval.config import RetrievalConfig

        config = RetrievalConfig()
        dumped = config.model_dump()
        assert dumped["models"]["llm_model"] == "openai/openai/gpt-5.2"
        assert dumped["agent"]["max_steps"] == 10
        assert dumped["agent"]["parallel_tasks"] is True
        assert dumped["database"]["directory"] == "outputs/"

        restored = RetrievalConfig(**dumped)
        assert restored == config


class TestRetrievalConfigImports:
    """Test that config types are exported correctly."""

    def test_import_from_retrieval_package(self):
        from video_ingestion_agent.retrieval import (  # noqa: F401
            RetrievalConfig,
            load_retrieval_config,
        )

    def test_import_from_config_module(self):
        from video_ingestion_agent.retrieval.config import (  # noqa: F401
            AgentConfig,
            OutputConfig,
            RetrievalConfig,
            RetrievalDatabaseConfig,
            RetrievalLoggingConfig,
            RetrievalModelConfig,
            load_retrieval_config,
        )
