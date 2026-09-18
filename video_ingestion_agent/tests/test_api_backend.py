# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Tests for the 'api' backend endpoint plumbing.

Covers the B.3 EVT bug class: the endpoint must be configurable
(models.api_url), each backend must receive its own URL, and auth
failures must fail fast with an actionable message.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from video_ingestion_agent.models.api_model import APIModel
from video_ingestion_agent.models.model_manager import ModelManager, get_api_model, resolve_api_url

API_URL = "https://gateway.example.com/v1/chat/completions"
VLLM_URL = "http://localhost:8000/v1"


class TestResolveApiUrl:
    """Backend-aware endpoint routing."""

    def test_vllm_backend_gets_vllm_url(self):
        url = resolve_api_url("vllm", vllm_url="http://localhost:8000/v1", api_url="https://x/v1")
        assert url == "http://localhost:8000/v1"

    def test_api_backend_gets_api_url(self):
        url = resolve_api_url("api", vllm_url="http://localhost:8000/v1", api_url="https://x/v1")
        assert url == "https://x/v1"

    def test_api_backend_defaults_to_none(self):
        # Routing preserves None; APIModel rejects it before making a request.
        assert resolve_api_url("api", vllm_url="http://localhost:8000/v1", api_url=None) is None

    def test_local_backend_gets_none(self):
        assert resolve_api_url("local", vllm_url="http://x/v1", api_url="https://y/v1") is None


class TestAPIModelEndpoint:
    """APIModel endpoint resolution and key sourcing."""

    @pytest.mark.parametrize("api_url", [None, "", "   "])
    def test_missing_endpoint_requires_provider_configuration(self, api_url):
        with pytest.raises(ValueError, match="models.api_url") as excinfo:
            APIModel(api_key="test-key", api_url=api_url)
        assert "model" in str(excinfo.value)

    def test_convenience_constructor_requires_endpoint(self):
        with pytest.raises(ValueError, match="models.api_url"):
            get_api_model("provider-model", api_key="test-key")

    def test_provider_endpoint_and_model_are_used_for_request(self):
        model = get_api_model("provider-model", api_key="test-key", api_url=API_URL)
        with patch("video_ingestion_agent.models.api_model.requests.post") as post:
            post.return_value.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            assert model.generate_text([{"role": "user", "content": "hi"}]) == "ok"
        assert post.call_args.args == (API_URL,)
        assert post.call_args.kwargs["json"]["model"] == "provider-model"
        ModelManager.reset()

    def test_endpoint_override(self):
        override = "https://gateway.example.com/v1/chat/completions"
        model = APIModel(api_key="test-key", api_url=override)
        assert model.api_url == override

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("NIM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="NIM_API_KEY"):
            APIModel(api_url=API_URL)

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("NIM_API_KEY", "env-key")
        model = APIModel(api_url=API_URL)
        assert model.api_key == "env-key"


class TestAuthFailFast:
    """401/403 must abort immediately with an actionable message."""

    @staticmethod
    def _http_error(status: int) -> requests.exceptions.HTTPError:
        response = MagicMock()
        response.status_code = status
        return requests.exceptions.HTTPError(response=response)

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_error_is_not_retried(self, status):
        model = APIModel(api_key="bad-key", api_url=API_URL)
        with patch("video_ingestion_agent.models.api_model.requests.post") as post:
            post.return_value.raise_for_status.side_effect = self._http_error(status)
            with pytest.raises(RuntimeError, match="Authentication failed"):
                model._make_request([{"role": "user", "content": "hi"}])
            assert post.call_count == 1  # no retries on auth errors

    def test_auth_error_message_names_endpoint_and_override(self):
        model = APIModel(api_key="bad-key", api_url=API_URL)
        with patch("video_ingestion_agent.models.api_model.requests.post") as post:
            post.return_value.raise_for_status.side_effect = self._http_error(401)
            with pytest.raises(RuntimeError) as excinfo:
                model._make_request([{"role": "user", "content": "hi"}])
        assert model.api_url in str(excinfo.value)
        assert "models.api_url" in str(excinfo.value)

    def test_transient_error_still_retries(self):
        model = APIModel(api_key="key", api_url=API_URL)
        with (
            patch("video_ingestion_agent.models.api_model.requests.post") as post,
            patch("video_ingestion_agent.models.api_model.time.sleep"),
        ):
            post.side_effect = requests.exceptions.ConnectionError("boom")
            with pytest.raises(RuntimeError, match="after 5 attempts"):
                model._make_request([{"role": "user", "content": "hi"}])
            assert post.call_count == 5


class TestModelManagerCacheKey:
    """Same model on two endpoints must not collide in the cache."""

    def test_distinct_api_urls_create_distinct_models(self):
        manager = ModelManager()
        manager._models.clear()
        with patch("video_ingestion_agent.models.model_manager.APIModelWrapper") as wrapper:
            wrapper.side_effect = lambda **kw: MagicMock(fps=kw.get("fps", 4))
            manager.get_model("m", backend="api", api_key="k", api_url="https://a/v1")
            manager.get_model("m", backend="api", api_key="k", api_url="https://b/v1")
            assert wrapper.call_count == 2
        manager._models.clear()

    def test_same_api_url_reuses_cached_model(self):
        manager = ModelManager()
        manager._models.clear()
        with patch("video_ingestion_agent.models.model_manager.APIModelWrapper") as wrapper:
            wrapper.side_effect = lambda **kw: MagicMock(fps=kw.get("fps", 4))
            manager.get_model("m", backend="api", api_key="k", api_url="https://a/v1")
            manager.get_model("m", backend="api", api_key="k", api_url="https://a/v1")
            assert wrapper.call_count == 1
        manager._models.clear()


@pytest.mark.parametrize("backend, expected_url", [("api", API_URL), ("vllm", VLLM_URL)])
@pytest.mark.parametrize("component", ["segmenter", "critic", "refiner"])
def test_ingestion_callers_preserve_backend_endpoint(backend, expected_url, component):
    from video_ingestion_agent.ingestion.config import ModelConfig, PipelineConfig
    from video_ingestion_agent.ingestion.segmentation.critic import Critic
    from video_ingestion_agent.ingestion.segmentation.segmenter import HybridSegmenter
    from video_ingestion_agent.ingestion.segmentation.strategies import ReannotateStrategy

    config = PipelineConfig(
        models=ModelConfig(
            vlm_model="configured-model",
            vlm_backend=backend,
            api_key="test-key",
            api_url=API_URL,
            vllm_url=VLLM_URL,
        )
    )
    cls = {"segmenter": HybridSegmenter, "critic": Critic, "refiner": ReannotateStrategy}[component]
    with patch.object(ModelManager, "get_model") as get_model:
        cls(config)._get_model()
    assert get_model.call_args.kwargs["api_url"] == expected_url
    assert get_model.call_args.kwargs["model_name"] == "configured-model"


@pytest.mark.parametrize("backend, expected_url", [("api", API_URL), ("vllm", VLLM_URL)])
def test_retrieval_caller_preserves_backend_endpoint(backend, expected_url):
    from video_ingestion_agent.retrieval.config import RetrievalConfig, RetrievalModelConfig
    from video_ingestion_agent.retrieval.nodes.task_decomposer import TaskDecomposerNode

    config = RetrievalConfig(
        models=RetrievalModelConfig(
            llm_model="configured-model",
            llm_backend=backend,
            api_key="test-key",
            api_url=API_URL,
            vllm_url=VLLM_URL,
        )
    )
    with patch.object(ModelManager, "get_model") as get_model:
        TaskDecomposerNode(config=config, tools={})._get_model()
    assert get_model.call_args.kwargs["api_url"] == expected_url
    assert get_model.call_args.kwargs["model_name"] == "configured-model"
