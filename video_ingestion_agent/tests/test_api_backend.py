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
from video_ingestion_agent.models.model_manager import ModelManager, resolve_api_url


class TestResolveApiUrl:
    """Backend-aware endpoint routing."""

    def test_vllm_backend_gets_vllm_url(self):
        url = resolve_api_url("vllm", vllm_url="http://localhost:8000/v1", api_url="https://x/v1")
        assert url == "http://localhost:8000/v1"

    def test_api_backend_gets_api_url(self):
        url = resolve_api_url("api", vllm_url="http://localhost:8000/v1", api_url="https://x/v1")
        assert url == "https://x/v1"

    def test_api_backend_defaults_to_none(self):
        # None lets APIModel fall back to its built-in endpoint
        assert resolve_api_url("api", vllm_url="http://localhost:8000/v1", api_url=None) is None

    def test_local_backend_gets_none(self):
        assert resolve_api_url("local", vllm_url="http://x/v1", api_url="https://y/v1") is None


class TestAPIModelEndpoint:
    """APIModel endpoint resolution and key sourcing."""

    def test_default_endpoint_used_when_url_omitted(self):
        model = APIModel(api_key="test-key")
        assert model.api_url == APIModel.DEFAULT_API_URL
        assert "inference-api.nvidia.com" in model.api_url

    def test_endpoint_override(self):
        override = "https://gateway.example.com/v1/chat/completions"
        model = APIModel(api_key="test-key", api_url=override)
        assert model.api_url == override

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("NIM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="NIM_API_KEY"):
            APIModel()

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("NIM_API_KEY", "env-key")
        model = APIModel()
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
        model = APIModel(api_key="bad-key")
        with patch("video_ingestion_agent.models.api_model.requests.post") as post:
            post.return_value.raise_for_status.side_effect = self._http_error(status)
            with pytest.raises(RuntimeError, match="Authentication failed"):
                model._make_request([{"role": "user", "content": "hi"}])
            assert post.call_count == 1  # no retries on auth errors

    def test_auth_error_message_names_endpoint_and_override(self):
        model = APIModel(api_key="bad-key")
        with patch("video_ingestion_agent.models.api_model.requests.post") as post:
            post.return_value.raise_for_status.side_effect = self._http_error(401)
            with pytest.raises(RuntimeError) as excinfo:
                model._make_request([{"role": "user", "content": "hi"}])
        assert model.api_url in str(excinfo.value)
        assert "models.api_url" in str(excinfo.value)

    def test_transient_error_still_retries(self):
        model = APIModel(api_key="key")
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
