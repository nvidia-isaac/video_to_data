"""Explicit deployment settings for tests that mock external submissions."""
import pytest


@pytest.fixture(autouse=True)
def configured_test_registry(monkeypatch):
    monkeypatch.setenv("V2D_IMAGE_REGISTRY", "registry.example.com/test-pipelines")
