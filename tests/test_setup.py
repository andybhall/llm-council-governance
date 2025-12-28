"""Tests to verify project setup and imports work correctly."""

import pytest


def test_config_loads():
    """Verify config.py loads without errors."""
    from backend.config import COUNCIL_MODELS, CHAIRMAN_MODEL

    assert len(COUNCIL_MODELS) == 4
    assert CHAIRMAN_MODEL is not None


def test_config_has_required_values():
    """Verify config has all required configuration values."""
    from backend.config import (
        COUNCIL_MODELS,
        CHAIRMAN_MODEL,
        OPENROUTER_API_URL,
    )

    assert all(isinstance(model, str) for model in COUNCIL_MODELS)
    assert isinstance(CHAIRMAN_MODEL, str)
    assert OPENROUTER_API_URL.startswith("https://")


def test_openrouter_imports():
    """Verify openrouter module imports correctly."""
    from backend.openrouter import query_model, query_models_parallel

    assert callable(query_model)
    assert callable(query_models_parallel)


@pytest.mark.asyncio
async def test_query_model_with_mock(monkeypatch):
    """Verify query_model works with mocked HTTP client."""
    from backend import openrouter

    # Reset shared client before test
    openrouter._shared_client = None

    # Mock response data
    mock_response_data = {
        "choices": [{"message": {"content": "Mock response. FINAL ANSWER: 42"}}]
    }

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_response_data

    class MockAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aclose(self):
            pass

        async def post(self, *args, **kwargs):
            return MockResponse()

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", MockAsyncClient)

    result = await openrouter.query_model(
        "test-model", [{"role": "user", "content": "What is 2+2?"}]
    )

    assert result["content"] == "Mock response. FINAL ANSWER: 42"

    # Clean up
    openrouter._shared_client = None


@pytest.mark.asyncio
async def test_query_models_parallel_with_mock(monkeypatch):
    """Verify query_models_parallel works with mocked HTTP client."""
    from backend import openrouter

    # Reset shared client before test
    openrouter._shared_client = None

    call_count = 0

    class MockResponse:
        def __init__(self, model_name):
            self.model_name = model_name

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"content": f"Response from {self.model_name}"}}
                ]
            }

    class MockAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aclose(self):
            pass

        async def post(self, url, headers=None, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            model_name = json.get("model", "unknown")
            return MockResponse(model_name)

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", MockAsyncClient)

    models = ["model-a", "model-b"]
    result = await openrouter.query_models_parallel(
        models, [{"role": "user", "content": "Test query"}]
    )

    assert len(result) == 2
    assert "model-a" in result
    assert "model-b" in result
    assert result["model-a"]["content"] == "Response from model-a"
    assert result["model-b"]["content"] == "Response from model-b"
    assert call_count == 2  # Verify both models were queried

    # Clean up
    openrouter._shared_client = None
