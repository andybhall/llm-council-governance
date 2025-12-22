"""OpenRouter API client for LLM queries."""

import asyncio
from typing import List, Dict, Any

import httpx

from backend.config import OPENROUTER_API_URL, OPENROUTER_API_KEY


async def query_model(model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: Model identifier (e.g., "openai/gpt-4.1")
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Dict containing 'content' key with the model's response text
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        # Extract content from OpenRouter response format
        content = data["choices"][0]["message"]["content"]
        return {"content": content}


async def query_models_parallel(
    models: List[str], messages: List[Dict[str, str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple models in parallel via OpenRouter API.

    Args:
        models: List of model identifiers
        messages: List of message dicts (same messages sent to all models)

    Returns:
        Dict mapping model name to response dict
    """
    tasks = [query_model(model, messages) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        model: result if not isinstance(result, Exception) else {"error": str(result)}
        for model, result in zip(models, results)
    }
