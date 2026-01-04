"""OpenRouter API client for LLM queries."""

import asyncio
import logging
from typing import List, Dict, Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

from backend.config import DEFAULT_API_TIMEOUT, OPENROUTER_API_URL, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

# Shared HTTP client for connection pooling (10-20x fewer TCP handshakes)
_shared_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


async def get_shared_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client with connection pooling."""
    global _shared_client
    if _shared_client is None:
        async with _client_lock:
            # Double-check after acquiring lock
            if _shared_client is None:
                _shared_client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=20,  # Conservative limit to avoid overwhelming API
                        max_keepalive_connections=10,
                    ),
                    timeout=httpx.Timeout(
                        connect=60.0,  # Connection timeout
                        read=DEFAULT_API_TIMEOUT,  # Read timeout (180s)
                        write=60.0,  # Write timeout
                        pool=120.0,  # Pool timeout
                    ),
                )
    return _shared_client


async def close_shared_client() -> None:
    """Close the shared HTTP client. Call this when done with all requests."""
    global _shared_client
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None


def _is_retryable_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry.

    Only retry on:
    - Timeouts
    - Rate limits (429)
    - Server errors (5xx)

    Do NOT retry on:
    - Client errors (400, 401, 403, 404) - these won't be fixed by retrying
    """
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        status = exception.response.status_code
        # Retry on rate limits and server errors
        return status == 429 or status >= 500
    return False


@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception(_is_retryable_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _make_api_request(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    """Make API request with retry logic for transient failures."""
    # Use explicit timeout object for per-request override
    request_timeout = httpx.Timeout(
        connect=30.0,  # Connection timeout
        read=timeout,  # Main API timeout
        write=30.0,  # Write timeout
        pool=60.0,  # Pool timeout for waiting on connections
    )

    async def do_request():
        response = await client.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        return response.json()

    # Hard asyncio timeout as a fallback (timeout + 30s buffer)
    try:
        return await asyncio.wait_for(do_request(), timeout=timeout + 30)
    except asyncio.TimeoutError:
        # Convert to httpx exception for retry logic
        raise httpx.ReadTimeout(f"Hard timeout after {timeout + 30}s")


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: Model identifier (e.g., "openai/gpt-4.1")
        messages: List of message dicts with 'role' and 'content' keys
        temperature: Sampling temperature (0.0 for deterministic outputs)
        timeout: Request timeout in seconds (uses DEFAULT_API_TIMEOUT if None)

    Returns:
        Dict containing 'content' key with the model's response text
    """
    timeout = timeout if timeout is not None else DEFAULT_API_TIMEOUT

    # Use fresh client for each request to avoid stale connection issues
    # in long-running experiments
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        data = await _make_api_request(client, model, messages, temperature, timeout)

    # Extract content from OpenRouter response format
    content = data["choices"][0]["message"]["content"]
    return {"content": content}


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple models in parallel via OpenRouter API.

    Args:
        models: List of model identifiers
        messages: List of message dicts (same messages sent to all models)
        temperature: Sampling temperature (0.0 for deterministic outputs)
        timeout: Request timeout in seconds (uses DEFAULT_API_TIMEOUT if None)

    Returns:
        Dict mapping model name to response dict
    """
    tasks = [query_model(model, messages, temperature, timeout) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        model: result if not isinstance(result, Exception) else {"error": str(result)}
        for model, result in zip(models, results)
    }
