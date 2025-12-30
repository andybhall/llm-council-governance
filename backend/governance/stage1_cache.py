"""Stage 1 response caching for fair comparisons across governance structures."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class Stage1Cache:
    """
    Cache for Stage 1 LLM responses.

    This enables fair comparisons between governance structures by ensuring
    they all use the same Stage 1 responses for a given question/model/params
    combination.

    Cache keys are based on:
    - benchmark name
    - question_id
    - model name
    - prompt content (hashed)
    - LLM parameters (temperature, etc.)
    """

    def __init__(self, cache_dir: str = "experiments/.cache/stage1"):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cached responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(
        self,
        benchmark: str,
        question_id: str,
        model: str,
        prompt: str,
        params: Dict[str, Any],
    ) -> str:
        """Compute a unique cache key from the inputs."""
        # Create a deterministic string representation
        content = json.dumps(
            {
                "benchmark": benchmark,
                "question_id": question_id,
                "model": model,
                "prompt": prompt,
                "params": params,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.json"

    def get(
        self,
        benchmark: str,
        question_id: str,
        model: str,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get a cached response if it exists.

        Args:
            benchmark: Benchmark name (e.g., "GSM8K")
            question_id: Question identifier
            model: Model name
            prompt: The prompt text
            params: LLM parameters (temperature, etc.)

        Returns:
            Cached response string, or None if not cached
        """
        params = params or {"temperature": 0.0}
        key = self._compute_key(benchmark, question_id, model, prompt, params)
        path = self._get_path(key)

        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("response")
            except (json.JSONDecodeError, KeyError):
                return None

        return None

    def set(
        self,
        benchmark: str,
        question_id: str,
        model: str,
        prompt: str,
        response: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache a response.

        Args:
            benchmark: Benchmark name
            question_id: Question identifier
            model: Model name
            prompt: The prompt text
            response: The model's response
            params: LLM parameters
        """
        params = params or {"temperature": 0.0}
        key = self._compute_key(benchmark, question_id, model, prompt, params)
        path = self._get_path(key)

        data = {
            "benchmark": benchmark,
            "question_id": question_id,
            "model": model,
            "prompt": prompt,
            "params": params,
            "response": response,
        }

        path.write_text(json.dumps(data, indent=2))

    def get_all_for_question(
        self,
        benchmark: str,
        question_id: str,
        models: List[str],
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Get cached responses for all models for a given question.

        Args:
            benchmark: Benchmark name
            question_id: Question identifier
            models: List of model names
            prompt: The prompt text
            params: LLM parameters

        Returns:
            Dict mapping model names to responses (None if not cached)
        """
        return {
            model: self.get(benchmark, question_id, model, prompt, params)
            for model in models
        }

    def has_all(
        self,
        benchmark: str,
        question_id: str,
        models: List[str],
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if all models have cached responses for a question.

        Args:
            benchmark: Benchmark name
            question_id: Question identifier
            models: List of model names
            prompt: The prompt text
            params: LLM parameters

        Returns:
            True if all models have cached responses
        """
        responses = self.get_all_for_question(
            benchmark, question_id, models, prompt, params
        )
        return all(r is not None for r in responses.values())

    def clear(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of files deleted
        """
        count = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for file in subdir.iterdir():
                    if file.suffix == ".json":
                        file.unlink()
                        count += 1
                # Remove empty subdirectory
                if not any(subdir.iterdir()):
                    subdir.rmdir()
        return count

    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (total_entries, size_bytes)
        """
        total_entries = 0
        total_size = 0

        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for file in subdir.iterdir():
                    if file.suffix == ".json":
                        total_entries += 1
                        total_size += file.stat().st_size

        return {
            "total_entries": total_entries,
            "size_bytes": total_size,
        }


# Global cache instance for convenience
_default_cache: Optional[Stage1Cache] = None


def get_cache(cache_dir: Optional[str] = None) -> Stage1Cache:
    """Get the default Stage 1 cache instance."""
    global _default_cache
    if _default_cache is None or cache_dir is not None:
        _default_cache = Stage1Cache(cache_dir or "experiments/.cache/stage1")
    return _default_cache
