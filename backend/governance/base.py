"""Base classes for governance structures."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from backend.governance.utils import (
    build_stage1_prompt,
    extract_final_answer,
    extract_final_answer_with_fallback,
)
from backend.openrouter import query_model, query_models_parallel

logger = logging.getLogger(__name__)


@dataclass
class CouncilResult:
    """Result from running a governance structure."""

    final_answer: str
    stage1_responses: Dict[str, str]  # model -> response
    stage2_data: Optional[Any] = None  # Structure-specific
    stage3_data: Optional[Any] = None  # Structure-specific
    metadata: Optional[Dict[str, Any]] = None  # Timings, token counts, etc.


class GovernanceStructure(ABC):
    """Abstract base class for governance structures."""

    def __init__(self, council_models: List[str], chairman_model: str):
        self.council_models = council_models
        self.chairman_model = chairman_model

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this structure."""
        pass

    @abstractmethod
    async def run(self, query: str) -> CouncilResult:
        """Execute the governance process and return result."""
        pass

    async def _stage1_collect_responses(self, query: str) -> Dict[str, str]:
        """
        Stage 1: Query all council models in parallel with standardized prompt.

        This is the common first stage for all governance structures.

        Args:
            query: The question/prompt to ask the council

        Returns:
            Dictionary mapping model names to their responses
        """
        prompt = build_stage1_prompt(query)
        messages = [{"role": "user", "content": prompt}]
        results = await query_models_parallel(self.council_models, messages)

        return {
            model: result.get("content", result.get("error", ""))
            for model, result in results.items()
        }

    async def _get_chairman_answer(self, query: str) -> str:
        """
        Get chairman's answer for tiebreaker.

        This is used by voting structures (B, C, E) to break ties.

        Args:
            query: The question/prompt to ask the chairman

        Returns:
            The extracted final answer from the chairman's response
        """
        prompt = build_stage1_prompt(query)
        messages = [{"role": "user", "content": prompt}]
        result = await query_model(self.chairman_model, messages)

        content = result.get("content", "")
        answer = extract_final_answer(content)

        if answer is None:
            logger.warning(
                "Chairman tiebreaker extraction failed for model %s, using fallback",
                self.chairman_model,
            )
            return extract_final_answer_with_fallback(content)

        return answer
