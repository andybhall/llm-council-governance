"""Self-Consistency Baseline: Single model with multiple samples."""

import asyncio
from typing import Dict, List, Optional, Any

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.utils import (
    build_stage1_prompt,
    compute_vote_metadata,
    extract_final_answer,
)
from backend.openrouter import query_model


class SelfConsistencyVoteStructure(GovernanceStructure):
    """
    Self-Consistency Baseline: Query a single model multiple times with temperature > 0.

    This is a compute-matched baseline for comparing model diversity (councils)
    vs prompt diversity (repeated sampling). Uses the same number of API calls
    as council structures but queries a single model multiple times with stochastic
    sampling instead of querying multiple different models.

    The final answer is determined by majority vote over the sampled responses.
    """

    def __init__(
        self,
        base_model: str = "google/gemini-2.0-flash-001",
        n_samples: int = 11,
        temperature: float = 0.7,
        council_models: Optional[List[str]] = None,
        chairman_model: Optional[str] = None,
    ):
        """
        Initialize self-consistency baseline structure.

        Args:
            base_model: The model to sample from (default: best council model)
            n_samples: Number of samples to take (default: 11, matches 2N+1 for N=5)
            temperature: Sampling temperature (default: 0.7)
            council_models: Ignored, accepted for API compatibility
            chairman_model: Ignored, accepted for API compatibility
        """
        # Initialize parent with dummy values (we don't use council/chairman)
        super().__init__(
            council_models=council_models or [base_model],
            chairman_model=chairman_model or base_model,
        )
        self.base_model = base_model
        self.n_samples = n_samples
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "Self-Consistency Vote"

    async def run(self, query: str) -> CouncilResult:
        """Execute the self-consistency sampling and voting process."""
        # Stage 1: Sample the base model n_samples times with temperature
        stage1_responses = await self._stage1_sample_model(query)

        # Stage 2: Extract answers from each sample
        extracted_answers = self._stage2_extract_answers(stage1_responses)

        # Stage 3: Majority vote over samples
        final_answer, vote_metadata = self._stage3_vote(extracted_answers)

        return CouncilResult(
            final_answer=final_answer,
            stage1_responses=stage1_responses,
            stage2_data={
                "extracted_answers": extracted_answers,
                "base_model": self.base_model,
                "n_samples": self.n_samples,
                "temperature": self.temperature,
            },
            stage3_data={
                "vote_result": final_answer,
                **vote_metadata,
            },
        )

    async def _stage1_sample_model(self, query: str) -> Dict[str, str]:
        """
        Stage 1: Sample the base model n_samples times in parallel.

        Returns:
            Dict mapping sample_0, sample_1, ... to response texts
        """
        prompt = build_stage1_prompt(query)
        messages = [{"role": "user", "content": prompt}]

        # Create n_samples parallel queries
        tasks = [
            query_model(self.base_model, messages, temperature=self.temperature)
            for _ in range(self.n_samples)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = {}
        for i, result in enumerate(results):
            sample_key = f"sample_{i}"
            if isinstance(result, Exception):
                responses[sample_key] = f"Error: {result}"
            else:
                responses[sample_key] = result.get("content", "")

        return responses

    def _stage2_extract_answers(
        self, stage1_responses: Dict[str, str]
    ) -> Dict[str, Optional[str]]:
        """Stage 2: Extract final answer from each sample."""
        return {
            key: extract_final_answer(response)
            for key, response in stage1_responses.items()
        }

    def _stage3_vote(
        self, extracted_answers: Dict[str, Optional[str]]
    ) -> tuple[str, Dict[str, Any]]:
        """
        Stage 3: Majority vote over samples.

        Uses deterministic tie-breaking (alphabetical order) since there's
        no chairman in self-consistency voting.

        Returns:
            Tuple of (winning answer, vote_metadata)
        """
        # No tiebreaker - use deterministic alphabetical ordering
        return compute_vote_metadata(extracted_answers, tiebreaker=None)
