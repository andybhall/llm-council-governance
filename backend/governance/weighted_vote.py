"""Structure E: Independent → Weighted Majority Vote."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.utils import (
    compute_weighted_vote_metadata,
    extract_final_answer,
)


class WeightedMajorityVote(GovernanceStructure):
    """
    Governance Structure E: Independent → Weighted Majority Vote.

    Similar to Structure B (Majority Vote) but weights each model's vote
    by its historical accuracy rate. Models with higher accuracy have
    more influence on the final answer.

    Stage 1: Each LLM answers independently (with FINAL ANSWER instruction)
    Stage 2: Extract final answer from each response
    Stage 3: Weighted majority vote determines winner (chairman as tiebreaker)
    """

    def __init__(
        self,
        council_models: List[str],
        chairman_model: str,
        weights: Optional[Dict[str, float]] = None,
        weights_file: Optional[str] = None,
    ):
        """
        Initialize weighted majority vote structure.

        Args:
            council_models: List of model identifiers for council members
            chairman_model: Model identifier for the chairman
            weights: Optional dictionary mapping model names to weights.
                     If not provided, loads from weights_file or uses equal weights.
            weights_file: Optional path to JSON file containing weights.
                          Default: "experiments/results/model_weights.json"
        """
        super().__init__(council_models, chairman_model)
        self._weights = self._load_weights(weights, weights_file)

    def _load_weights(
        self,
        weights: Optional[Dict[str, float]],
        weights_file: Optional[str],
    ) -> Dict[str, float]:
        """Load weights from provided dict, file, or use defaults."""
        # Use provided weights if available
        if weights is not None:
            return weights

        # Try to load from file
        if weights_file is None:
            weights_file = "experiments/results/model_weights.json"

        weights_path = Path(weights_file)
        if weights_path.exists():
            try:
                with open(weights_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Default: equal weights of 1.0
        return {model: 1.0 for model in self.council_models}

    @property
    def name(self) -> str:
        return "Independent → Weighted Majority Vote"

    @property
    def weights(self) -> Dict[str, float]:
        """Return the current weights dictionary."""
        return self._weights

    async def run(self, query: str) -> CouncilResult:
        """Execute the weighted voting governance process."""
        # Stage 1: Collect independent responses with FINAL ANSWER instruction
        stage1_responses = await self._stage1_collect_responses(query)

        # Stage 2: Extract answers from each response
        extracted_answers = self._stage2_extract_answers(stage1_responses)

        # Stage 3: Weighted majority vote
        final_answer, chairman_answer, vote_metadata = await self._stage3_weighted_vote(
            query, extracted_answers
        )

        return CouncilResult(
            final_answer=final_answer,
            stage1_responses=stage1_responses,
            stage2_data={
                "extracted_answers": extracted_answers,
                "weights_used": {
                    model: self._weights.get(model, 1.0)
                    for model in extracted_answers.keys()
                },
            },
            stage3_data={
                "chairman_tiebreaker": chairman_answer,
                "vote_result": final_answer,
                **vote_metadata,
            },
        )

    def _stage2_extract_answers(
        self, stage1_responses: Dict[str, str]
    ) -> Dict[str, Optional[str]]:
        """Stage 2: Extract final answer from each response."""
        return {
            model: extract_final_answer(response)
            for model, response in stage1_responses.items()
        }

    async def _stage3_weighted_vote(
        self, query: str, extracted_answers: Dict[str, Optional[str]]
    ) -> tuple[str, Optional[str], Dict]:
        """
        Stage 3: Perform weighted majority vote with chairman as tiebreaker.

        Returns:
            Tuple of (winning answer, chairman's answer for tiebreaker, vote_metadata)
        """
        # Prepare parallel lists for weighted voting
        models = list(extracted_answers.keys())
        answers = [extracted_answers[m] for m in models]

        # Filter out None answers
        valid_pairs = [
            (ans, model)
            for ans, model in zip(answers, models)
            if ans is not None
        ]

        if not valid_pairs:
            # No valid answers extracted, ask chairman directly
            chairman_answer = await self._get_chairman_answer(query)
            empty_metadata = {
                "raw_answers": {m: a for m, a in extracted_answers.items()},
                "normalized_answers": {m: None for m in extracted_answers.keys()},
                "vote_counts": {},
                "is_tie": False,
                "winning_answer": chairman_answer,
                "tiebreaker_used": False,
            }
            return chairman_answer, chairman_answer, empty_metadata

        # Get chairman's answer for tiebreaker
        chairman_answer = await self._get_chairman_answer(query)

        # Perform weighted majority vote with metadata
        final_answer, vote_metadata = compute_weighted_vote_metadata(
            extracted_answers, self._weights, tiebreaker=chairman_answer
        )

        return final_answer, chairman_answer, vote_metadata
