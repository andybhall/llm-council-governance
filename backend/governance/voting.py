"""Voting strategy abstraction for fair comparison of voting methods.

This module separates voting logic from API calls, enabling:
1. Fair comparison: Apply multiple voting strategies to the SAME Stage 1 responses
2. Efficiency: Run Stage 1 once, apply N voting strategies
3. Clean testing: Test voting logic without mocking API calls
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from backend.governance.utils import (
    extract_final_answer,
    smart_majority_vote,
    smart_weighted_majority_vote,
)


class VotingStrategy(ABC):
    """
    Abstract base class for voting strategies.

    A VotingStrategy takes pre-collected Stage 1 responses and applies
    a voting algorithm to determine the final answer. This separation
    allows fair comparison of different voting methods on identical inputs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this voting strategy."""
        pass

    @property
    def short_name(self) -> str:
        """Short name for charts/tables. Defaults to name."""
        return self.name

    @abstractmethod
    def vote(
        self,
        stage1_responses: Dict[str, str],
        chairman_answer: Optional[str] = None,
    ) -> str:
        """
        Apply voting strategy to pre-collected responses.

        Args:
            stage1_responses: Dict mapping model names to their full response text
            chairman_answer: Optional tiebreaker answer from chairman

        Returns:
            The winning answer string
        """
        pass

    def extract_answers(
        self, stage1_responses: Dict[str, str]
    ) -> Dict[str, Optional[str]]:
        """
        Extract final answers from Stage 1 responses.

        Args:
            stage1_responses: Dict mapping model names to full response text

        Returns:
            Dict mapping model names to extracted answers (or None if extraction failed)
        """
        return {
            model: extract_final_answer(response)
            for model, response in stage1_responses.items()
        }

    def get_valid_answers(
        self, stage1_responses: Dict[str, str]
    ) -> tuple[List[str], List[str]]:
        """
        Extract and filter valid answers from Stage 1 responses.

        Args:
            stage1_responses: Dict mapping model names to full response text

        Returns:
            Tuple of (valid_answers, valid_models) - parallel lists
        """
        extracted = self.extract_answers(stage1_responses)
        valid_pairs = [
            (ans, model)
            for model, ans in extracted.items()
            if ans is not None
        ]

        if not valid_pairs:
            return [], []

        valid_answers = [p[0] for p in valid_pairs]
        valid_models = [p[1] for p in valid_pairs]
        return valid_answers, valid_models


class MajorityVoteStrategy(VotingStrategy):
    """
    Simple majority vote: each model gets one equal vote.

    The answer with the most votes wins. In case of a tie,
    the chairman's answer is used as a tiebreaker.
    """

    @property
    def name(self) -> str:
        return "Majority Vote"

    @property
    def short_name(self) -> str:
        return "MV"

    def vote(
        self,
        stage1_responses: Dict[str, str],
        chairman_answer: Optional[str] = None,
    ) -> str:
        valid_answers, _ = self.get_valid_answers(stage1_responses)

        if not valid_answers:
            return chairman_answer or ""

        return smart_majority_vote(valid_answers, tiebreaker=chairman_answer)


class WeightedMajorityVoteStrategy(VotingStrategy):
    """
    Weighted majority vote: each model's vote is weighted by its historical accuracy.

    Models with higher accuracy have more influence on the final answer.
    Weights are typically accuracy rates from a previous evaluation run.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        weights_file: Optional[str] = None,
    ):
        """
        Initialize with weights.

        Args:
            weights: Dict mapping model names to weights (e.g., accuracy rates)
            weights_file: Path to JSON file containing weights
                         (used if weights is None)
        """
        self._weights = self._load_weights(weights, weights_file)
        self._weights_source = "provided" if weights else (weights_file or "default")

    def _load_weights(
        self,
        weights: Optional[Dict[str, float]],
        weights_file: Optional[str],
    ) -> Dict[str, float]:
        """Load weights from provided dict, file, or use empty dict (defaults to 1.0)."""
        if weights is not None:
            return weights

        if weights_file:
            weights_path = Path(weights_file)
            if weights_path.exists():
                try:
                    with open(weights_path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

        # Return empty dict - will default to 1.0 per model in vote()
        return {}

    @property
    def name(self) -> str:
        return "Weighted Majority Vote"

    @property
    def short_name(self) -> str:
        return "WMV"

    @property
    def weights(self) -> Dict[str, float]:
        """Return the current weights dictionary."""
        return self._weights

    def vote(
        self,
        stage1_responses: Dict[str, str],
        chairman_answer: Optional[str] = None,
    ) -> str:
        valid_answers, valid_models = self.get_valid_answers(stage1_responses)

        if not valid_answers:
            return chairman_answer or ""

        return smart_weighted_majority_vote(
            valid_answers,
            valid_models,
            self._weights,
            tiebreaker=chairman_answer,
        )

    def get_vote_details(
        self,
        stage1_responses: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Get detailed voting information for analysis.

        Returns dict with:
            - extracted_answers: Dict of model -> answer
            - weights_used: Dict of model -> weight applied
            - vote_totals: Dict of answer -> total weight
        """
        extracted = self.extract_answers(stage1_responses)

        weights_used = {
            model: self._weights.get(model, 1.0)
            for model in stage1_responses.keys()
        }

        # Calculate vote totals
        vote_totals: Dict[str, float] = {}
        for model, answer in extracted.items():
            if answer is None:
                continue
            weight = weights_used[model]
            if answer not in vote_totals:
                vote_totals[answer] = 0.0
            vote_totals[answer] += weight

        return {
            "extracted_answers": extracted,
            "weights_used": weights_used,
            "vote_totals": vote_totals,
        }


class OracleWeightedVoteStrategy(VotingStrategy):
    """
    Oracle weighted vote: weights are computed per-question based on correctness.

    This is for analysis purposes only - it uses knowledge of which models
    got the current question correct to weight votes. This represents an
    upper bound on how well weighted voting could perform with perfect
    model reliability estimation.
    """

    def __init__(self, correctness_map: Optional[Dict[str, Dict[str, bool]]] = None):
        """
        Initialize with correctness information.

        Args:
            correctness_map: Dict mapping question_id -> (model -> is_correct)
        """
        self._correctness_map = correctness_map or {}
        self._current_question_id: Optional[str] = None

    @property
    def name(self) -> str:
        return "Oracle Weighted Vote"

    @property
    def short_name(self) -> str:
        return "Oracle"

    def set_question(self, question_id: str) -> None:
        """Set the current question for oracle lookup."""
        self._current_question_id = question_id

    def vote(
        self,
        stage1_responses: Dict[str, str],
        chairman_answer: Optional[str] = None,
    ) -> str:
        valid_answers, valid_models = self.get_valid_answers(stage1_responses)

        if not valid_answers:
            return chairman_answer or ""

        # Get oracle weights for current question
        if self._current_question_id and self._current_question_id in self._correctness_map:
            correctness = self._correctness_map[self._current_question_id]
            # Weight correct models at 1.0, incorrect at 0.0
            weights = {
                model: 1.0 if correctness.get(model, False) else 0.0
                for model in valid_models
            }
        else:
            # Fall back to equal weights if no oracle data
            weights = {model: 1.0 for model in valid_models}

        return smart_weighted_majority_vote(
            valid_answers,
            valid_models,
            weights,
            tiebreaker=chairman_answer,
        )


# Convenience function to create standard voting strategies
def create_voting_strategies(
    weights_file: Optional[str] = None,
    include_oracle: bool = False,
) -> List[VotingStrategy]:
    """
    Create the standard set of voting strategies for comparison.

    Args:
        weights_file: Path to JSON file with model weights for weighted voting
        include_oracle: Whether to include the oracle strategy (for analysis)

    Returns:
        List of VotingStrategy instances
    """
    strategies = [
        MajorityVoteStrategy(),
        WeightedMajorityVoteStrategy(weights_file=weights_file),
    ]

    if include_oracle:
        strategies.append(OracleWeightedVoteStrategy())

    return strategies
