"""Governance structures for LLM council decision-making."""

from typing import Dict, Type

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.independent_rank_synthesize import IndependentRankSynthesize
from backend.governance.majority_vote import MajorityVoteStructure
from backend.governance.deliberate_vote import DeliberateVoteStructure
from backend.governance.deliberate_synthesize import DeliberateSynthesizeStructure
from backend.governance.weighted_vote import WeightedMajorityVote
from backend.governance.self_consistency_vote import SelfConsistencyVoteStructure
from backend.governance.utils import (
    extract_final_answer,
    extract_final_answer_with_fallback,
    majority_vote,
    majority_vote_normalized,
    normalize_answer,
    smart_weighted_majority_vote,
    weighted_majority_vote,
)
from backend.governance.voting import (
    VotingStrategy,
    MajorityVoteStrategy,
    WeightedMajorityVoteStrategy,
    OracleWeightedVoteStrategy,
    create_voting_strategies,
)


# Structure registry for configuration-driven experiments
STRUCTURES: Dict[str, Type[GovernanceStructure]] = {
    "rank_synthesize": IndependentRankSynthesize,
    "majority_vote": MajorityVoteStructure,
    "deliberate_vote": DeliberateVoteStructure,
    "deliberate_synthesize": DeliberateSynthesizeStructure,
    "weighted_vote": WeightedMajorityVote,
    "self_consistency": SelfConsistencyVoteStructure,
    # Aliases for backward compatibility
    "structure_a": IndependentRankSynthesize,
    "structure_b": MajorityVoteStructure,
    "structure_c": DeliberateVoteStructure,
    "structure_d": DeliberateSynthesizeStructure,
    "structure_e": WeightedMajorityVote,
}


def get_structure(name: str) -> Type[GovernanceStructure]:
    """
    Get a governance structure class by name.

    Args:
        name: Structure name (e.g., 'majority_vote', 'deliberate_vote')

    Returns:
        The governance structure class

    Raises:
        KeyError: If the structure name is not found
    """
    if name not in STRUCTURES:
        available = ", ".join(sorted(STRUCTURES.keys()))
        raise KeyError(f"Unknown structure '{name}'. Available: {available}")
    return STRUCTURES[name]


def list_structures() -> list[str]:
    """Return list of available structure names (excluding aliases)."""
    return [
        "rank_synthesize",
        "majority_vote",
        "deliberate_vote",
        "deliberate_synthesize",
        "weighted_vote",
        "self_consistency",
    ]

__all__ = [
    # Base classes
    "CouncilResult",
    "GovernanceStructure",
    # Full governance structures (with API calls)
    "IndependentRankSynthesize",
    "MajorityVoteStructure",
    "DeliberateVoteStructure",
    "DeliberateSynthesizeStructure",
    "WeightedMajorityVote",
    "SelfConsistencyVoteStructure",
    # Structure registry
    "STRUCTURES",
    "get_structure",
    "list_structures",
    # Voting strategies (pure voting logic, no API calls)
    "VotingStrategy",
    "MajorityVoteStrategy",
    "WeightedMajorityVoteStrategy",
    "OracleWeightedVoteStrategy",
    "create_voting_strategies",
    # Utility functions
    "extract_final_answer",
    "extract_final_answer_with_fallback",
    "majority_vote",
    "majority_vote_normalized",
    "normalize_answer",
    "smart_weighted_majority_vote",
    # Note: weighted_majority_vote is intentionally not exported.
    # Use smart_weighted_majority_vote instead, which auto-detects answer types.
]
