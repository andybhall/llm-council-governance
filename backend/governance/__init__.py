"""Governance structures for LLM council decision-making."""

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.independent_rank_synthesize import IndependentRankSynthesize
from backend.governance.structure_b import MajorityVoteStructure
from backend.governance.structure_c import DeliberateVoteStructure
from backend.governance.structure_d import DeliberateSynthesizeStructure
from backend.governance.structure_e import WeightedMajorityVote
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
    "weighted_majority_vote",
]
