"""Backward compatibility shim - import from weighted_vote instead.

DEPRECATED: This module has been renamed to weighted_vote.py.
This shim is provided for backward compatibility.
"""

import warnings

warnings.warn(
    "backend.governance.structure_e is deprecated. "
    "Import from backend.governance.weighted_vote instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location
from backend.governance.weighted_vote import WeightedMajorityVote

__all__ = ["WeightedMajorityVote"]
