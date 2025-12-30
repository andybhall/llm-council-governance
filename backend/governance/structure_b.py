"""Backward compatibility shim - import from majority_vote instead.

DEPRECATED: This module has been renamed to majority_vote.py.
This shim is provided for backward compatibility.
"""

import warnings

warnings.warn(
    "backend.governance.structure_b is deprecated. "
    "Import from backend.governance.majority_vote instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location
from backend.governance.majority_vote import MajorityVoteStructure

__all__ = ["MajorityVoteStructure"]
