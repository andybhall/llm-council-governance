"""Backward compatibility shim - import from deliberate_vote instead.

DEPRECATED: This module has been renamed to deliberate_vote.py.
This shim is provided for backward compatibility.
"""

import warnings

warnings.warn(
    "backend.governance.structure_c is deprecated. "
    "Import from backend.governance.deliberate_vote instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location
from backend.governance.deliberate_vote import DeliberateVoteStructure

__all__ = ["DeliberateVoteStructure"]
