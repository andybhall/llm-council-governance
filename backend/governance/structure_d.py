"""Backward compatibility shim - import from deliberate_synthesize instead.

DEPRECATED: This module has been renamed to deliberate_synthesize.py.
This shim is provided for backward compatibility.
"""

import warnings

warnings.warn(
    "backend.governance.structure_d is deprecated. "
    "Import from backend.governance.deliberate_synthesize instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location
from backend.governance.deliberate_synthesize import DeliberateSynthesizeStructure

__all__ = ["DeliberateSynthesizeStructure"]
