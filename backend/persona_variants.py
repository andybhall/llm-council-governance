"""Backward compatibility shim - import from experiments.assets.persona_variants instead.

DEPRECATED: This module has been moved to experiments/assets/persona_variants.py.
This shim is provided for backward compatibility.
"""

import warnings

warnings.warn(
    "backend.persona_variants is deprecated. "
    "Import from experiments.assets.persona_variants instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from experiments.assets.persona_variants import (
    BASE_MODEL,
    PERSONA_MODELS,
    PERSONA_TRANSFORMERS,
    is_persona_model,
    get_actual_model,
    transform_prompt_for_persona,
    transform_mathematician,
    transform_scientist,
    transform_engineer,
    transform_teacher,
    query_model_with_personas,
    query_models_parallel_with_personas,
)

# Also import original_query_model for backward compatibility with tests
from backend.openrouter import query_model as original_query_model

__all__ = [
    "BASE_MODEL",
    "PERSONA_MODELS",
    "PERSONA_TRANSFORMERS",
    "is_persona_model",
    "get_actual_model",
    "transform_prompt_for_persona",
    "transform_mathematician",
    "transform_scientist",
    "transform_engineer",
    "transform_teacher",
    "query_model_with_personas",
    "query_models_parallel_with_personas",
    "original_query_model",
]
