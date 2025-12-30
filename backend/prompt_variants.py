"""Backward compatibility shim - import from experiments.assets.prompt_variants instead.

DEPRECATED: This module has been moved to experiments/assets/prompt_variants.py.
This shim is provided for backward compatibility.
"""

import warnings

warnings.warn(
    "backend.prompt_variants is deprecated. "
    "Import from experiments.assets.prompt_variants instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from experiments.assets.prompt_variants import (
    BASE_MODEL,
    VARIANT_MODELS,
    VARIANT_TRANSFORMERS,
    is_prompt_variant_model,
    get_actual_model,
    transform_prompt_for_variant,
    transform_step_by_step,
    transform_identify_then_solve,
    transform_skeptical_verifier,
    transform_example_based,
    query_model_with_variants,
    query_models_parallel_with_variants,
)

# Also import original_query_model for backward compatibility with tests
from backend.openrouter import query_model as original_query_model

__all__ = [
    "BASE_MODEL",
    "VARIANT_MODELS",
    "VARIANT_TRANSFORMERS",
    "is_prompt_variant_model",
    "get_actual_model",
    "transform_prompt_for_variant",
    "transform_step_by_step",
    "transform_identify_then_solve",
    "transform_skeptical_verifier",
    "transform_example_based",
    "query_model_with_variants",
    "query_models_parallel_with_variants",
    "original_query_model",
]
