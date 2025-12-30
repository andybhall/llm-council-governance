"""Experiment-specific assets (prompt variants, persona variants, etc.)."""

from experiments.assets.prompt_variants import (
    VARIANT_MODELS,
    VARIANT_TRANSFORMERS,
    is_prompt_variant_model,
    query_model_with_variants,
    query_models_parallel_with_variants,
)

from experiments.assets.persona_variants import (
    PERSONA_MODELS,
    PERSONA_TRANSFORMERS,
    is_persona_model,
    query_model_with_personas,
    query_models_parallel_with_personas,
)

__all__ = [
    # Prompt variants
    "VARIANT_MODELS",
    "VARIANT_TRANSFORMERS",
    "is_prompt_variant_model",
    "query_model_with_variants",
    "query_models_parallel_with_variants",
    # Persona variants
    "PERSONA_MODELS",
    "PERSONA_TRANSFORMERS",
    "is_persona_model",
    "query_model_with_personas",
    "query_models_parallel_with_personas",
]
