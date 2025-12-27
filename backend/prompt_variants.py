"""Prompt variant transformations for the prompt council experiment.

This module provides:
1. Prompt variant definitions that transform the base Stage-1 prompt
2. A model wrapper that maps pseudo-model names to variant transformations
3. Drop-in replacements for query_model/query_models_parallel
"""

import asyncio
from typing import Dict, List, Callable, Any

from backend.openrouter import query_model as original_query_model

# The actual model to use for all variants
BASE_MODEL = "google/gemma-2-9b-it"

# Pseudo-model identifiers that governance structures will use
VARIANT_MODELS = [
    "prompt-variant/step-by-step",
    "prompt-variant/identify-then-solve",
    "prompt-variant/skeptical-verifier",
    "prompt-variant/example-based",
]


def transform_step_by_step(prompt: str) -> str:
    """Variant 1: Step-by-step Chain of Thought."""
    return f"""Think through this step by step. Break down the problem into smaller parts and solve each one carefully before moving to the next.

{prompt}"""


def transform_identify_then_solve(prompt: str) -> str:
    """Variant 2: Identify key information, then solve."""
    return f"""First, identify the key information and constraints in this problem. List them explicitly. Then, use that information to systematically solve the problem.

{prompt}"""


def transform_skeptical_verifier(prompt: str) -> str:
    """Variant 3: Skeptical verifier approach."""
    return f"""Consider common mistakes people make with problems like this. As you solve it, verify each step of your reasoning and check for errors before proceeding.

{prompt}"""


def transform_example_based(prompt: str) -> str:
    """Variant 4: Example-based reasoning."""
    return f"""Think of similar problems you've seen before. What patterns or approaches worked for those? Apply those insights to solve this problem.

{prompt}"""


# Mapping from pseudo-model name to transformation function
VARIANT_TRANSFORMERS: Dict[str, Callable[[str], str]] = {
    "prompt-variant/step-by-step": transform_step_by_step,
    "prompt-variant/identify-then-solve": transform_identify_then_solve,
    "prompt-variant/skeptical-verifier": transform_skeptical_verifier,
    "prompt-variant/example-based": transform_example_based,
}


def is_prompt_variant_model(model: str) -> bool:
    """Check if a model name is a prompt variant pseudo-model."""
    return model in VARIANT_TRANSFORMERS


def get_actual_model(pseudo_model: str) -> str:
    """Get the actual model to use for a pseudo-model."""
    if is_prompt_variant_model(pseudo_model):
        return BASE_MODEL
    return pseudo_model


def transform_prompt_for_variant(model: str, prompt: str) -> str:
    """Apply variant transformation if model is a pseudo-model."""
    if model in VARIANT_TRANSFORMERS:
        return VARIANT_TRANSFORMERS[model](prompt)
    return prompt


async def query_model_with_variants(
    model: str,
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Query a model, applying prompt variant transformation if needed.

    This is a drop-in replacement for openrouter.query_model that:
    1. Checks if the model is a prompt variant pseudo-model
    2. If so, transforms the user message and routes to BASE_MODEL
    3. Otherwise, passes through to the original query_model
    """
    actual_model = get_actual_model(model)

    if is_prompt_variant_model(model):
        # Transform the last user message
        transformed_messages = []
        for msg in messages:
            if msg["role"] == "user":
                transformed_content = transform_prompt_for_variant(
                    model, msg["content"]
                )
                transformed_messages.append({
                    "role": "user",
                    "content": transformed_content
                })
            else:
                transformed_messages.append(msg)
        messages = transformed_messages

    return await original_query_model(actual_model, messages)


async def query_models_parallel_with_variants(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple models in parallel, applying prompt variants as needed.

    Unlike the original query_models_parallel (which sends the same prompt
    to all models), this version can apply different prompt transformations
    to each model when using prompt variant pseudo-models.
    """
    tasks = [query_model_with_variants(model, messages) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        model: result if not isinstance(result, Exception) else {"error": str(result)}
        for model, result in zip(models, results)
    }
