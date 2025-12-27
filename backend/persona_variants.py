"""Persona variant transformations for the persona council experiment.

This module provides:
1. Persona definitions that transform prompts with distinct character voices
2. A model wrapper that maps pseudo-model names to persona transformations
3. Drop-in replacements for query_model/query_models_parallel
"""

import asyncio
from typing import Dict, List, Callable, Any

from backend.openrouter import query_model as original_query_model

# The actual model to use for all personas
BASE_MODEL = "google/gemma-2-9b-it"

# Pseudo-model identifiers that governance structures will use
PERSONA_MODELS = [
    "persona/mathematician",
    "persona/scientist",
    "persona/engineer",
    "persona/teacher",
]


def transform_mathematician(prompt: str) -> str:
    """Persona 1: The Rigorous Mathematician."""
    return f"""You are a rigorous mathematician who demands formal precision. You find sloppy reasoning physically painful. Every claim must be justified, every step must follow logically from the previous one. You don't trust intuition—you trust proof. If something seems obvious, that's exactly when you should verify it carefully.

{prompt}"""


def transform_scientist(prompt: str) -> str:
    """Persona 2: The Skeptical Scientist."""
    return f"""You are a skeptical scientist who questions everything. Your first instinct is to ask "Is this actually true?" You've seen too many confident wrong answers to trust anything at face value. You look for hidden assumptions, check edge cases, and actively try to find flaws in reasoning—including your own.

{prompt}"""


def transform_engineer(prompt: str) -> str:
    """Persona 3: The Practical Engineer."""
    return f"""You are a practical engineer who cares about real-world correctness. Before accepting any answer, you ask: "Does this make sense?" You use estimation and sanity checks. If a math problem gives you a negative number of apples or a person's age of 500 years, something went wrong. You trust your intuition about what's reasonable.

{prompt}"""


def transform_teacher(prompt: str) -> str:
    """Persona 4: The Enthusiastic Teacher."""
    return f"""You are an enthusiastic teacher who loves making things crystal clear. You believe any problem can be solved if you break it down carefully enough. You explain your reasoning as if teaching a student, making each step explicit. You use concrete examples when helpful and always double-check your work before giving a final answer.

{prompt}"""


# Mapping from pseudo-model name to transformation function
PERSONA_TRANSFORMERS: Dict[str, Callable[[str], str]] = {
    "persona/mathematician": transform_mathematician,
    "persona/scientist": transform_scientist,
    "persona/engineer": transform_engineer,
    "persona/teacher": transform_teacher,
}


def is_persona_model(model: str) -> bool:
    """Check if a model name is a persona pseudo-model."""
    return model in PERSONA_TRANSFORMERS


def get_actual_model(pseudo_model: str) -> str:
    """Get the actual model to use for a pseudo-model."""
    if is_persona_model(pseudo_model):
        return BASE_MODEL
    return pseudo_model


def transform_prompt_for_persona(model: str, prompt: str) -> str:
    """Apply persona transformation if model is a pseudo-model."""
    if model in PERSONA_TRANSFORMERS:
        return PERSONA_TRANSFORMERS[model](prompt)
    return prompt


async def query_model_with_personas(
    model: str,
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Query a model, applying persona transformation if needed.

    This is a drop-in replacement for openrouter.query_model that:
    1. Checks if the model is a persona pseudo-model
    2. If so, transforms the user message and routes to BASE_MODEL
    3. Otherwise, passes through to the original query_model
    """
    actual_model = get_actual_model(model)

    if is_persona_model(model):
        # Transform the last user message
        transformed_messages = []
        for msg in messages:
            if msg["role"] == "user":
                transformed_content = transform_prompt_for_persona(
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


async def query_models_parallel_with_personas(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple models in parallel, applying persona transformations as needed.

    Unlike the original query_models_parallel (which sends the same prompt
    to all models), this version can apply different persona transformations
    to each model when using persona pseudo-models.
    """
    tasks = [query_model_with_personas(model, messages) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        model: result if not isinstance(result, Exception) else {"error": str(result)}
        for model, result in zip(models, results)
    }
