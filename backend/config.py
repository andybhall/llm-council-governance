"""Configuration for LLM Council Governance Study."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# =============================================================================
# Model Configuration
# =============================================================================
# Set USE_CHEAP_MODELS=true in .env for low-cost testing
# Cheap models: ~$0.10-0.50 per 1M tokens
# Frontier models: ~$5-30 per 1M tokens

USE_CHEAP_MODELS = os.getenv("USE_CHEAP_MODELS", "true").lower() == "true"

if USE_CHEAP_MODELS:
    # Cheap models for testing (~$0.001 per run)
    COUNCIL_MODELS = [
        "meta-llama/llama-3.1-8b-instruct",      # ~$0.05/1M tokens
        "mistralai/mistral-7b-instruct",          # ~$0.06/1M tokens
        "google/gemma-2-9b-it",                   # ~$0.08/1M tokens
        "qwen/qwen-2.5-7b-instruct",              # ~$0.05/1M tokens
    ]
    CHAIRMAN_MODEL = "meta-llama/llama-3.1-8b-instruct"

    # Extended model pool for council size experiments (6 models)
    EXTENDED_MODELS = COUNCIL_MODELS + [
        "microsoft/phi-3-medium-128k-instruct:free",  # ~14B params
        "openchat/openchat-7b:free",                   # Different architecture
    ]
else:
    # Frontier models for real pilot study (December 2025)
    COUNCIL_MODELS = [
        "openai/gpt-5.2",                    # GPT-5.2 (Dec 2025)
        "google/gemini-3-pro-preview",       # Gemini 3 Pro (Nov 2025)
        "anthropic/claude-opus-4.5",         # Claude Opus 4.5 (Nov 2025)
        "x-ai/grok-4",                       # Grok 4 (Jul 2025)
    ]
    CHAIRMAN_MODEL = "anthropic/claude-opus-4.5"

    # Extended model pool for council size experiments
    EXTENDED_MODELS = COUNCIL_MODELS + [
        "meta-llama/llama-3.3-70b-instruct",  # Larger Llama
        "mistralai/mistral-large",             # Mistral Large
    ]

# =============================================================================
# Weighted Voting Configuration
# =============================================================================
# Path to JSON file containing model weights (accuracy rates from pilot study)
WEIGHTS_FILE = os.getenv(
    "WEIGHTS_FILE",
    str(Path(__file__).parent.parent / "experiments" / "results" / "model_weights.json")
)

# Default weights if no weights file exists
# These are placeholder values; run analyze_pilot.py to generate actual weights
MODEL_WEIGHTS = {
    "meta-llama/llama-3.1-8b-instruct": 0.75,
    "mistralai/mistral-7b-instruct": 0.69,
    "google/gemma-2-9b-it": 0.84,
    "qwen/qwen-2.5-7b-instruct": 0.80,
    "microsoft/phi-3-medium-128k-instruct:free": 0.75,
    "openchat/openchat-7b:free": 0.70,
}

# =============================================================================
# Council Size Experiment Configuration
# =============================================================================
# Council sizes to test (for inverted-U curve analysis)
COUNCIL_SIZES = [2, 3, 4, 5, 6]

# =============================================================================
# API Configuration
# =============================================================================
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Default API timeout in seconds (90s with 7 retries = ample time)
DEFAULT_API_TIMEOUT = float(os.getenv("API_TIMEOUT", "90.0"))
