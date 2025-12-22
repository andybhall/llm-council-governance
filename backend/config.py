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
else:
    # Frontier models for real pilot study (December 2025)
    COUNCIL_MODELS = [
        "openai/gpt-5.2",                    # GPT-5.2 (Dec 2025)
        "google/gemini-3-pro-preview",       # Gemini 3 Pro (Nov 2025)
        "anthropic/claude-opus-4.5",         # Claude Opus 4.5 (Nov 2025)
        "x-ai/grok-4",                       # Grok 4 (Jul 2025)
    ]
    CHAIRMAN_MODEL = "anthropic/claude-opus-4.5"

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
