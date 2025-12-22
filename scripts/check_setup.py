#!/usr/bin/env python3
"""
Setup verification and connectivity test for LLM Council Governance Study.

Run this script before executing the pilot study to verify:
1. All dependencies are installed
2. OpenRouter API key is configured
3. API connectivity works for all models
4. HuggingFace datasets can be loaded

Usage:
    python scripts/check_setup.py
    python scripts/check_setup.py --skip-models  # Skip model API tests
    python scripts/check_setup.py --skip-datasets  # Skip dataset loading tests
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print("=" * 60)


def print_status(name: str, success: bool, message: str = "") -> None:
    """Print a status line."""
    icon = "\u2713" if success else "\u2717"
    status = "OK" if success else "FAILED"
    msg = f" - {message}" if message else ""
    print(f"  [{icon}] {name}: {status}{msg}")


def check_dependencies() -> bool:
    """Check that all required dependencies are installed."""
    print_header("Checking Dependencies")

    all_ok = True
    dependencies = [
        ("httpx", "HTTP client for API calls"),
        ("pandas", "Data analysis"),
        ("datasets", "HuggingFace datasets"),
        ("scipy", "Statistical analysis"),
        ("dotenv", "Environment variable loading"),
    ]

    for module, description in dependencies:
        try:
            __import__(module)
            print_status(f"{module} ({description})", True)
        except ImportError:
            print_status(f"{module} ({description})", False, "not installed")
            all_ok = False

    return all_ok


def check_api_key() -> bool:
    """Check that OpenRouter API key is configured."""
    print_header("Checking API Configuration")

    from backend.config import (
        CHAIRMAN_MODEL,
        COUNCIL_MODELS,
        OPENROUTER_API_KEY,
        USE_CHEAP_MODELS,
    )

    # Show model mode
    mode = "CHEAP (testing)" if USE_CHEAP_MODELS else "FRONTIER (production)"
    print_status("Model mode", True, mode)
    print(f"\n  Council models:")
    for model in COUNCIL_MODELS:
        print(f"    - {model}")
    print(f"  Chairman: {CHAIRMAN_MODEL}\n")

    if not OPENROUTER_API_KEY:
        print_status("OPENROUTER_API_KEY", False, "not set")
        print("\n  To fix this:")
        print("    1. Copy .env.example to .env")
        print("    2. Add your API key from https://openrouter.ai/keys")
        print("    3. Run this script again")
        return False

    # Mask the key for display
    masked = OPENROUTER_API_KEY[:8] + "..." + OPENROUTER_API_KEY[-4:]
    print_status("OPENROUTER_API_KEY", True, f"configured ({masked})")
    return True


async def check_model_connectivity(model: str) -> tuple[bool, str]:
    """Test connectivity to a specific model."""
    from backend.openrouter import query_model

    try:
        response = await query_model(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )

        if "content" in response:
            return True, response["content"][:50]
        elif "error" in response:
            return False, response["error"].get("message", "Unknown error")
        else:
            return False, "Unexpected response format"

    except Exception as e:
        return False, str(e)[:50]


async def check_all_models() -> bool:
    """Check connectivity to all configured models."""
    print_header("Checking Model Connectivity")

    from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS

    all_models = list(set(COUNCIL_MODELS + [CHAIRMAN_MODEL]))
    all_ok = True

    print(f"  Testing {len(all_models)} models (this may take a minute)...\n")

    for model in all_models:
        success, message = await check_model_connectivity(model)
        print_status(model, success, message)
        if not success:
            all_ok = False

    return all_ok


def check_datasets() -> bool:
    """Check that benchmark datasets can be loaded."""
    print_header("Checking Dataset Access")

    all_ok = True

    # Check GSM8K
    try:
        from datasets import load_dataset

        print("  Loading GSM8K (test split, first 5 examples)...")
        gsm8k = load_dataset("gsm8k", "main", split="test[:5]")
        print_status("GSM8K", True, f"{len(gsm8k)} examples loaded")
    except Exception as e:
        print_status("GSM8K", False, str(e)[:50])
        all_ok = False

    # Check TruthfulQA
    try:
        print("  Loading TruthfulQA (validation split, first 5 examples)...")
        truthfulqa = load_dataset("truthful_qa", "multiple_choice", split="validation[:5]")
        print_status("TruthfulQA", True, f"{len(truthfulqa)} examples loaded")
    except Exception as e:
        print_status("TruthfulQA", False, str(e)[:50])
        all_ok = False

    return all_ok


def check_project_structure() -> bool:
    """Check that all required project files exist."""
    print_header("Checking Project Structure")

    all_ok = True
    root = Path(__file__).parent.parent

    required_files = [
        "backend/__init__.py",
        "backend/config.py",
        "backend/openrouter.py",
        "backend/governance/__init__.py",
        "backend/governance/base.py",
        "backend/governance/utils.py",
        "backend/governance/independent_rank_synthesize.py",
        "backend/governance/structure_b.py",
        "backend/governance/structure_c.py",
        "backend/governance/structure_d.py",
        "backend/evaluation/__init__.py",
        "backend/evaluation/base.py",
        "backend/evaluation/gsm8k.py",
        "backend/evaluation/truthfulqa.py",
        "experiments/run_pilot.py",
        "experiments/analyze_pilot.py",
    ]

    for filepath in required_files:
        full_path = root / filepath
        if full_path.exists():
            print_status(filepath, True)
        else:
            print_status(filepath, False, "missing")
            all_ok = False

    return all_ok


def estimate_pilot_cost() -> None:
    """Estimate the cost of running the pilot study."""
    print_header("Pilot Study Cost Estimate")

    from backend.config import USE_CHEAP_MODELS

    # Configuration
    n_questions = 80  # 40 GSM8K + 40 TruthfulQA
    n_structures = 4
    n_replications = 3

    # API calls per structure
    calls_per_structure = {
        "A (Rank→Synthesize)": 9,  # 4 + 4 + 1
        "B (Majority Vote)": 4,     # 4 + 0 + 0
        "C (Deliberate→Vote)": 8,   # 4 + 4 + 0
        "D (Deliberate→Synth)": 9,  # 4 + 4 + 1
    }

    total_calls_per_question = sum(calls_per_structure.values())
    total_api_calls = n_questions * n_replications * total_calls_per_question

    print(f"  Questions: {n_questions}")
    print(f"  Structures: {n_structures}")
    print(f"  Replications: {n_replications}")
    print(f"  Total trials: {n_questions * n_structures * n_replications}")
    print(f"  Total API calls: ~{total_api_calls:,}")
    print()

    if USE_CHEAP_MODELS:
        print("  Estimated cost: $2 - $5 (cheap models)")
        print("  Estimated time: 1 - 2 hours")
    else:
        print("  Estimated cost: $150 - $350 (frontier models)")
        print("  Estimated time: 4 - 8 hours")

    print()
    print("  To switch modes, set USE_CHEAP_MODELS in .env")
    print("  Check https://openrouter.ai/models for current rates.")


async def main():
    """Run all setup checks."""
    parser = argparse.ArgumentParser(description="Verify setup for pilot study")
    parser.add_argument("--skip-models", action="store_true", help="Skip model API tests")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset loading tests")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LLM COUNCIL GOVERNANCE STUDY - SETUP VERIFICATION")
    print("=" * 60)

    results = []

    # Check dependencies
    results.append(("Dependencies", check_dependencies()))

    # Check project structure
    results.append(("Project Structure", check_project_structure()))

    # Check API key
    api_key_ok = check_api_key()
    results.append(("API Key", api_key_ok))

    # Check model connectivity (if API key is set)
    if not args.skip_models:
        if api_key_ok:
            models_ok = await check_all_models()
            results.append(("Model Connectivity", models_ok))
        else:
            print_header("Checking Model Connectivity")
            print("  Skipped - API key not configured")
            results.append(("Model Connectivity", False))

    # Check datasets
    if not args.skip_datasets:
        results.append(("Datasets", check_datasets()))

    # Show cost estimate
    estimate_pilot_cost()

    # Summary
    print_header("Summary")

    all_passed = True
    for name, passed in results:
        print_status(name, passed)
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All checks passed! You're ready to run the pilot study.")
        print()
        print("  To run the pilot:")
        print("    python -m experiments.run_pilot")
        print()
        print("  To analyze results:")
        print("    python -m experiments.analyze_pilot")
        return 0
    else:
        print("  Some checks failed. Please fix the issues above before running.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
