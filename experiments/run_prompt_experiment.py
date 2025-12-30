"""Experiment runner for prompt variant council study.

Tests whether using a single model with different reasoning prompts
can be as effective as a council of different models.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd

from backend.evaluation.base import Benchmark, Question
from backend.governance.base import CouncilResult, GovernanceStructure
from backend.prompt_variants import (
    VARIANT_MODELS,
    BASE_MODEL,
    query_model_with_variants,
    query_models_parallel_with_variants,
)


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    filename: str = "prompt_experiment_results.json",
) -> None:
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(
    output_dir: str,
    filename: str = "prompt_experiment_results.json",
) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    filepath = Path(output_dir) / filename
    if not filepath.exists():
        return []
    with open(filepath) as f:
        return json.load(f)


def get_completed_keys(results: List[Dict[str, Any]]) -> set:
    """Get set of completed experiment keys for resumption."""
    return {
        (r["benchmark"], r["question_id"], r["structure"], r["replication"])
        for r in results
    }


async def run_single_trial(
    structure: GovernanceStructure,
    question: Question,
    benchmark: Benchmark,
) -> Dict[str, Any]:
    """Run a single trial of a governance structure on a question."""
    start_time = time.time()
    council_result: CouncilResult = await structure.run(question.text)
    elapsed_time = time.time() - start_time
    eval_result = benchmark.evaluate(question, council_result.final_answer)

    return {
        "is_correct": eval_result.is_correct,
        "predicted": eval_result.predicted,
        "expected": eval_result.expected,
        "question_text": question.text,
        "final_answer": council_result.final_answer,
        "elapsed_time": elapsed_time,
        "stage1_responses": council_result.stage1_responses,
        "stage2_data": council_result.stage2_data,
        "stage3_data": council_result.stage3_data,
        "metadata": council_result.metadata,
    }


def create_variant_patches():
    """Create patches to use variant-aware query functions in all governance structures."""
    import importlib

    # Use importlib to avoid namespace collision with majority_vote function
    base_module = importlib.import_module("backend.governance.base")
    struct_a = importlib.import_module("backend.governance.independent_rank_synthesize")
    struct_c = importlib.import_module("backend.governance.deliberate_vote")
    struct_d = importlib.import_module("backend.governance.deliberate_synthesize")

    return [
        # Base class has both query_model (chairman) and query_models_parallel (Stage 1)
        patch.object(base_module, "query_model", query_model_with_variants),
        patch.object(base_module, "query_models_parallel", query_models_parallel_with_variants),
        # These structures still import query_model directly for their specific stages
        patch.object(struct_a, "query_model", query_model_with_variants),  # synthesis
        patch.object(struct_a, "query_models_parallel", query_models_parallel_with_variants),  # Stage 2 rankings
        patch.object(struct_c, "query_model", query_model_with_variants),  # deliberation
        patch.object(struct_d, "query_model", query_model_with_variants),  # deliberation
    ]


async def run_prompt_experiment(
    structures: List[GovernanceStructure],
    benchmarks: List[Benchmark],
    n_questions: Optional[int] = None,
    n_replications: int = 3,
    output_dir: str = "experiments/results_prompt_variants",
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the prompt variant experiment.

    Uses monkeypatching to replace openrouter functions with variant-aware
    versions during the experiment.
    """
    results = load_results(output_dir) if resume else []
    completed = get_completed_keys(results) if resume else set()

    if verbose and completed:
        print(f"Resuming from {len(completed)} completed trials")

    experiment_start = datetime.now().isoformat()

    # Calculate totals
    total_questions = sum(
        len(benchmark.load_questions(n_questions))
        for benchmark in benchmarks
    )
    total_trials = total_questions * len(structures) * n_replications

    if verbose:
        print(f"Running prompt variant experiment: {total_trials} total trials")
        print(f"  Base model: {BASE_MODEL}")
        print(f"  Prompt variants: {VARIANT_MODELS}")
        print(f"  Structures: {[s.name for s in structures]}")
        print(f"  Benchmarks: {[b.name for b in benchmarks]}")
        print(f"  Replications: {n_replications}")

    # Apply patches for variant-aware query functions
    patches = create_variant_patches()
    for p in patches:
        p.start()

    try:
        trial_count = len(completed)

        for benchmark in benchmarks:
            questions = benchmark.load_questions(n_questions)

            if verbose:
                print(f"\nBenchmark: {benchmark.name} ({len(questions)} questions)")

            for question in questions:
                for structure in structures:
                    for rep in range(n_replications):
                        key = (benchmark.name, question.id, structure.name, rep)
                        if key in completed:
                            continue

                        trial_count += 1
                        if verbose:
                            print(
                                f"  [{trial_count}/{total_trials}] "
                                f"{structure.name} on {question.id} (rep {rep + 1})"
                            )

                        try:
                            trial_result = await run_single_trial(
                                structure, question, benchmark
                            )

                            result = {
                                "benchmark": benchmark.name,
                                "question_id": question.id,
                                "structure": structure.name,
                                "base_model": BASE_MODEL,
                                "prompt_variants": VARIANT_MODELS,
                                "replication": rep,
                                "timestamp": datetime.now().isoformat(),
                                "experiment_start": experiment_start,
                                **trial_result,
                            }

                            results.append(result)
                            save_results(results, output_dir)

                        except Exception as e:
                            if verbose:
                                print(f"    ERROR: {e}")

                            results.append({
                                "benchmark": benchmark.name,
                                "question_id": question.id,
                                "structure": structure.name,
                                "base_model": BASE_MODEL,
                                "prompt_variants": VARIANT_MODELS,
                                "replication": rep,
                                "timestamp": datetime.now().isoformat(),
                                "experiment_start": experiment_start,
                                "is_correct": None,
                                "error": str(e),
                            })
                            save_results(results, output_dir)
    finally:
        # Always stop patches
        for p in patches:
            p.stop()

    if verbose:
        print(f"\nExperiment complete. {len(results)} total results saved.")

    return pd.DataFrame(results)


async def run_prompt_pilot(
    n_questions: int = 40,
    n_replications: int = 3,
    output_dir: str = "experiments/results_prompt_variants",
) -> pd.DataFrame:
    """Run prompt variant experiment with default configuration."""
    from backend.evaluation import GSM8KBenchmark, TruthfulQABenchmark
    from backend.governance import (
        IndependentRankSynthesize,
        MajorityVoteStructure,
        DeliberateVoteStructure,
        DeliberateSynthesizeStructure,
    )

    # Use prompt variant pseudo-models as council
    # Chairman also uses the step-by-step variant
    structures = [
        IndependentRankSynthesize(VARIANT_MODELS, VARIANT_MODELS[0]),
        MajorityVoteStructure(VARIANT_MODELS, VARIANT_MODELS[0]),
        DeliberateVoteStructure(VARIANT_MODELS, VARIANT_MODELS[0]),
        DeliberateSynthesizeStructure(VARIANT_MODELS, VARIANT_MODELS[0]),
    ]

    benchmarks = [
        GSM8KBenchmark(),
        TruthfulQABenchmark(),
    ]

    return await run_prompt_experiment(
        structures=structures,
        benchmarks=benchmarks,
        n_questions=n_questions,
        n_replications=n_replications,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    results_df = asyncio.run(run_prompt_pilot())
    print(results_df.head())
