"""Experiment runner for council size analysis.

Tests the hypothesis that governance quality follows an inverted-U curve
as council size varies. Uses Structure B (majority vote) as baseline.
"""

import asyncio
import json
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config import CHAIRMAN_MODEL, COUNCIL_SIZES, EXTENDED_MODELS
from backend.evaluation.base import Benchmark, Question
from backend.governance.base import CouncilResult
from backend.governance.structure_b import MajorityVoteStructure


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    filename: str = "council_size_results.json",
) -> None:
    """Save results to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(
    output_dir: str,
    filename: str = "council_size_results.json",
) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    filepath = Path(output_dir) / filename
    if not filepath.exists():
        return []

    with open(filepath) as f:
        return json.load(f)


def get_completed_keys(results: List[Dict[str, Any]]) -> set:
    """Get set of completed experiment keys for resumption."""
    return {
        (
            r["benchmark"],
            r["question_id"],
            r["council_size"],
            tuple(sorted(r["council_models"])),
            r["replication"],
        )
        for r in results
    }


def get_model_combinations(
    model_pool: List[str],
    council_size: int,
) -> List[List[str]]:
    """
    Get all combinations of models for a given council size.

    For smaller experiments, we use a fixed subset rather than all combinations
    to keep the experiment tractable.

    Args:
        model_pool: List of available models
        council_size: Number of models to select

    Returns:
        List of model combinations (each is a list of model names)
    """
    if council_size > len(model_pool):
        raise ValueError(
            f"Council size {council_size} exceeds model pool size {len(model_pool)}"
        )

    # For practical experiments, use just the first N models
    # This keeps the experiment tractable while still testing the hypothesis
    return [model_pool[:council_size]]


async def run_single_trial(
    structure: MajorityVoteStructure,
    question: Question,
    benchmark: Benchmark,
) -> Dict[str, Any]:
    """Run a single trial of a governance structure on a question."""
    start_time = time.time()

    # Run the council
    council_result: CouncilResult = await structure.run(question.text)

    elapsed_time = time.time() - start_time

    # Evaluate the result
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


async def run_council_size_experiment(
    benchmarks: List[Benchmark],
    model_pool: Optional[List[str]] = None,
    council_sizes: Optional[List[int]] = None,
    chairman_model: Optional[str] = None,
    n_questions: Optional[int] = None,
    n_replications: int = 3,
    output_dir: str = "experiments/results_council_size",
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the council size experiment.

    Tests different council sizes using Structure B (majority vote) to determine
    the optimal number of models in a council.

    Args:
        benchmarks: List of benchmarks to evaluate on
        model_pool: Pool of models to sample from (default: EXTENDED_MODELS)
        council_sizes: List of council sizes to test (default: COUNCIL_SIZES)
        chairman_model: Model to use as chairman (default: CHAIRMAN_MODEL)
        n_questions: Number of questions per benchmark (None = all)
        n_replications: Number of replications per condition
        output_dir: Directory to save results
        resume: Whether to resume from existing results
        verbose: Whether to print progress

    Returns:
        DataFrame with all experiment results
    """
    # Use defaults from config
    if model_pool is None:
        model_pool = EXTENDED_MODELS
    if council_sizes is None:
        council_sizes = COUNCIL_SIZES
    if chairman_model is None:
        chairman_model = CHAIRMAN_MODEL

    # Load existing results if resuming
    results = load_results(output_dir) if resume else []
    completed = get_completed_keys(results) if resume else set()

    if verbose and completed:
        print(f"Resuming from {len(completed)} completed trials")

    # Track experiment start
    experiment_start = datetime.now().isoformat()

    # Calculate total trials
    total_questions = 0
    for benchmark in benchmarks:
        questions = benchmark.load_questions(n_questions)
        total_questions += len(questions)

    total_trials = (
        total_questions
        * len(council_sizes)
        * n_replications
    )
    completed_trials = len(completed)

    if verbose:
        print(f"Running council size experiment:")
        print(f"  Model pool: {len(model_pool)} models")
        print(f"  Council sizes: {council_sizes}")
        print(f"  Benchmarks: {[b.name for b in benchmarks]}")
        print(f"  Questions per benchmark: {n_questions or 'all'}")
        print(f"  Replications: {n_replications}")
        print(f"  Total trials: {total_trials}")

    # Run experiment loop
    trial_count = completed_trials

    for council_size in council_sizes:
        # Get model combinations for this council size
        model_combos = get_model_combinations(model_pool, council_size)

        for council_models in model_combos:
            # Create structure with this council configuration
            structure = MajorityVoteStructure(
                council_models=council_models,
                chairman_model=chairman_model,
            )

            for benchmark in benchmarks:
                questions = benchmark.load_questions(n_questions)

                if verbose:
                    print(
                        f"\nCouncil size {council_size}, "
                        f"Benchmark: {benchmark.name}"
                    )

                for question in questions:
                    for rep in range(n_replications):
                        # Check if already completed
                        key = (
                            benchmark.name,
                            question.id,
                            council_size,
                            tuple(sorted(council_models)),
                            rep,
                        )
                        if key in completed:
                            continue

                        trial_count += 1
                        if verbose:
                            print(
                                f"  [{trial_count}/{total_trials}] "
                                f"size={council_size}, {question.id} (rep {rep + 1})"
                            )

                        try:
                            # Run the trial
                            trial_result = await run_single_trial(
                                structure, question, benchmark
                            )

                            # Build result record
                            result = {
                                "benchmark": benchmark.name,
                                "question_id": question.id,
                                "council_size": council_size,
                                "council_models": council_models,
                                "chairman_model": chairman_model,
                                "replication": rep,
                                "timestamp": datetime.now().isoformat(),
                                "experiment_start": experiment_start,
                                **trial_result,
                            }

                            results.append(result)

                            # Save incrementally
                            save_results(results, output_dir)

                        except Exception as e:
                            if verbose:
                                print(f"    ERROR: {e}")

                            # Log the error
                            results.append(
                                {
                                    "benchmark": benchmark.name,
                                    "question_id": question.id,
                                    "council_size": council_size,
                                    "council_models": council_models,
                                    "chairman_model": chairman_model,
                                    "replication": rep,
                                    "timestamp": datetime.now().isoformat(),
                                    "experiment_start": experiment_start,
                                    "is_correct": None,
                                    "predicted": None,
                                    "expected": question.ground_truth,
                                    "error": str(e),
                                }
                            )
                            save_results(results, output_dir)

    if verbose:
        print(f"\nExperiment complete. {len(results)} total results saved.")

    return pd.DataFrame(results)


async def run_council_size_pilot(
    n_questions: int = 40,
    n_replications: int = 3,
    output_dir: str = "experiments/results_council_size",
) -> pd.DataFrame:
    """
    Run the council size experiment with default configuration.

    Args:
        n_questions: Number of questions per benchmark
        n_replications: Number of replications per condition
        output_dir: Directory to save results

    Returns:
        DataFrame with all experiment results
    """
    from backend.evaluation import GSM8KBenchmark, TruthfulQABenchmark

    benchmarks = [
        GSM8KBenchmark(),
        TruthfulQABenchmark(),
    ]

    return await run_council_size_experiment(
        benchmarks=benchmarks,
        n_questions=n_questions,
        n_replications=n_replications,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    results_df = asyncio.run(run_council_size_pilot())
    print(results_df.head())
