"""Experiment runner for LLM council governance pilot study."""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.evaluation.base import Benchmark, Question
from backend.governance.base import CouncilResult, GovernanceStructure


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    filename: str = "pilot_results.json",
) -> None:
    """
    Save results to a JSON file.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        filename: Name of the output file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(
    output_dir: str,
    filename: str = "pilot_results.json",
) -> List[Dict[str, Any]]:
    """
    Load results from a JSON file.

    Args:
        output_dir: Directory containing results
        filename: Name of the results file

    Returns:
        List of result dictionaries, or empty list if file doesn't exist
    """
    filepath = Path(output_dir) / filename
    if not filepath.exists():
        return []

    with open(filepath) as f:
        return json.load(f)


def get_completed_keys(results: List[Dict[str, Any]]) -> set:
    """
    Get set of completed experiment keys for resumption.

    Each key is a tuple of (benchmark, question_id, structure, replication).
    """
    return {
        (r["benchmark"], r["question_id"], r["structure"], r["replication"])
        for r in results
    }


async def run_single_trial(
    structure: GovernanceStructure,
    question: Question,
    benchmark: Benchmark,
) -> Dict[str, Any]:
    """
    Run a single trial of a governance structure on a question.

    Args:
        structure: The governance structure to run
        question: The question to answer
        benchmark: The benchmark for evaluation

    Returns:
        Dictionary with trial results
    """
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
        "final_answer": council_result.final_answer,
        "elapsed_time": elapsed_time,
        "stage1_responses": council_result.stage1_responses,
        "stage2_data": council_result.stage2_data,
        "stage3_data": council_result.stage3_data,
        "metadata": council_result.metadata,
    }


async def run_experiment(
    structures: List[GovernanceStructure],
    benchmarks: List[Benchmark],
    n_questions: Optional[int] = None,
    n_replications: int = 3,
    output_dir: str = "experiments/results",
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full pilot experiment.

    Args:
        structures: List of governance structures to test
        benchmarks: List of benchmarks to evaluate on
        n_questions: Number of questions per benchmark (None = all)
        n_replications: Number of replications per condition
        output_dir: Directory to save results
        resume: Whether to resume from existing results
        verbose: Whether to print progress

    Returns:
        DataFrame with all experiment results
    """
    # Load existing results if resuming
    results = load_results(output_dir) if resume else []
    completed = get_completed_keys(results) if resume else set()

    if verbose and completed:
        print(f"Resuming from {len(completed)} completed trials")

    # Track experiment start
    experiment_start = datetime.now().isoformat()

    # Calculate total trials for progress reporting
    total_questions = 0
    for benchmark in benchmarks:
        questions = benchmark.load_questions(n_questions)
        total_questions += len(questions)

    total_trials = total_questions * len(structures) * n_replications
    completed_trials = len(completed)

    if verbose:
        print(f"Running experiment: {total_trials} total trials")
        print(f"  Benchmarks: {[b.name for b in benchmarks]}")
        print(f"  Structures: {[s.name for s in structures]}")
        print(f"  Replications: {n_replications}")

    # Run experiment loop
    trial_count = completed_trials

    for benchmark in benchmarks:
        questions = benchmark.load_questions(n_questions)

        if verbose:
            print(f"\nBenchmark: {benchmark.name} ({len(questions)} questions)")

        for question in questions:
            for structure in structures:
                for rep in range(n_replications):
                    # Check if already completed
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
                        # Run the trial
                        trial_result = await run_single_trial(
                            structure, question, benchmark
                        )

                        # Build result record
                        result = {
                            "benchmark": benchmark.name,
                            "question_id": question.id,
                            "structure": structure.name,
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
                                "structure": structure.name,
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


async def run_pilot(
    n_questions: int = 40,
    n_replications: int = 3,
    output_dir: str = "experiments/results",
) -> pd.DataFrame:
    """
    Run the pilot study with default configuration.

    Args:
        n_questions: Number of questions per benchmark
        n_replications: Number of replications per condition
        output_dir: Directory to save results

    Returns:
        DataFrame with all experiment results
    """
    from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS
    from backend.evaluation import GSM8KBenchmark, TruthfulQABenchmark
    from backend.governance import (
        DeliberateSynthesizeStructure,
        DeliberateVoteStructure,
        IndependentRankSynthesize,
        MajorityVoteStructure,
    )

    # Initialize governance structures
    structures = [
        IndependentRankSynthesize(COUNCIL_MODELS, CHAIRMAN_MODEL),
        MajorityVoteStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        DeliberateVoteStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        DeliberateSynthesizeStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
    ]

    # Initialize benchmarks
    benchmarks = [
        GSM8KBenchmark(),
        TruthfulQABenchmark(),
    ]

    return await run_experiment(
        structures=structures,
        benchmarks=benchmarks,
        n_questions=n_questions,
        n_replications=n_replications,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    # Run the pilot experiment
    results_df = asyncio.run(run_pilot())
    print(results_df.head())
