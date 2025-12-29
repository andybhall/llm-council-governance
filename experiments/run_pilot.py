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
from experiments.manifest import create_manifest


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
        "question_text": question.text,
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
    max_concurrent: int = 20,
) -> pd.DataFrame:
    """
    Run the full pilot experiment with parallel trial execution.

    Args:
        structures: List of governance structures to test
        benchmarks: List of benchmarks to evaluate on
        n_questions: Number of questions per benchmark (None = all)
        n_replications: Number of replications per condition
        output_dir: Directory to save results
        resume: Whether to resume from existing results
        verbose: Whether to print progress
        max_concurrent: Maximum number of concurrent trials (default: 20)

    Returns:
        DataFrame with all experiment results
    """
    from backend.openrouter import close_shared_client

    # Generate experiment manifest for reproducibility
    config = {
        "structures": [s.name for s in structures],
        "benchmarks": [b.name for b in benchmarks],
        "n_questions": n_questions,
        "n_replications": n_replications,
        "temperature": 0.0,  # Default from openrouter.py
    }
    create_manifest(config, output_dir=output_dir)

    # Load existing results if resuming
    results = load_results(output_dir) if resume else []
    completed = get_completed_keys(results) if resume else set()

    if verbose and completed:
        print(f"Resuming from {len(completed)} completed trials")

    # Track experiment start
    experiment_start = datetime.now().isoformat()

    # Build list of all pending trials
    pending_trials = []
    for benchmark in benchmarks:
        questions = benchmark.load_questions(n_questions)
        for question in questions:
            for structure in structures:
                for rep in range(n_replications):
                    key = (benchmark.name, question.id, structure.name, rep)
                    if key not in completed:
                        pending_trials.append({
                            "benchmark": benchmark,
                            "question": question,
                            "structure": structure,
                            "rep": rep,
                            "key": key,
                        })

    total_trials = len(completed) + len(pending_trials)

    if verbose:
        print(f"Running experiment: {total_trials} total trials")
        print(f"  Pending: {len(pending_trials)} trials")
        print(f"  Benchmarks: {[b.name for b in benchmarks]}")
        print(f"  Structures: {[s.name for s in structures]}")
        print(f"  Replications: {n_replications}")
        print(f"  Max concurrent: {max_concurrent}")

    if not pending_trials:
        if verbose:
            print("\nNo pending trials. Experiment already complete.")
        return pd.DataFrame(results)

    # Semaphore to limit concurrent trials
    semaphore = asyncio.Semaphore(max_concurrent)

    # Thread-safe counters and result collection
    results_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    completed_count = [0]  # Use list for mutable reference
    last_save_count = [0]

    # Per-trial timeout (5 minutes max per trial to prevent hangs)
    trial_timeout = 300

    async def run_bounded_trial(trial_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single trial with semaphore-bounded concurrency and timeout."""
        async with semaphore:
            benchmark = trial_info["benchmark"]
            question = trial_info["question"]
            structure = trial_info["structure"]
            rep = trial_info["rep"]

            try:
                # Wrap with timeout to prevent indefinite hangs
                trial_result = await asyncio.wait_for(
                    run_single_trial(structure, question, benchmark),
                    timeout=trial_timeout,
                )
                result = {
                    "benchmark": benchmark.name,
                    "question_id": question.id,
                    "structure": structure.name,
                    "replication": rep,
                    "timestamp": datetime.now().isoformat(),
                    "experiment_start": experiment_start,
                    **trial_result,
                }
            except asyncio.TimeoutError:
                result = {
                    "benchmark": benchmark.name,
                    "question_id": question.id,
                    "structure": structure.name,
                    "replication": rep,
                    "timestamp": datetime.now().isoformat(),
                    "experiment_start": experiment_start,
                    "is_correct": None,
                    "predicted": None,
                    "expected": question.ground_truth,
                    "error": f"Trial timeout after {trial_timeout}s",
                }
                if verbose:
                    async with progress_lock:
                        print(f"  TIMEOUT on {question.id}/{structure.name}")
            except Exception as e:
                result = {
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
                if verbose:
                    async with progress_lock:
                        print(f"  ERROR on {question.id}: {e}")

            # Update progress and save periodically
            async with results_lock:
                results.append(result)
                completed_count[0] += 1

                # Progress reporting
                if verbose and completed_count[0] % 10 == 0:
                    print(
                        f"  Progress: {completed_count[0] + len(completed)}/{total_trials} "
                        f"({100 * (completed_count[0] + len(completed)) / total_trials:.1f}%)"
                    )

                # Save every 25 results instead of every single one
                if completed_count[0] - last_save_count[0] >= 25:
                    save_results(results, output_dir)
                    last_save_count[0] = completed_count[0]

            return result

    # Run all pending trials in parallel (semaphore limits concurrency)
    if verbose:
        print(f"\nStarting parallel execution...")
        start_time = time.time()

    try:
        await asyncio.gather(*[run_bounded_trial(t) for t in pending_trials])
    finally:
        # Always close the shared HTTP client when done
        await close_shared_client()

    # Final save
    save_results(results, output_dir)

    if verbose:
        elapsed = time.time() - start_time
        rate = len(pending_trials) / elapsed * 60 if elapsed > 0 else 0
        print(f"\nExperiment complete. {len(results)} total results saved.")
        print(f"  Time: {elapsed:.1f}s ({rate:.1f} trials/min)")

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
    from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS, WEIGHTS_FILE
    from backend.evaluation import GSM8KBenchmark, TruthfulQABenchmark
    from backend.governance import (
        DeliberateSynthesizeStructure,
        DeliberateVoteStructure,
        IndependentRankSynthesize,
        MajorityVoteStructure,
        SelfConsistencyVoteStructure,
        WeightedMajorityVote,
    )

    # Initialize governance structures
    structures = [
        IndependentRankSynthesize(COUNCIL_MODELS, CHAIRMAN_MODEL),
        MajorityVoteStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        DeliberateVoteStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        DeliberateSynthesizeStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        WeightedMajorityVote(COUNCIL_MODELS, CHAIRMAN_MODEL, weights_file=WEIGHTS_FILE),
        # Self-consistency baseline: compute-matched single model sampling
        SelfConsistencyVoteStructure(
            base_model="google/gemini-2.0-flash-001",  # Best council model
            n_samples=2 * len(COUNCIL_MODELS) + 1,  # Match 2N+1 call budget
            temperature=0.7,
        ),
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
