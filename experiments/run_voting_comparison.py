"""Experiment runner for fair voting strategy comparison.

This module runs Stage 1 ONCE per (question, replication) and applies
multiple voting strategies to the SAME responses. This enables fair
comparison of voting methods without the confound of different LLM responses.

Key differences from run_pilot.py:
1. Stage 1 is shared across voting strategies
2. Voting strategies are pure functions (no API calls)
3. Results include vote details for analysis
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.evaluation.base import Benchmark, Question
from backend.governance.utils import build_stage1_prompt, extract_final_answer
from backend.governance.voting import (
    VotingStrategy,
    MajorityVoteStrategy,
    WeightedMajorityVoteStrategy,
    create_voting_strategies,
)
from backend.openrouter import query_model, query_models_parallel


async def collect_stage1_responses(
    council_models: List[str],
    query: str,
) -> Dict[str, str]:
    """
    Collect Stage 1 responses from all council models.

    This is called ONCE per (question, replication) and the responses
    are shared across all voting strategies.

    Args:
        council_models: List of model identifiers
        query: The question text

    Returns:
        Dict mapping model names to their full response text
    """
    prompt = build_stage1_prompt(query)
    messages = [{"role": "user", "content": prompt}]
    results = await query_models_parallel(council_models, messages)

    return {
        model: result.get("content", result.get("error", ""))
        for model, result in results.items()
    }


async def get_chairman_answer(
    chairman_model: str,
    query: str,
) -> str:
    """
    Get chairman's answer for tiebreaker.

    Args:
        chairman_model: Model identifier for chairman
        query: The question text

    Returns:
        Chairman's extracted answer
    """
    prompt = build_stage1_prompt(query)
    messages = [{"role": "user", "content": prompt}]
    result = await query_model(chairman_model, messages)

    content = result.get("content", "")
    answer = extract_final_answer(content)

    return answer if answer else content.strip()[-100:]


def apply_voting_strategies(
    stage1_responses: Dict[str, str],
    voting_strategies: List[VotingStrategy],
    chairman_answer: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Apply multiple voting strategies to the same Stage 1 responses.

    Args:
        stage1_responses: Dict mapping model names to response text
        voting_strategies: List of voting strategies to apply
        chairman_answer: Optional tiebreaker answer

    Returns:
        Dict mapping strategy name to result dict with:
            - final_answer: The winning answer
            - vote_details: Additional voting metadata (if available)
    """
    results = {}

    for strategy in voting_strategies:
        final_answer = strategy.vote(stage1_responses, chairman_answer)

        result = {"final_answer": final_answer}

        # Include vote details if available
        if hasattr(strategy, "get_vote_details"):
            result["vote_details"] = strategy.get_vote_details(stage1_responses)

        results[strategy.name] = result

    return results


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    filename: str = "voting_comparison_results.json",
) -> None:
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(
    output_dir: str,
    filename: str = "voting_comparison_results.json",
) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    filepath = Path(output_dir) / filename
    if not filepath.exists():
        return []
    with open(filepath) as f:
        return json.load(f)


def get_completed_keys(results: List[Dict[str, Any]]) -> set:
    """Get set of completed (question_id, replication) pairs for resumption."""
    return {
        (r["question_id"], r["replication"])
        for r in results
    }


async def run_voting_comparison(
    benchmarks: List[Benchmark],
    council_models: List[str],
    chairman_model: str,
    voting_strategies: List[VotingStrategy],
    n_questions: Optional[int] = None,
    n_replications: int = 3,
    output_dir: str = "experiments/results_voting_comparison",
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run voting strategy comparison experiment.

    For each (question, replication):
    1. Run Stage 1 ONCE
    2. Get chairman answer ONCE
    3. Apply ALL voting strategies to the SAME responses
    4. Record results for each strategy

    Args:
        benchmarks: List of benchmarks to evaluate
        council_models: List of model identifiers for council
        chairman_model: Model identifier for chairman/tiebreaker
        voting_strategies: List of voting strategies to compare
        n_questions: Number of questions per benchmark (None = all)
        n_replications: Number of replications per question
        output_dir: Directory to save results
        resume: Whether to resume from existing results
        verbose: Whether to print progress

    Returns:
        DataFrame with all experiment results
    """
    results = load_results(output_dir) if resume else []
    completed = get_completed_keys(results) if resume else set()

    if verbose and completed:
        print(f"Resuming from {len(completed)} completed (question, replication) pairs")

    experiment_start = datetime.now().isoformat()

    # Calculate totals
    total_questions = sum(
        len(benchmark.load_questions(n_questions))
        for benchmark in benchmarks
    )
    total_stage1_calls = total_questions * n_replications
    total_results = total_stage1_calls * len(voting_strategies)

    if verbose:
        print(f"Running voting strategy comparison")
        print(f"  Council models: {[m.split('/')[-1] for m in council_models]}")
        print(f"  Chairman: {chairman_model.split('/')[-1]}")
        print(f"  Voting strategies: {[s.name for s in voting_strategies]}")
        print(f"  Benchmarks: {[b.name for b in benchmarks]}")
        print(f"  Replications: {n_replications}")
        print(f"  Total Stage 1 calls: {total_stage1_calls}")
        print(f"  Total result records: {total_results}")

    stage1_count = len(completed)

    for benchmark in benchmarks:
        questions = benchmark.load_questions(n_questions)

        if verbose:
            print(f"\nBenchmark: {benchmark.name} ({len(questions)} questions)")

        for question in questions:
            for rep in range(n_replications):
                key = (question.id, rep)
                if key in completed:
                    continue

                stage1_count += 1
                if verbose:
                    print(
                        f"  [{stage1_count}/{total_stage1_calls}] "
                        f"{question.id} (rep {rep + 1})"
                    )

                try:
                    start_time = time.time()

                    # Stage 1: Run ONCE
                    stage1_responses = await collect_stage1_responses(
                        council_models, question.text
                    )

                    # Chairman: Query ONCE
                    chairman_answer = await get_chairman_answer(
                        chairman_model, question.text
                    )

                    elapsed_time = time.time() - start_time

                    # Apply ALL voting strategies
                    voting_results = apply_voting_strategies(
                        stage1_responses, voting_strategies, chairman_answer
                    )

                    # Create result record for EACH strategy
                    for strategy in voting_strategies:
                        strategy_result = voting_results[strategy.name]
                        final_answer = strategy_result["final_answer"]

                        # Evaluate
                        eval_result = benchmark.evaluate(question, final_answer)

                        result = {
                            "benchmark": benchmark.name,
                            "question_id": question.id,
                            "replication": rep,
                            "voting_strategy": strategy.name,
                            "timestamp": datetime.now().isoformat(),
                            "experiment_start": experiment_start,
                            "is_correct": eval_result.is_correct,
                            "predicted": eval_result.predicted,
                            "expected": eval_result.expected,
                            "final_answer": final_answer,
                            "chairman_answer": chairman_answer,
                            "stage1_responses": stage1_responses,
                            "elapsed_time": elapsed_time,
                        }

                        # Include vote details if available
                        if "vote_details" in strategy_result:
                            result["vote_details"] = strategy_result["vote_details"]

                        results.append(result)

                    # Save incrementally
                    save_results(results, output_dir)

                except Exception as e:
                    if verbose:
                        print(f"    ERROR: {e}")

                    # Log error for each strategy
                    for strategy in voting_strategies:
                        results.append({
                            "benchmark": benchmark.name,
                            "question_id": question.id,
                            "replication": rep,
                            "voting_strategy": strategy.name,
                            "timestamp": datetime.now().isoformat(),
                            "experiment_start": experiment_start,
                            "is_correct": None,
                            "error": str(e),
                        })
                    save_results(results, output_dir)

    if verbose:
        print(f"\nExperiment complete. {len(results)} total results saved.")

    return pd.DataFrame(results)


async def run_voting_pilot(
    n_questions: int = 40,
    n_replications: int = 3,
    output_dir: str = "experiments/results_voting_comparison",
    weights_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run voting comparison with default configuration.

    Args:
        n_questions: Number of questions per benchmark
        n_replications: Number of replications
        output_dir: Directory to save results
        weights_file: Path to model weights for weighted voting

    Returns:
        DataFrame with all experiment results
    """
    from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS, WEIGHTS_FILE
    from backend.evaluation import GSM8KBenchmark, TruthfulQABenchmark

    # Use provided weights file or default from config
    weights_path = weights_file or WEIGHTS_FILE

    # Create voting strategies
    voting_strategies = [
        MajorityVoteStrategy(),
        WeightedMajorityVoteStrategy(weights_file=weights_path),
    ]

    benchmarks = [
        GSM8KBenchmark(),
        TruthfulQABenchmark(),
    ]

    return await run_voting_comparison(
        benchmarks=benchmarks,
        council_models=COUNCIL_MODELS,
        chairman_model=CHAIRMAN_MODEL,
        voting_strategies=voting_strategies,
        n_questions=n_questions,
        n_replications=n_replications,
        output_dir=output_dir,
    )


def analyze_voting_comparison(
    output_dir: str = "experiments/results_voting_comparison",
) -> None:
    """
    Analyze voting comparison results and print summary.

    Args:
        output_dir: Directory containing results
    """
    results = load_results(output_dir)
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    valid_df = df[df["is_correct"].notna()]

    print("=" * 60)
    print("VOTING STRATEGY COMPARISON - FAIR ANALYSIS")
    print("=" * 60)
    print(f"\nTotal result records: {len(df)}")
    print(f"Valid results: {len(valid_df)}")

    # Unique Stage 1 calls
    unique_stage1 = df.groupby(["question_id", "replication"]).ngroups
    print(f"Unique (question, replication) pairs: {unique_stage1}")

    print("\n" + "-" * 40)
    print("ACCURACY BY VOTING STRATEGY")
    print("-" * 40)

    strategy_acc = (
        valid_df.groupby("voting_strategy")
        .agg(
            n_trials=("is_correct", "count"),
            n_correct=("is_correct", "sum"),
        )
        .reset_index()
    )
    strategy_acc["accuracy"] = strategy_acc["n_correct"] / strategy_acc["n_trials"]
    strategy_acc = strategy_acc.sort_values("accuracy", ascending=False)

    for _, row in strategy_acc.iterrows():
        print(
            f"  {row['voting_strategy']}: {row['accuracy']:.1%} "
            f"({int(row['n_correct'])}/{int(row['n_trials'])})"
        )

    # Pairwise comparison
    if len(strategy_acc) >= 2:
        print("\n" + "-" * 40)
        print("PAIRWISE COMPARISON (same responses)")
        print("-" * 40)

        strategies = valid_df["voting_strategy"].unique()
        if len(strategies) >= 2:
            s1, s2 = strategies[0], strategies[1]

            # Get paired results
            s1_df = valid_df[valid_df["voting_strategy"] == s1].set_index(
                ["question_id", "replication"]
            )
            s2_df = valid_df[valid_df["voting_strategy"] == s2].set_index(
                ["question_id", "replication"]
            )

            common_idx = s1_df.index.intersection(s2_df.index)

            if len(common_idx) > 0:
                s1_correct = s1_df.loc[common_idx, "is_correct"]
                s2_correct = s2_df.loc[common_idx, "is_correct"]

                both_correct = (s1_correct & s2_correct).sum()
                s1_only = (s1_correct & ~s2_correct).sum()
                s2_only = (~s1_correct & s2_correct).sum()
                both_wrong = (~s1_correct & ~s2_correct).sum()

                print(f"\n  Comparing {s1} vs {s2}:")
                print(f"    Both correct:      {both_correct}")
                print(f"    {s1} only:  {s1_only}")
                print(f"    {s2} only:  {s2_only}")
                print(f"    Both wrong:        {both_wrong}")
                print(f"    Total pairs:       {len(common_idx)}")

                # McNemar's test
                if s1_only + s2_only > 0:
                    from scipy import stats

                    # Exact McNemar's test
                    n_discordant = s1_only + s2_only
                    if n_discordant < 25:
                        p_value = stats.binomtest(
                            s1_only, n_discordant, 0.5, alternative="two-sided"
                        ).pvalue
                    else:
                        chi2 = (abs(s1_only - s2_only) - 1) ** 2 / n_discordant
                        p_value = 1 - stats.chi2.cdf(chi2, df=1)

                    print(f"\n    McNemar's test p-value: {p_value:.4f}")
                    if p_value < 0.05:
                        print("    Result: Significant difference")
                    else:
                        print("    Result: No significant difference")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_voting_comparison()
    else:
        results_df = asyncio.run(run_voting_pilot())
        print("\nAnalyzing results...")
        analyze_voting_comparison()
