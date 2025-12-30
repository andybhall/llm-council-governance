"""Reanalyze existing pilot results with fair voting comparison.

This script applies multiple voting strategies to the SAME Stage 1 responses
from an existing pilot study, enabling fair comparison of voting methods.

Usage:
    python -m experiments.reanalyze_pilot [--results-file PATH] [--weights-file PATH]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.governance.voting import (
    MajorityVoteStrategy,
    WeightedMajorityVoteStrategy,
    create_voting_strategies,
)


def load_pilot_results(
    results_file: str = "experiments/results/pilot_results.json",
) -> List[Dict[str, Any]]:
    """Load pilot results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def get_unique_stage1_trials(
    results: List[Dict[str, Any]],
    base_structure: str = "Independent → Majority Vote",
) -> List[Dict[str, Any]]:
    """
    Extract unique (question, replication) trials from a specific structure.

    We use one structure's Stage 1 responses as the "ground truth" for
    fair comparison, since each structure ran its own API calls.

    Args:
        results: Full pilot results
        base_structure: Structure to use as source of Stage 1 responses

    Returns:
        List of trial dicts with stage1_responses
    """
    return [
        r for r in results
        if r.get("structure") == base_structure
        and r.get("stage1_responses")
        and r.get("expected") is not None
    ]


def apply_voting_strategies_to_trial(
    trial: Dict[str, Any],
    strategies: List,
    chairman_answer: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Apply multiple voting strategies to a single trial's Stage 1 responses.

    Args:
        trial: Trial dict containing stage1_responses
        strategies: List of VotingStrategy instances
        chairman_answer: Optional tiebreaker

    Returns:
        Dict mapping strategy name to result dict
    """
    stage1 = trial["stage1_responses"]

    results = {}
    for strategy in strategies:
        final_answer = strategy.vote(stage1, chairman_answer)
        results[strategy.name] = {
            "final_answer": final_answer,
        }
        if hasattr(strategy, "get_vote_details"):
            results[strategy.name]["vote_details"] = strategy.get_vote_details(stage1)

    return results


def extract_number(text: str) -> Optional[str]:
    """Extract a numeric answer from text, handling common formats."""
    import re
    if not text:
        return None

    # Normalize: remove $ and commas, handle ** markdown
    text = text.replace("**", "").strip()

    # Try to find a number (handles $18, 18, 18.5, -18, etc.)
    # Look for patterns like $123, 123, 123.45, -123
    match = re.search(r'\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        # Normalize: remove commas
        return match.group(1).replace(",", "")
    return None


def extract_letter(text: str) -> Optional[str]:
    """Extract a letter answer (A, B, C, etc.) from text."""
    import re
    if not text:
        return None

    text = text.strip().upper()

    # Look for standalone letter
    match = re.search(r'\b([A-Z])\b', text)
    if match:
        return match.group(1)

    # Just return first char if it's a letter
    if text and text[0].isalpha():
        return text[0]

    return None


def evaluate_answer(
    predicted: str,
    expected: str,
    benchmark: str,
) -> bool:
    """
    Evaluate if predicted answer matches expected.

    Uses proper extraction logic matching the benchmark evaluators.

    Args:
        predicted: Predicted answer string
        expected: Expected answer string
        benchmark: Benchmark name (GSM8K or TruthfulQA)

    Returns:
        True if correct
    """
    if not predicted or not expected:
        return False

    if benchmark == "GSM8K":
        # Numeric comparison with proper extraction
        pred_num = extract_number(predicted)
        exp_num = extract_number(expected)

        if pred_num is None or exp_num is None:
            return False

        try:
            return float(pred_num) == float(exp_num)
        except ValueError:
            return False
    else:
        # Letter comparison (TruthfulQA)
        pred_letter = extract_letter(predicted)
        exp_letter = extract_letter(expected)

        if pred_letter is None or exp_letter is None:
            return False

        return pred_letter == exp_letter


def reanalyze_pilot(
    results_file: str = "experiments/results/pilot_results.json",
    weights_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Reanalyze pilot results with fair voting comparison.

    Args:
        results_file: Path to pilot results JSON
        weights_file: Path to model weights JSON (optional)
        output_file: Path to save reanalysis results (optional)

    Returns:
        DataFrame with reanalysis results
    """
    print("Loading pilot results...")
    results = load_pilot_results(results_file)

    print("Extracting unique Stage 1 trials...")
    trials = get_unique_stage1_trials(results)
    print(f"  Found {len(trials)} unique (question, replication) trials")

    # Create voting strategies
    strategies = [
        MajorityVoteStrategy(),
        WeightedMajorityVoteStrategy(weights_file=weights_file),
    ]

    if weights_file:
        print(f"  Using weights from: {weights_file}")
    else:
        print("  Using equal weights (no weights file)")

    print("\nApplying voting strategies to shared Stage 1 responses...")

    reanalysis_results = []

    for trial in trials:
        # Get chairman tiebreaker from original trial (for fair comparison)
        chairman_answer = trial.get("stage3_data", {}).get("chairman_tiebreaker")

        # Apply all strategies with same tiebreaker
        voting_results = apply_voting_strategies_to_trial(
            trial,
            strategies,
            chairman_answer=chairman_answer,
        )

        # Evaluate each strategy
        for strategy in strategies:
            strategy_result = voting_results[strategy.name]
            final_answer = strategy_result["final_answer"]

            is_correct = evaluate_answer(
                final_answer,
                trial["expected"],
                trial["benchmark"],
            )

            reanalysis_results.append({
                "question_id": trial["question_id"],
                "replication": trial["replication"],
                "benchmark": trial["benchmark"],
                "voting_strategy": strategy.name,
                "final_answer": final_answer,
                "expected": trial["expected"],
                "is_correct": is_correct,
            })

    df = pd.DataFrame(reanalysis_results)

    # Print summary
    print("\n" + "=" * 60)
    print("FAIR VOTING COMPARISON - REANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nTotal trials analyzed: {len(trials)}")
    print(f"Results per strategy: {len(trials)}")

    print("\n" + "-" * 40)
    print("ACCURACY BY VOTING STRATEGY")
    print("-" * 40)

    for strategy in strategies:
        strategy_df = df[df["voting_strategy"] == strategy.name]
        n_correct = strategy_df["is_correct"].sum()
        n_total = len(strategy_df)
        accuracy = n_correct / n_total if n_total > 0 else 0

        print(f"  {strategy.name}: {accuracy:.1%} ({n_correct}/{n_total})")

    # Pairwise comparison
    print("\n" + "-" * 40)
    print("PAIRWISE COMPARISON")
    print("-" * 40)

    mv_df = df[df["voting_strategy"] == "Majority Vote"].set_index(
        ["question_id", "replication"]
    )
    wv_df = df[df["voting_strategy"] == "Weighted Majority Vote"].set_index(
        ["question_id", "replication"]
    )

    common_idx = mv_df.index.intersection(wv_df.index)

    mv_correct = mv_df.loc[common_idx, "is_correct"]
    wv_correct = wv_df.loc[common_idx, "is_correct"]

    both_correct = (mv_correct & wv_correct).sum()
    mv_only = (mv_correct & ~wv_correct).sum()
    wv_only = (~mv_correct & wv_correct).sum()
    both_wrong = (~mv_correct & ~wv_correct).sum()

    print(f"\n  Both correct:             {both_correct}")
    print(f"  Majority Vote only:       {mv_only}")
    print(f"  Weighted Vote only:       {wv_only}")
    print(f"  Both wrong:               {both_wrong}")
    print(f"  Total pairs:              {len(common_idx)}")

    net_change = wv_only - mv_only
    print(f"\n  Net change from weighting: {net_change:+d} correct answers")

    if mv_only + wv_only > 0:
        from scipy import stats

        n_discordant = mv_only + wv_only
        if n_discordant < 25:
            p_value = stats.binomtest(
                mv_only, n_discordant, 0.5, alternative="two-sided"
            ).pvalue
        else:
            chi2 = (abs(mv_only - wv_only) - 1) ** 2 / n_discordant
            p_value = 1 - stats.chi2.cdf(chi2, df=1)

        print(f"\n  McNemar's test p-value:   {p_value:.4f}")
        if p_value < 0.05:
            print("  Result: Significant difference")
        else:
            print("  Result: No significant difference")

    print("\n" + "=" * 60)

    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_file, orient="records", indent=2)
        print(f"\nResults saved to: {output_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Reanalyze pilot results with fair voting comparison"
    )
    parser.add_argument(
        "--results-file",
        default="experiments/results/pilot_results.json",
        help="Path to pilot results JSON",
    )
    parser.add_argument(
        "--weights-file",
        default="experiments/results/model_weights.json",
        help="Path to model weights JSON",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to save reanalysis results",
    )

    args = parser.parse_args()

    reanalyze_pilot(
        results_file=args.results_file,
        weights_file=args.weights_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
