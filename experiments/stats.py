"""Statistical utilities for experiment analysis."""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def paired_bootstrap_accuracy_diff(
    df: pd.DataFrame,
    structure_a: str,
    structure_b: str,
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict:
    """
    Compute paired bootstrap confidence interval for accuracy difference.

    Uses per-question pairing: each question is evaluated by both structures,
    so we compute the difference in accuracy per question and bootstrap over questions.

    Args:
        df: DataFrame with columns 'structure', 'benchmark', 'question_id', 'is_correct'
        structure_a: Name of first structure
        structure_b: Name of second structure
        n_boot: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - diff: Observed mean accuracy difference (A - B)
        - ci_low: 2.5th percentile of bootstrap distribution
        - ci_high: 97.5th percentile of bootstrap distribution
        - p_value: Approximate two-sided p-value
        - n_questions: Number of questions used
    """
    rng = np.random.default_rng(seed)

    # Filter to just the two structures
    df_a = df[df["structure"] == structure_a]
    df_b = df[df["structure"] == structure_b]

    # Aggregate to per-question mean accuracy for each structure
    # (averaging over replications if any)
    acc_a = df_a.groupby(["benchmark", "question_id"])["is_correct"].mean()
    acc_b = df_b.groupby(["benchmark", "question_id"])["is_correct"].mean()

    # Get common questions
    common_idx = acc_a.index.intersection(acc_b.index)

    if len(common_idx) == 0:
        return {
            "diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "n_questions": 0,
        }

    # Compute per-question differences
    diff_per_q = acc_a.loc[common_idx].values - acc_b.loc[common_idx].values
    n_questions = len(diff_per_q)
    observed_diff = diff_per_q.mean()

    # Bootstrap
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        # Resample questions with replacement
        indices = rng.choice(n_questions, size=n_questions, replace=True)
        boot_diffs[i] = diff_per_q[indices].mean()

    # Compute percentile CI
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    # Approximate two-sided p-value using bootstrap
    # Under H0, the distribution is centered at 0
    # We use the proportion of bootstrap samples that are more extreme than observed
    centered_diffs = boot_diffs - observed_diff  # Center at observed
    p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))

    # Alternative: use proportion of samples on wrong side of zero
    # p_value = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))

    return {
        "diff": float(observed_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "n_questions": n_questions,
    }


def run_pairwise_bootstrap_tests(
    df: pd.DataFrame,
    baseline: str,
    structures: Optional[list] = None,
    n_boot: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Run paired bootstrap tests comparing each structure to a baseline.

    Args:
        df: DataFrame with results
        baseline: Name of baseline structure
        structures: List of structure names to compare (if None, uses all except baseline)
        n_boot: Number of bootstrap iterations
        seed: Random seed

    Returns:
        DataFrame with columns: structure, diff, ci_low, ci_high, p_value, n_questions
    """
    if structures is None:
        structures = [s for s in df["structure"].unique() if s != baseline]

    results = []
    for structure in structures:
        result = paired_bootstrap_accuracy_diff(
            df, structure, baseline, n_boot=n_boot, seed=seed
        )
        results.append(
            {
                "structure": structure,
                "baseline": baseline,
                **result,
            }
        )

    return pd.DataFrame(results)
