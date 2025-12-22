"""Analysis script for LLM council governance pilot study."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from experiments.run_pilot import load_results


def load_results_as_dataframe(
    output_dir: str = "experiments/results",
    filename: str = "pilot_results.json",
) -> pd.DataFrame:
    """
    Load experiment results as a pandas DataFrame.

    Args:
        output_dir: Directory containing results
        filename: Name of the results file

    Returns:
        DataFrame with experiment results
    """
    results = load_results(output_dir, filename)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def compute_accuracy_by_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each governance structure.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with structure, n_trials, n_correct, accuracy
    """
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(columns=["structure", "n_trials", "n_correct", "accuracy"])

    # Filter to only rows with valid is_correct values
    valid_df = df[df["is_correct"].notna()].copy()

    summary = (
        valid_df.groupby("structure")
        .agg(
            n_trials=("is_correct", "count"),
            n_correct=("is_correct", "sum"),
        )
        .reset_index()
    )

    summary["accuracy"] = summary["n_correct"] / summary["n_trials"]
    return summary


def compute_accuracy_by_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each benchmark.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with benchmark, n_trials, n_correct, accuracy
    """
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(columns=["benchmark", "n_trials", "n_correct", "accuracy"])

    valid_df = df[df["is_correct"].notna()].copy()

    summary = (
        valid_df.groupby("benchmark")
        .agg(
            n_trials=("is_correct", "count"),
            n_correct=("is_correct", "sum"),
        )
        .reset_index()
    )

    summary["accuracy"] = summary["n_correct"] / summary["n_trials"]
    return summary


def compute_accuracy_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each structure × benchmark combination.

    Args:
        df: DataFrame with experiment results

    Returns:
        Pivot table with structures as rows, benchmarks as columns, accuracy as values
    """
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame()

    valid_df = df[df["is_correct"].notna()].copy()

    # Group by structure and benchmark
    summary = (
        valid_df.groupby(["structure", "benchmark"])
        .agg(
            n_trials=("is_correct", "count"),
            n_correct=("is_correct", "sum"),
        )
        .reset_index()
    )

    summary["accuracy"] = summary["n_correct"] / summary["n_trials"]

    # Pivot to matrix form
    matrix = summary.pivot(index="structure", columns="benchmark", values="accuracy")

    return matrix


def compute_accuracy_with_ci(
    df: pd.DataFrame, confidence: float = 0.95
) -> pd.DataFrame:
    """
    Compute accuracy with confidence intervals for each structure.

    Uses Wilson score interval for binomial proportion.

    Args:
        df: DataFrame with experiment results
        confidence: Confidence level (default 0.95)

    Returns:
        DataFrame with structure, accuracy, ci_lower, ci_upper
    """
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(
            columns=["structure", "accuracy", "ci_lower", "ci_upper", "n_trials"]
        )

    valid_df = df[df["is_correct"].notna()].copy()

    results = []
    for structure in valid_df["structure"].unique():
        struct_df = valid_df[valid_df["structure"] == structure]
        n = len(struct_df)
        k = struct_df["is_correct"].sum()

        accuracy = k / n if n > 0 else 0
        ci_lower, ci_upper = wilson_score_interval(k, n, confidence)

        results.append(
            {
                "structure": structure,
                "accuracy": accuracy,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_trials": n,
            }
        )

    return pd.DataFrame(results)


def wilson_score_interval(
    k: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for binomial proportion.

    Args:
        k: Number of successes
        n: Number of trials
        confidence: Confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n == 0:
        return (0.0, 0.0)

    from scipy import stats

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n

    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) + z**2 / (4 * n)) / n) ** 0.5

    lower = (center - margin) / denominator
    upper = (center + margin) / denominator

    return (max(0, lower), min(1, upper))


def run_chi_square_test(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Run chi-square test for independence between structure and correctness.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with chi2 statistic, p-value, and interpretation, or None if test cannot be run
    """
    if df.empty or "is_correct" not in df.columns or "structure" not in df.columns:
        return None

    valid_df = df[df["is_correct"].notna()].copy()

    if len(valid_df) < 5 or valid_df["structure"].nunique() < 2:
        return None

    from scipy import stats

    # Create contingency table
    contingency = pd.crosstab(valid_df["structure"], valid_df["is_correct"])

    # Check minimum expected frequency
    if contingency.min().min() < 5:
        # Use Fisher's exact test for 2x2, otherwise note the limitation
        if contingency.shape == (2, 2):
            _, p_value = stats.fisher_exact(contingency)
            return {
                "test": "Fisher's exact test",
                "p_value": p_value,
                "significant": p_value < 0.05,
                "interpretation": (
                    "Significant difference between structures"
                    if p_value < 0.05
                    else "No significant difference between structures"
                ),
            }

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    return {
        "test": "Chi-square test",
        "chi2": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "significant": p_value < 0.05,
        "interpretation": (
            "Significant difference between structures"
            if p_value < 0.05
            else "No significant difference between structures"
        ),
    }


def run_pairwise_tests(
    df: pd.DataFrame, alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Run pairwise proportion tests between structures.

    Uses two-proportion z-test with Bonferroni correction.

    Args:
        df: DataFrame with experiment results
        alpha: Significance level before correction

    Returns:
        List of test results for each pair
    """
    if df.empty or "is_correct" not in df.columns:
        return []

    valid_df = df[df["is_correct"].notna()].copy()
    structures = valid_df["structure"].unique()

    if len(structures) < 2:
        return []

    from itertools import combinations

    from scipy import stats

    results = []
    n_comparisons = len(list(combinations(structures, 2)))
    corrected_alpha = alpha / n_comparisons  # Bonferroni correction

    for s1, s2 in combinations(structures, 2):
        df1 = valid_df[valid_df["structure"] == s1]
        df2 = valid_df[valid_df["structure"] == s2]

        n1, k1 = len(df1), df1["is_correct"].sum()
        n2, k2 = len(df2), df2["is_correct"].sum()

        p1 = k1 / n1 if n1 > 0 else 0
        p2 = k2 / n2 if n2 > 0 else 0

        # Pooled proportion
        p_pool = (k1 + k2) / (n1 + n2) if (n1 + n2) > 0 else 0

        # Standard error
        se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5 if p_pool > 0 else 0

        # Z-statistic
        z = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        results.append(
            {
                "structure_1": s1,
                "structure_2": s2,
                "accuracy_1": p1,
                "accuracy_2": p2,
                "difference": p1 - p2,
                "z_statistic": z,
                "p_value": p_value,
                "p_value_corrected": min(p_value * n_comparisons, 1.0),
                "significant": p_value < corrected_alpha,
            }
        )

    return results


def compute_timing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute timing statistics by structure.

    Args:
        df: DataFrame with experiment results (must have elapsed_time column)

    Returns:
        DataFrame with timing statistics per structure
    """
    if df.empty or "elapsed_time" not in df.columns:
        return pd.DataFrame(
            columns=["structure", "mean_time", "std_time", "min_time", "max_time"]
        )

    valid_df = df[df["elapsed_time"].notna()].copy()

    summary = (
        valid_df.groupby("structure")["elapsed_time"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    summary.columns = [
        "structure",
        "mean_time",
        "std_time",
        "min_time",
        "max_time",
        "n_trials",
    ]

    return summary


def _extract_number_from_response(response: str) -> Optional[str]:
    """
    Extract a numerical answer from a model response (for GSM8K).
    """
    # Try FINAL ANSWER pattern first
    match = re.search(
        r"FINAL ANSWER:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", response, re.IGNORECASE
    )
    if match:
        return _normalize_number(match.group(1))

    # Try to find any number after "answer is" or similar
    match = re.search(
        r"(?:answer is|equals?|=)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        response,
        re.IGNORECASE,
    )
    if match:
        return _normalize_number(match.group(1))

    # Fallback: find the last number in the response
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", response)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def _normalize_number(num_str: str) -> str:
    """Normalize a number string for comparison."""
    num_str = num_str.replace(",", "")
    try:
        num = float(num_str)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return num_str


def _extract_letter_from_response(response: str) -> Optional[str]:
    """
    Extract a letter answer from a model response (for TruthfulQA).
    """
    # Try FINAL ANSWER pattern first
    match = re.search(
        r"FINAL ANSWER:\s*([A-Ja-j])\b", response, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Try "answer is [letter]" pattern
    match = re.search(
        r"(?:answer is|choose|select)\s*([A-Ja-j])\b", response, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Try to find a standalone letter at end
    match = re.search(r"\b([A-Ja-j])\s*[.)]?\s*$", response.strip())
    if match:
        return match.group(1).upper()

    # Fallback: find first letter A-J
    match = re.search(r"\b([A-Ja-j])\b", response)
    if match:
        return match.group(1).upper()

    return None


def _evaluate_individual_response(response: str, expected: str, benchmark: str) -> bool:
    """
    Evaluate if an individual model's response is correct.
    """
    if not response or not expected:
        return False

    if benchmark == "GSM8K":
        predicted = _extract_number_from_response(response)
        if predicted is None:
            return False
        return predicted == _normalize_number(expected)
    else:  # TruthfulQA
        predicted = _extract_letter_from_response(response)
        if predicted is None:
            return False
        return predicted.upper() == expected.upper()


def compute_individual_model_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each individual model acting alone.

    Extracts each model's stage1 response and evaluates it against the expected answer.

    Args:
        df: DataFrame with experiment results (must have stage1_responses, expected, benchmark)

    Returns:
        DataFrame with model, n_trials, n_correct, accuracy
    """
    if df.empty or "stage1_responses" not in df.columns:
        return pd.DataFrame(columns=["model", "n_trials", "n_correct", "accuracy"])

    # Filter to valid rows with stage1_responses
    valid_df = df[df["stage1_responses"].notna()].copy()

    # We need to avoid double-counting since same question appears for each structure
    # Group by question_id, replication, benchmark to get unique question instances
    # Use only rows from one structure to avoid duplication
    first_structure = valid_df["structure"].iloc[0] if len(valid_df) > 0 else None
    unique_df = valid_df[valid_df["structure"] == first_structure].copy()

    model_results = {}

    for _, row in unique_df.iterrows():
        stage1 = row.get("stage1_responses", {})
        expected = row.get("expected", "")
        benchmark = row.get("benchmark", "")

        if not isinstance(stage1, dict):
            continue

        for model, response in stage1.items():
            if model not in model_results:
                model_results[model] = {"n_trials": 0, "n_correct": 0}

            model_results[model]["n_trials"] += 1
            if _evaluate_individual_response(response, expected, benchmark):
                model_results[model]["n_correct"] += 1

    # Build results DataFrame
    results = []
    for model, stats in model_results.items():
        n_trials = stats["n_trials"]
        n_correct = stats["n_correct"]
        accuracy = n_correct / n_trials if n_trials > 0 else 0
        results.append({
            "model": model,
            "n_trials": n_trials,
            "n_correct": n_correct,
            "accuracy": accuracy,
        })

    return pd.DataFrame(results).sort_values("accuracy", ascending=False)


def compute_individual_model_accuracy_by_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each individual model, broken down by benchmark.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with model, benchmark, n_trials, n_correct, accuracy
    """
    if df.empty or "stage1_responses" not in df.columns:
        return pd.DataFrame(columns=["model", "benchmark", "n_trials", "n_correct", "accuracy"])

    valid_df = df[df["stage1_responses"].notna()].copy()

    # Use only rows from one structure to avoid duplication
    first_structure = valid_df["structure"].iloc[0] if len(valid_df) > 0 else None
    unique_df = valid_df[valid_df["structure"] == first_structure].copy()

    model_results = {}

    for _, row in unique_df.iterrows():
        stage1 = row.get("stage1_responses", {})
        expected = row.get("expected", "")
        benchmark = row.get("benchmark", "")

        if not isinstance(stage1, dict):
            continue

        for model, response in stage1.items():
            key = (model, benchmark)
            if key not in model_results:
                model_results[key] = {"n_trials": 0, "n_correct": 0}

            model_results[key]["n_trials"] += 1
            if _evaluate_individual_response(response, expected, benchmark):
                model_results[key]["n_correct"] += 1

    results = []
    for (model, benchmark), stats in model_results.items():
        n_trials = stats["n_trials"]
        n_correct = stats["n_correct"]
        accuracy = n_correct / n_trials if n_trials > 0 else 0
        results.append({
            "model": model,
            "benchmark": benchmark,
            "n_trials": n_trials,
            "n_correct": n_correct,
            "accuracy": accuracy,
        })

    return pd.DataFrame(results)


def analyze_deliberation_changes(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how models change answers after deliberation.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with deliberation analysis results
    """
    if df.empty or "stage1_responses" not in df.columns:
        return {}

    # Filter to deliberation structures only
    delib_df = df[df["structure"].str.contains("Deliberate", na=False)].copy()
    if delib_df.empty:
        return {}

    model_stats = {}
    total_changes = 0
    changes_to_correct = 0
    changes_to_wrong = 0
    agreement_before = []
    agreement_after = []

    from collections import Counter

    for _, row in delib_df.iterrows():
        stage1 = row.get("stage1_responses", {})
        stage2 = row.get("stage2_data", {})
        expected = row.get("expected", "")
        benchmark = row.get("benchmark", "")

        if not isinstance(stage2, dict) or "deliberation_responses" not in stage2:
            continue

        delib_responses = stage2["deliberation_responses"]

        # Get initial and post-deliberation answers
        initial_answers = []
        post_answers = []

        for model in stage1.keys():
            if model not in delib_responses:
                continue

            initial_answer = _evaluate_individual_response(stage1[model], expected, benchmark)
            post_answer = _evaluate_individual_response(delib_responses[model], expected, benchmark)

            # Extract actual answers for agreement calculation
            if benchmark == "GSM8K":
                init_ans = _extract_number_from_response(stage1[model])
                post_ans = _extract_number_from_response(delib_responses[model])
            else:
                init_ans = _extract_letter_from_response(stage1[model])
                post_ans = _extract_letter_from_response(delib_responses[model])

            if init_ans:
                initial_answers.append(init_ans)
            if post_ans:
                post_answers.append(post_ans)

            if init_ans is None or post_ans is None:
                continue

            # Initialize model stats
            if model not in model_stats:
                model_stats[model] = {
                    "total": 0,
                    "changed": 0,
                    "changed_to_correct": 0,
                    "changed_to_wrong": 0,
                    "was_correct_stayed": 0,
                    "was_correct_broke": 0,
                    "was_wrong_fixed": 0,
                    "was_wrong_stayed": 0,
                }

            model_stats[model]["total"] += 1
            changed = init_ans != post_ans
            init_correct = initial_answer  # This is already a bool from _evaluate
            post_correct = post_answer

            # Re-evaluate correctness properly
            if benchmark == "GSM8K":
                init_correct = init_ans == _normalize_number(expected) if expected else False
                post_correct = post_ans == _normalize_number(expected) if expected else False
            else:
                init_correct = init_ans.upper() == expected.upper() if init_ans and expected else False
                post_correct = post_ans.upper() == expected.upper() if post_ans and expected else False

            if changed:
                model_stats[model]["changed"] += 1
                total_changes += 1
                if post_correct and not init_correct:
                    model_stats[model]["changed_to_correct"] += 1
                    model_stats[model]["was_wrong_fixed"] += 1
                    changes_to_correct += 1
                elif not post_correct and init_correct:
                    model_stats[model]["changed_to_wrong"] += 1
                    model_stats[model]["was_correct_broke"] += 1
                    changes_to_wrong += 1
                elif not init_correct:
                    model_stats[model]["was_wrong_stayed"] += 1
            else:
                if init_correct:
                    model_stats[model]["was_correct_stayed"] += 1
                else:
                    model_stats[model]["was_wrong_stayed"] += 1

        # Calculate agreement levels
        if initial_answers:
            counts = Counter(initial_answers)
            agreement_before.append(counts.most_common(1)[0][1] / len(initial_answers))
        if post_answers:
            counts = Counter(post_answers)
            agreement_after.append(counts.most_common(1)[0][1] / len(post_answers))

    return {
        "model_stats": model_stats,
        "total_changes": total_changes,
        "changes_to_correct": changes_to_correct,
        "changes_to_wrong": changes_to_wrong,
        "net_benefit": changes_to_correct - changes_to_wrong,
        "avg_agreement_before": sum(agreement_before) / len(agreement_before) if agreement_before else 0,
        "avg_agreement_after": sum(agreement_after) / len(agreement_after) if agreement_after else 0,
    }


def run_ttest_vs_best_individual(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Run t-tests comparing each governance structure vs the best individual model.

    Uses paired t-test since the same questions are evaluated.

    Args:
        df: DataFrame with experiment results

    Returns:
        List of t-test results for each structure
    """
    if df.empty or "stage1_responses" not in df.columns:
        return []

    from scipy import stats

    valid_df = df[df["is_correct"].notna() & df["stage1_responses"].notna()].copy()

    # First, compute individual model accuracy to find the best one
    model_acc = compute_individual_model_accuracy(df)
    if model_acc.empty:
        return []

    best_model = model_acc.iloc[0]["model"]
    best_model_acc = model_acc.iloc[0]["accuracy"]

    # Build a mapping of (question_id, replication) -> best model correctness
    first_structure = valid_df["structure"].iloc[0]
    unique_df = valid_df[valid_df["structure"] == first_structure].copy()

    best_model_correctness = {}
    for _, row in unique_df.iterrows():
        key = (row["question_id"], row["replication"])
        stage1 = row.get("stage1_responses", {})
        expected = row.get("expected", "")
        benchmark = row.get("benchmark", "")

        if isinstance(stage1, dict) and best_model in stage1:
            response = stage1[best_model]
            correct = _evaluate_individual_response(response, expected, benchmark)
            best_model_correctness[key] = 1 if correct else 0

    # Run t-tests for each structure
    results = []
    structures = valid_df["structure"].unique()

    for structure in structures:
        struct_df = valid_df[valid_df["structure"] == structure]

        # Build paired arrays
        struct_correct = []
        model_correct = []

        for _, row in struct_df.iterrows():
            key = (row["question_id"], row["replication"])
            if key in best_model_correctness:
                struct_correct.append(1 if row["is_correct"] else 0)
                model_correct.append(best_model_correctness[key])

        if len(struct_correct) < 2:
            continue

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(struct_correct, model_correct)

        struct_mean = sum(struct_correct) / len(struct_correct)
        model_mean = sum(model_correct) / len(model_correct)
        diff = struct_mean - model_mean

        results.append({
            "structure": structure,
            "structure_accuracy": struct_mean,
            "best_model": best_model,
            "best_model_accuracy": model_mean,
            "difference": diff,
            "t_statistic": t_stat,
            "p_value": p_value,
            "n_pairs": len(struct_correct),
            "significant": p_value < 0.05,
        })

    return results


def generate_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a full analysis report.

    Args:
        df: DataFrame with experiment results
        output_path: Optional path to save the report

    Returns:
        Report as a string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("LLM COUNCIL GOVERNANCE PILOT STUDY - ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    if df.empty:
        lines.append("No results to analyze.")
        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report)
        return report

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total trials: {len(df)}")
    lines.append(f"Structures tested: {df['structure'].nunique()}")
    lines.append(f"Benchmarks used: {df['benchmark'].nunique()}")

    valid_df = df[df["is_correct"].notna()]
    if len(valid_df) > 0:
        lines.append(f"Valid trials (with results): {len(valid_df)}")
        lines.append(f"Overall accuracy: {valid_df['is_correct'].mean():.1%}")
    lines.append("")

    # Individual model accuracy
    lines.append("INDIVIDUAL MODEL ACCURACY")
    lines.append("-" * 40)
    model_acc = compute_individual_model_accuracy(df)
    if not model_acc.empty:
        for _, row in model_acc.iterrows():
            # Shorten model name for display
            short_name = row["model"].split("/")[-1] if "/" in row["model"] else row["model"]
            lines.append(
                f"  {short_name}: {row['accuracy']:.1%} "
                f"({int(row['n_correct'])}/{int(row['n_trials'])})"
            )
    else:
        lines.append("  (No individual model data available)")
    lines.append("")

    # Individual model accuracy by benchmark
    lines.append("INDIVIDUAL MODEL ACCURACY BY BENCHMARK")
    lines.append("-" * 40)
    model_bench_acc = compute_individual_model_accuracy_by_benchmark(df)
    if not model_bench_acc.empty:
        for benchmark in model_bench_acc["benchmark"].unique():
            lines.append(f"  {benchmark}:")
            bench_df = model_bench_acc[model_bench_acc["benchmark"] == benchmark].sort_values(
                "accuracy", ascending=False
            )
            for _, row in bench_df.iterrows():
                short_name = row["model"].split("/")[-1] if "/" in row["model"] else row["model"]
                lines.append(
                    f"    {short_name}: {row['accuracy']:.1%} "
                    f"({int(row['n_correct'])}/{int(row['n_trials'])})"
                )
    else:
        lines.append("  (No individual model data available)")
    lines.append("")

    # Accuracy by structure
    lines.append("ACCURACY BY STRUCTURE")
    lines.append("-" * 40)
    struct_acc = compute_accuracy_by_structure(df)
    if not struct_acc.empty:
        for _, row in struct_acc.iterrows():
            lines.append(
                f"  {row['structure']}: {row['accuracy']:.1%} "
                f"({int(row['n_correct'])}/{int(row['n_trials'])})"
            )
    lines.append("")

    # Accuracy by benchmark
    lines.append("ACCURACY BY BENCHMARK")
    lines.append("-" * 40)
    bench_acc = compute_accuracy_by_benchmark(df)
    if not bench_acc.empty:
        for _, row in bench_acc.iterrows():
            lines.append(
                f"  {row['benchmark']}: {row['accuracy']:.1%} "
                f"({int(row['n_correct'])}/{int(row['n_trials'])})"
            )
    lines.append("")

    # Accuracy matrix
    lines.append("ACCURACY MATRIX (Structure x Benchmark)")
    lines.append("-" * 40)
    matrix = compute_accuracy_matrix(df)
    if not matrix.empty:
        # Format as percentage
        matrix_pct = matrix.map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        lines.append(matrix_pct.to_string())
    lines.append("")

    # Confidence intervals
    lines.append("ACCURACY WITH 95% CONFIDENCE INTERVALS")
    lines.append("-" * 40)
    try:
        ci_df = compute_accuracy_with_ci(df)
        if not ci_df.empty:
            for _, row in ci_df.iterrows():
                lines.append(
                    f"  {row['structure']}: {row['accuracy']:.1%} "
                    f"[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]"
                )
    except ImportError:
        lines.append("  (scipy not available for CI computation)")
    lines.append("")

    # Statistical tests
    lines.append("STATISTICAL TESTS")
    lines.append("-" * 40)
    try:
        chi2_result = run_chi_square_test(df)
        if chi2_result:
            lines.append(f"  {chi2_result['test']}:")
            if "chi2" in chi2_result:
                lines.append(f"    Chi-square statistic: {chi2_result['chi2']:.2f}")
            lines.append(f"    p-value: {chi2_result['p_value']:.4f}")
            lines.append(f"    Interpretation: {chi2_result['interpretation']}")
        else:
            lines.append("  Insufficient data for chi-square test")
    except ImportError:
        lines.append("  (scipy not available for statistical tests)")
    lines.append("")

    # Pairwise comparisons
    lines.append("PAIRWISE COMPARISONS (Bonferroni corrected)")
    lines.append("-" * 40)
    try:
        pairwise = run_pairwise_tests(df)
        if pairwise:
            for result in pairwise:
                sig = "*" if result["significant"] else ""
                lines.append(
                    f"  {result['structure_1']} vs {result['structure_2']}: "
                    f"{result['difference']:+.1%} (p={result['p_value_corrected']:.4f}){sig}"
                )
        else:
            lines.append("  Insufficient data for pairwise tests")
    except ImportError:
        lines.append("  (scipy not available for pairwise tests)")
    lines.append("")

    # T-tests vs best individual model
    lines.append("T-TESTS: STRUCTURES VS BEST INDIVIDUAL MODEL")
    lines.append("-" * 40)
    try:
        ttest_results = run_ttest_vs_best_individual(df)
        if ttest_results:
            best_model = ttest_results[0]["best_model"]
            best_model_short = best_model.split("/")[-1] if "/" in best_model else best_model
            best_acc = ttest_results[0]["best_model_accuracy"]
            lines.append(f"  Best individual model: {best_model_short} ({best_acc:.1%})")
            lines.append("")
            for result in ttest_results:
                sig = "*" if result["significant"] else ""
                # Shorten structure name for readability
                struct_short = result["structure"].replace("Independent → ", "")
                lines.append(
                    f"  {struct_short}: {result['structure_accuracy']:.1%} vs {result['best_model_accuracy']:.1%} "
                    f"(diff={result['difference']:+.1%}, t={result['t_statistic']:.2f}, p={result['p_value']:.4f}){sig}"
                )
        else:
            lines.append("  Insufficient data for t-tests")
    except ImportError:
        lines.append("  (scipy not available for t-tests)")
    lines.append("")

    # Deliberation analysis
    lines.append("DELIBERATION ANALYSIS")
    lines.append("-" * 40)
    delib_analysis = analyze_deliberation_changes(df)
    if delib_analysis:
        lines.append("  Answer changes after deliberation:")
        lines.append(f"    Total changes:        {delib_analysis['total_changes']}")
        lines.append(f"    Changed to correct:   {delib_analysis['changes_to_correct']}")
        lines.append(f"    Changed to wrong:     {delib_analysis['changes_to_wrong']}")
        lines.append(f"    Net benefit:          {delib_analysis['net_benefit']:+d} answers")
        lines.append("")
        lines.append("  Agreement levels (groupthink indicator):")
        lines.append(f"    Before deliberation:  {delib_analysis['avg_agreement_before']:.1%}")
        lines.append(f"    After deliberation:   {delib_analysis['avg_agreement_after']:.1%}")
        lines.append(f"    Change:               {(delib_analysis['avg_agreement_after'] - delib_analysis['avg_agreement_before'])*100:+.1f}%")
        lines.append("")
        lines.append("  By model (change rate | fix rate when wrong | break rate when correct):")
        for model, stats in delib_analysis["model_stats"].items():
            short_name = model.split("/")[-1] if "/" in model else model
            change_rate = stats["changed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            wrong_total = stats["was_wrong_fixed"] + stats["was_wrong_stayed"]
            fix_rate = stats["was_wrong_fixed"] / wrong_total * 100 if wrong_total > 0 else 0
            correct_total = stats["was_correct_stayed"] + stats["was_correct_broke"]
            break_rate = stats["was_correct_broke"] / correct_total * 100 if correct_total > 0 else 0
            lines.append(f"    {short_name}: {change_rate:.1f}% | {fix_rate:.1f}% | {break_rate:.1f}%")
        lines.append("")
        lines.append("  Interpretation:")
        if delib_analysis['avg_agreement_after'] > delib_analysis['avg_agreement_before']:
            lines.append("    Deliberation increases agreement (potential groupthink)")
        if delib_analysis['net_benefit'] > 0:
            lines.append(f"    Net positive: deliberation fixes more than it breaks (+{delib_analysis['net_benefit']})")
        else:
            lines.append(f"    Net negative: deliberation breaks more than it fixes ({delib_analysis['net_benefit']})")
        lines.append("    Simple majority vote may outperform deliberation by preserving independence")
    else:
        lines.append("  (No deliberation data available)")
    lines.append("")

    # Timing summary
    if "elapsed_time" in df.columns:
        lines.append("TIMING SUMMARY (seconds)")
        lines.append("-" * 40)
        timing = compute_timing_summary(df)
        if not timing.empty:
            for _, row in timing.iterrows():
                lines.append(
                    f"  {row['structure']}: "
                    f"mean={row['mean_time']:.2f}s, "
                    f"std={row['std_time']:.2f}s"
                )
        lines.append("")

    # Error summary
    if "error" in df.columns:
        error_df = df[df["error"].notna()]
        if len(error_df) > 0:
            lines.append("ERRORS")
            lines.append("-" * 40)
            lines.append(f"  Total errors: {len(error_df)}")
            error_counts = error_df.groupby("structure").size()
            for struct, count in error_counts.items():
                lines.append(f"    {struct}: {count}")
            lines.append("")

    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)

    return report


def generate_accuracy_by_structure_chart(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate a bar chart of accuracy by governance structure.

    Args:
        df: DataFrame with experiment results
        output_path: Path to save the chart image
    """
    import matplotlib.pyplot as plt

    struct_acc = compute_accuracy_by_structure(df)
    if struct_acc.empty:
        return

    # Sort by accuracy descending
    struct_acc = struct_acc.sort_values("accuracy", ascending=True)

    # Create short labels for structures
    short_labels = {
        "Independent → Rank → Synthesize": "A: Rank→Synth",
        "Independent → Majority Vote": "B: Majority Vote",
        "Independent → Deliberate → Vote": "C: Delib→Vote",
        "Independent → Deliberate → Synthesize": "D: Delib→Synth",
    }
    struct_acc["short_name"] = struct_acc["structure"].map(
        lambda x: short_labels.get(x, x[:20])
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e74c3c" if acc < 0.7 else "#f39c12" if acc < 0.9 else "#27ae60"
              for acc in struct_acc["accuracy"]]

    bars = ax.barh(struct_acc["short_name"], struct_acc["accuracy"], color=colors)

    # Add value labels on bars
    for bar, acc, n_correct, n_trials in zip(
        bars,
        struct_acc["accuracy"],
        struct_acc["n_correct"],
        struct_acc["n_trials"]
    ):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%} ({int(n_correct)}/{int(n_trials)})",
            va="center",
            fontsize=10,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Governance Structure", fontsize=14, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.5, label="90%")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_accuracy_by_benchmark_chart(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate a bar chart of accuracy by benchmark.

    Args:
        df: DataFrame with experiment results
        output_path: Path to save the chart image
    """
    import matplotlib.pyplot as plt

    bench_acc = compute_accuracy_by_benchmark(df)
    if bench_acc.empty:
        return

    # Sort by accuracy descending
    bench_acc = bench_acc.sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#3498db" for _ in range(len(bench_acc))]

    bars = ax.barh(bench_acc["benchmark"], bench_acc["accuracy"], color=colors)

    # Add value labels on bars
    for bar, acc, n_correct, n_trials in zip(
        bars,
        bench_acc["accuracy"],
        bench_acc["n_correct"],
        bench_acc["n_trials"]
    ):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%} ({int(n_correct)}/{int(n_trials)})",
            va="center",
            fontsize=10,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Benchmark", fontsize=14, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_accuracy_matrix_chart(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate a grouped bar chart of accuracy by structure and benchmark.

    Args:
        df: DataFrame with experiment results
        output_path: Path to save the chart image
    """
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = compute_accuracy_matrix(df)
    if matrix.empty:
        return

    # Short labels for structures
    short_labels = {
        "Independent → Rank → Synthesize": "A: Rank→Synth",
        "Independent → Majority Vote": "B: Majority Vote",
        "Independent → Deliberate → Vote": "C: Delib→Vote",
        "Independent → Deliberate → Synthesize": "D: Delib→Synth",
    }

    structures = [short_labels.get(s, s[:20]) for s in matrix.index]
    benchmarks = list(matrix.columns)

    x = np.arange(len(structures))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

    for i, benchmark in enumerate(benchmarks):
        offset = width * (i - len(benchmarks) / 2 + 0.5)
        values = [matrix.loc[s, benchmark] if pd.notna(matrix.loc[s, benchmark]) else 0
                  for s in matrix.index]
        bars = ax.bar(x + offset, values, width, label=benchmark, color=colors[i % len(colors)])

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Structure and Benchmark", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=15, ha="right")
    ax.legend(title="Benchmark")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_council_vs_individual_chart(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate a chart comparing council structures vs individual models by benchmark.

    Args:
        df: DataFrame with experiment results
        output_path: Path to save the chart image
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get individual model accuracy by benchmark
    model_bench_acc = compute_individual_model_accuracy_by_benchmark(df)
    if model_bench_acc.empty:
        return

    # Get structure accuracy by benchmark
    struct_matrix = compute_accuracy_matrix(df)
    if struct_matrix.empty:
        return

    benchmarks = list(struct_matrix.columns)

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(14, 6))
    if len(benchmarks) == 1:
        axes = [axes]

    colors_individual = ["#e74c3c", "#e67e22", "#f39c12", "#d35400"]  # Reds/oranges for individuals
    colors_council = ["#27ae60", "#2ecc71", "#1abc9c", "#16a085"]  # Greens for councils

    # Short labels
    structure_short = {
        "Independent → Rank → Synthesize": "A: Rank→Synth",
        "Independent → Majority Vote": "B: Majority Vote",
        "Independent → Deliberate → Vote": "C: Delib→Vote",
        "Independent → Deliberate → Synthesize": "D: Delib→Synth",
    }

    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]

        # Individual models for this benchmark
        bench_models = model_bench_acc[model_bench_acc["benchmark"] == benchmark].copy()
        bench_models = bench_models.sort_values("accuracy", ascending=True)

        # Council structures for this benchmark
        struct_acc = []
        for structure in struct_matrix.index:
            acc = struct_matrix.loc[structure, benchmark]
            if pd.notna(acc):
                struct_acc.append({
                    "name": structure_short.get(structure, structure[:15]),
                    "accuracy": acc,
                    "type": "council"
                })

        # Combine data
        all_data = []

        # Add individual models
        for i, (_, row) in enumerate(bench_models.iterrows()):
            short_name = row["model"].split("/")[-1] if "/" in row["model"] else row["model"]
            short_name = short_name.replace("-instruct", "").replace("-it", "")
            all_data.append({
                "name": short_name,
                "accuracy": row["accuracy"],
                "type": "individual",
                "color": colors_individual[i % len(colors_individual)]
            })

        # Add council structures (sorted by accuracy)
        struct_acc_sorted = sorted(struct_acc, key=lambda x: x["accuracy"])
        for i, item in enumerate(struct_acc_sorted):
            all_data.append({
                "name": item["name"],
                "accuracy": item["accuracy"],
                "type": "council",
                "color": colors_council[i % len(colors_council)]
            })

        # Plot
        names = [d["name"] for d in all_data]
        accuracies = [d["accuracy"] for d in all_data]
        colors = [d["color"] for d in all_data]

        bars = ax.barh(names, accuracies, color=colors)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}",
                va="center",
                fontsize=9,
            )

        # Add dividing line between individuals and councils
        n_individuals = sum(1 for d in all_data if d["type"] == "individual")
        if n_individuals > 0 and n_individuals < len(all_data):
            ax.axhline(y=n_individuals - 0.5, color="gray", linestyle="--", alpha=0.7)

        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Accuracy", fontsize=11)
        ax.set_title(f"{benchmark}", fontsize=13, fontweight="bold")
        ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Individual Models"),
        Patch(facecolor="#27ae60", label="Council Structures"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Council Structures vs Individual Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_charts(
    df: pd.DataFrame,
    output_dir: str,
) -> List[str]:
    """
    Generate all analysis charts.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save chart images

    Returns:
        List of paths to generated chart files
    """
    if df.empty:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    charts = []

    # Accuracy by structure
    struct_chart = str(output_path / "accuracy_by_structure.png")
    generate_accuracy_by_structure_chart(df, struct_chart)
    charts.append(struct_chart)
    print(f"  Saved: {struct_chart}")

    # Accuracy by benchmark
    bench_chart = str(output_path / "accuracy_by_benchmark.png")
    generate_accuracy_by_benchmark_chart(df, bench_chart)
    charts.append(bench_chart)
    print(f"  Saved: {bench_chart}")

    # Accuracy matrix (structure x benchmark)
    if df["benchmark"].nunique() > 1:
        matrix_chart = str(output_path / "accuracy_matrix.png")
        generate_accuracy_matrix_chart(df, matrix_chart)
        charts.append(matrix_chart)
        print(f"  Saved: {matrix_chart}")

    # Council vs individual models comparison
    if df["benchmark"].nunique() >= 1:
        comparison_chart = str(output_path / "council_vs_individual.png")
        generate_council_vs_individual_chart(df, comparison_chart)
        charts.append(comparison_chart)
        print(f"  Saved: {comparison_chart}")

    return charts


def analyze_pilot(
    output_dir: str = "experiments/results",
    report_path: Optional[str] = None,
    generate_charts_flag: bool = True,
) -> pd.DataFrame:
    """
    Analyze pilot study results and generate report.

    Args:
        output_dir: Directory containing results
        report_path: Optional path to save report (default: output_dir/analysis_report.txt)
        generate_charts_flag: Whether to generate chart images

    Returns:
        DataFrame with experiment results
    """
    df = load_results_as_dataframe(output_dir)

    if report_path is None:
        report_path = str(Path(output_dir) / "analysis_report.txt")

    report = generate_report(df, report_path)
    print(report)

    if generate_charts_flag and not df.empty:
        print("\nGenerating charts...")
        generate_charts(df, output_dir)

    return df


if __name__ == "__main__":
    analyze_pilot()
