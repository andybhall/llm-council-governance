"""Analysis script for prompt variant experiment.

Compares prompt variant council performance against single model baseline
from the pilot study.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from backend.prompt_variants import BASE_MODEL, VARIANT_MODELS


def load_prompt_experiment_results(
    results_dir: str = "experiments/results_prompt_variants",
    filename: str = "prompt_experiment_results.json",
) -> pd.DataFrame:
    """Load prompt experiment results as DataFrame."""
    filepath = Path(results_dir) / filename
    if not filepath.exists():
        return pd.DataFrame()

    with open(filepath) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_pilot_results(
    results_dir: str = "experiments/results",
    filename: str = "pilot_results.json",
) -> pd.DataFrame:
    """Load pilot study results as DataFrame."""
    filepath = Path(results_dir) / filename
    if not filepath.exists():
        return pd.DataFrame()

    with open(filepath) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def compute_accuracy_by_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy for each governance structure."""
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(columns=["structure", "n_trials", "n_correct", "accuracy"])

    valid_df = df[df["is_correct"].notna()].copy()

    results = []
    for structure in valid_df["structure"].unique():
        struct_df = valid_df[valid_df["structure"] == structure]
        n_trials = len(struct_df)
        n_correct = struct_df["is_correct"].sum()
        accuracy = n_correct / n_trials if n_trials > 0 else 0

        results.append({
            "structure": structure,
            "n_trials": n_trials,
            "n_correct": int(n_correct),
            "accuracy": accuracy,
        })

    return pd.DataFrame(results).sort_values("accuracy", ascending=False)


def compute_accuracy_by_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy for each benchmark."""
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(columns=["benchmark", "n_trials", "n_correct", "accuracy"])

    valid_df = df[df["is_correct"].notna()].copy()

    results = []
    for benchmark in valid_df["benchmark"].unique():
        bench_df = valid_df[valid_df["benchmark"] == benchmark]
        n_trials = len(bench_df)
        n_correct = bench_df["is_correct"].sum()
        accuracy = n_correct / n_trials if n_trials > 0 else 0

        results.append({
            "benchmark": benchmark,
            "n_trials": n_trials,
            "n_correct": int(n_correct),
            "accuracy": accuracy,
        })

    return pd.DataFrame(results).sort_values("accuracy", ascending=False)


def compute_accuracy_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy matrix (structure x benchmark)."""
    if df.empty:
        return pd.DataFrame()

    valid_df = df[df["is_correct"].notna()].copy()

    # Compute accuracy per structure-benchmark combination
    pivot_data = valid_df.groupby(["structure", "benchmark"])["is_correct"].mean()
    return pivot_data.unstack(fill_value=0)


def get_baseline_model_accuracy(
    pilot_df: pd.DataFrame,
    model: str = BASE_MODEL,
) -> Dict[str, float]:
    """
    Extract baseline model accuracy from pilot study stage1_responses.

    Returns dict with 'overall', 'GSM8K', and 'TruthfulQA' accuracy.
    """
    if pilot_df.empty or "stage1_responses" not in pilot_df.columns:
        return {"overall": None, "GSM8K": None, "TruthfulQA": None}

    valid_df = pilot_df[pilot_df["stage1_responses"].notna()].copy()

    # Import evaluation helpers from analyze_pilot
    from experiments.analyze_pilot import _evaluate_individual_response

    results_by_benchmark = {"GSM8K": {"n": 0, "correct": 0}, "TruthfulQA": {"n": 0, "correct": 0}}
    total_n = 0
    total_correct = 0

    for _, row in valid_df.iterrows():
        stage1 = row.get("stage1_responses", {})
        expected = row.get("expected", "")
        benchmark = row.get("benchmark", "")

        if not isinstance(stage1, dict) or model not in stage1:
            continue

        response = stage1[model]
        is_correct = _evaluate_individual_response(response, expected, benchmark)

        total_n += 1
        if is_correct:
            total_correct += 1

        if benchmark in results_by_benchmark:
            results_by_benchmark[benchmark]["n"] += 1
            if is_correct:
                results_by_benchmark[benchmark]["correct"] += 1

    return {
        "overall": total_correct / total_n if total_n > 0 else None,
        "GSM8K": (
            results_by_benchmark["GSM8K"]["correct"] / results_by_benchmark["GSM8K"]["n"]
            if results_by_benchmark["GSM8K"]["n"] > 0 else None
        ),
        "TruthfulQA": (
            results_by_benchmark["TruthfulQA"]["correct"] / results_by_benchmark["TruthfulQA"]["n"]
            if results_by_benchmark["TruthfulQA"]["n"] > 0 else None
        ),
        "n_trials": total_n,
    }


def generate_comparison_report(
    prompt_df: pd.DataFrame,
    pilot_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate comparison report between prompt variant council and baseline.

    Args:
        prompt_df: Prompt experiment results
        pilot_df: Pilot study results (for baseline)
        output_path: Optional path to save report

    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PROMPT VARIANT COUNCIL EXPERIMENT - ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Experiment info
    lines.append("EXPERIMENT CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Base model: {BASE_MODEL}")
    lines.append(f"Prompt variants: {len(VARIANT_MODELS)}")
    for i, variant in enumerate(VARIANT_MODELS, 1):
        lines.append(f"  {i}. {variant}")
    lines.append("")

    # Get baseline accuracy
    baseline = get_baseline_model_accuracy(pilot_df, BASE_MODEL)

    lines.append("BASELINE: SINGLE MODEL ACCURACY (from pilot)")
    lines.append("-" * 40)
    if baseline["overall"] is not None:
        lines.append(f"Model: {BASE_MODEL}")
        lines.append(f"Overall: {baseline['overall']:.1%} ({baseline['n_trials']} trials)")
        if baseline["GSM8K"] is not None:
            lines.append(f"  GSM8K: {baseline['GSM8K']:.1%}")
        if baseline["TruthfulQA"] is not None:
            lines.append(f"  TruthfulQA: {baseline['TruthfulQA']:.1%}")
    else:
        lines.append("No baseline data available from pilot study.")
    lines.append("")

    # Prompt experiment results
    if prompt_df.empty:
        lines.append("PROMPT EXPERIMENT RESULTS")
        lines.append("-" * 40)
        lines.append("No results yet. Run: python -m experiments.run_prompt_experiment")
        lines.append("")
    else:
        # Overall accuracy
        valid_df = prompt_df[prompt_df["is_correct"].notna()]
        overall_accuracy = valid_df["is_correct"].mean() if len(valid_df) > 0 else 0

        lines.append("PROMPT COUNCIL RESULTS")
        lines.append("-" * 40)
        lines.append(f"Total trials: {len(valid_df)}")
        lines.append(f"Overall accuracy: {overall_accuracy:.1%}")
        lines.append("")

        # By structure
        struct_acc = compute_accuracy_by_structure(prompt_df)
        lines.append("Accuracy by Structure:")
        for _, row in struct_acc.iterrows():
            lines.append(f"  {row['structure']}: {row['accuracy']:.1%} ({row['n_correct']}/{row['n_trials']})")
        lines.append("")

        # By benchmark
        bench_acc = compute_accuracy_by_benchmark(prompt_df)
        lines.append("Accuracy by Benchmark:")
        for _, row in bench_acc.iterrows():
            lines.append(f"  {row['benchmark']}: {row['accuracy']:.1%} ({row['n_correct']}/{row['n_trials']})")
        lines.append("")

        # Comparison
        lines.append("COMPARISON: COUNCIL vs SINGLE MODEL")
        lines.append("-" * 40)
        if baseline["overall"] is not None:
            diff = overall_accuracy - baseline["overall"]
            direction = "+" if diff >= 0 else ""
            lines.append(f"Single model baseline: {baseline['overall']:.1%}")
            lines.append(f"Prompt variant council: {overall_accuracy:.1%}")
            lines.append(f"Difference: {direction}{diff:.1%}")
            lines.append("")

            if diff > 0.02:
                lines.append("FINDING: Prompt diversity IMPROVES over single model")
            elif diff < -0.02:
                lines.append("FINDING: Prompt diversity HURTS compared to single model")
            else:
                lines.append("FINDING: Prompt diversity has MINIMAL effect")

            # Best structure comparison
            if not struct_acc.empty:
                best_struct = struct_acc.iloc[0]
                best_diff = best_struct["accuracy"] - baseline["overall"]
                best_dir = "+" if best_diff >= 0 else ""
                lines.append("")
                lines.append(f"Best structure: {best_struct['structure']}")
                lines.append(f"  Accuracy: {best_struct['accuracy']:.1%} ({best_dir}{best_diff:.1%} vs baseline)")
        else:
            lines.append("Cannot compare - no baseline data available.")

        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report


def analyze_prompt_experiment(
    prompt_results_dir: str = "experiments/results_prompt_variants",
    pilot_results_dir: str = "experiments/results",
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main analysis function for prompt variant experiment.

    Args:
        prompt_results_dir: Directory with prompt experiment results
        pilot_results_dir: Directory with pilot study results
        output_dir: Optional directory to save analysis outputs

    Returns:
        DataFrame with prompt experiment results
    """
    # Load data
    prompt_df = load_prompt_experiment_results(prompt_results_dir)
    pilot_df = load_pilot_results(pilot_results_dir)

    # Set output directory
    if output_dir is None:
        output_dir = prompt_results_dir

    # Generate and save report
    report_path = str(Path(output_dir) / "analysis_report.txt")
    report = generate_comparison_report(prompt_df, pilot_df, report_path)
    print(report)

    return prompt_df


if __name__ == "__main__":
    df = analyze_prompt_experiment()
