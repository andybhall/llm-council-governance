"""Analysis script for council size experiment.

Analyzes how accuracy varies with council size to test the inverted-U
curve hypothesis (too few models = insufficient diversity, too many = noise).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from experiments.run_council_size import load_results


def load_results_as_dataframe(
    output_dir: str = "experiments/results_council_size",
    filename: str = "council_size_results.json",
) -> pd.DataFrame:
    """Load experiment results as a pandas DataFrame."""
    results = load_results(output_dir, filename)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def compute_accuracy_by_council_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each council size.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with council_size, n_trials, n_correct, accuracy
    """
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(
            columns=["council_size", "n_trials", "n_correct", "accuracy"]
        )

    valid_df = df[df["is_correct"].notna()].copy()

    summary = (
        valid_df.groupby("council_size")
        .agg(
            n_trials=("is_correct", "count"),
            n_correct=("is_correct", "sum"),
        )
        .reset_index()
    )

    summary["accuracy"] = summary["n_correct"] / summary["n_trials"]
    return summary.sort_values("council_size")


def compute_accuracy_by_size_and_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy for each council size and benchmark combination.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with council_size, benchmark, n_trials, n_correct, accuracy
    """
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(
            columns=["council_size", "benchmark", "n_trials", "n_correct", "accuracy"]
        )

    valid_df = df[df["is_correct"].notna()].copy()

    summary = (
        valid_df.groupby(["council_size", "benchmark"])
        .agg(
            n_trials=("is_correct", "count"),
            n_correct=("is_correct", "sum"),
        )
        .reset_index()
    )

    summary["accuracy"] = summary["n_correct"] / summary["n_trials"]
    return summary.sort_values(["council_size", "benchmark"])


def wilson_score_interval(
    k: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for binomial proportion."""
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


def compute_accuracy_with_ci(
    df: pd.DataFrame, confidence: float = 0.95
) -> pd.DataFrame:
    """
    Compute accuracy with confidence intervals for each council size.

    Args:
        df: DataFrame with experiment results
        confidence: Confidence level (default 0.95)

    Returns:
        DataFrame with council_size, accuracy, ci_lower, ci_upper
    """
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(
            columns=["council_size", "accuracy", "ci_lower", "ci_upper", "n_trials"]
        )

    valid_df = df[df["is_correct"].notna()].copy()

    results = []
    for size in sorted(valid_df["council_size"].unique()):
        size_df = valid_df[valid_df["council_size"] == size]
        n = len(size_df)
        k = int(size_df["is_correct"].sum())

        accuracy = k / n if n > 0 else 0
        ci_lower, ci_upper = wilson_score_interval(k, n, confidence)

        results.append(
            {
                "council_size": size,
                "accuracy": accuracy,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_trials": n,
            }
        )

    return pd.DataFrame(results)


def find_optimal_size(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Find the optimal council size based on accuracy.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with optimal_size, peak_accuracy, and analysis
    """
    acc_df = compute_accuracy_by_council_size(df)

    if acc_df.empty:
        return {"optimal_size": None, "peak_accuracy": None, "analysis": "No data"}

    # Find size with highest accuracy
    best_idx = acc_df["accuracy"].idxmax()
    best_row = acc_df.loc[best_idx]

    optimal_size = int(best_row["council_size"])
    peak_accuracy = best_row["accuracy"]

    # Check if it's truly an inverted-U (peak not at boundary)
    sizes = sorted(acc_df["council_size"].unique())
    is_inverted_u = optimal_size not in [min(sizes), max(sizes)]

    if is_inverted_u:
        analysis = (
            f"Inverted-U pattern detected: peak at size {optimal_size} "
            f"with {peak_accuracy:.1%} accuracy"
        )
    elif optimal_size == max(sizes):
        analysis = (
            f"Monotonically increasing: accuracy increases with council size. "
            f"Consider testing larger councils."
        )
    else:
        analysis = (
            f"Monotonically decreasing: accuracy decreases with council size. "
            f"Consider testing smaller councils."
        )

    return {
        "optimal_size": optimal_size,
        "peak_accuracy": peak_accuracy,
        "is_inverted_u": is_inverted_u,
        "analysis": analysis,
    }


def compute_marginal_benefit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute marginal benefit of adding each additional model.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with council_size, accuracy, marginal_benefit
    """
    acc_df = compute_accuracy_by_council_size(df)

    if acc_df.empty or len(acc_df) < 2:
        return pd.DataFrame(
            columns=["council_size", "accuracy", "marginal_benefit"]
        )

    acc_df = acc_df.sort_values("council_size")
    acc_df["marginal_benefit"] = acc_df["accuracy"].diff()

    return acc_df


def run_trend_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run statistical test for trend in accuracy across council sizes.

    Uses Spearman correlation to test for monotonic trend.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with correlation coefficient, p-value, and interpretation
    """
    if df.empty or "is_correct" not in df.columns:
        return {"error": "Insufficient data"}

    from scipy import stats

    valid_df = df[df["is_correct"].notna()].copy()

    # Compute accuracy by size
    acc_df = compute_accuracy_by_council_size(valid_df)

    if len(acc_df) < 3:
        return {"error": "Need at least 3 council sizes for trend test"}

    # Spearman correlation between size and accuracy
    correlation, p_value = stats.spearmanr(
        acc_df["council_size"], acc_df["accuracy"]
    )

    if p_value < 0.05:
        if correlation > 0:
            trend = "Significant positive trend (more models = higher accuracy)"
        else:
            trend = "Significant negative trend (more models = lower accuracy)"
    else:
        trend = "No significant linear trend (possible inverted-U pattern)"

    return {
        "spearman_correlation": correlation,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "trend": trend,
    }


def generate_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a full analysis report for council size experiment.

    Args:
        df: DataFrame with experiment results
        output_path: Optional path to save the report

    Returns:
        Report as a string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("COUNCIL SIZE EXPERIMENT - ANALYSIS REPORT")
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
    lines.append(f"Council sizes tested: {sorted(df['council_size'].unique())}")
    lines.append(f"Benchmarks: {df['benchmark'].unique().tolist()}")

    valid_df = df[df["is_correct"].notna()]
    if len(valid_df) > 0:
        lines.append(f"Valid trials: {len(valid_df)}")
        lines.append(f"Overall accuracy: {valid_df['is_correct'].mean():.1%}")
    lines.append("")

    # Accuracy by council size
    lines.append("ACCURACY BY COUNCIL SIZE")
    lines.append("-" * 40)
    acc_df = compute_accuracy_by_council_size(df)
    if not acc_df.empty:
        for _, row in acc_df.iterrows():
            lines.append(
                f"  Size {int(row['council_size'])}: {row['accuracy']:.1%} "
                f"({int(row['n_correct'])}/{int(row['n_trials'])})"
            )
    lines.append("")

    # Accuracy with confidence intervals
    lines.append("ACCURACY WITH 95% CONFIDENCE INTERVALS")
    lines.append("-" * 40)
    try:
        ci_df = compute_accuracy_with_ci(df)
        if not ci_df.empty:
            for _, row in ci_df.iterrows():
                lines.append(
                    f"  Size {int(row['council_size'])}: {row['accuracy']:.1%} "
                    f"[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]"
                )
    except ImportError:
        lines.append("  (scipy not available for CI computation)")
    lines.append("")

    # Accuracy by size and benchmark
    lines.append("ACCURACY BY SIZE AND BENCHMARK")
    lines.append("-" * 40)
    size_bench = compute_accuracy_by_size_and_benchmark(df)
    if not size_bench.empty:
        for benchmark in size_bench["benchmark"].unique():
            lines.append(f"  {benchmark}:")
            bench_df = size_bench[size_bench["benchmark"] == benchmark]
            for _, row in bench_df.iterrows():
                lines.append(
                    f"    Size {int(row['council_size'])}: {row['accuracy']:.1%}"
                )
    lines.append("")

    # Optimal size analysis
    lines.append("OPTIMAL COUNCIL SIZE ANALYSIS")
    lines.append("-" * 40)
    optimal = find_optimal_size(df)
    if optimal["optimal_size"] is not None:
        lines.append(f"  Optimal size: {optimal['optimal_size']}")
        lines.append(f"  Peak accuracy: {optimal['peak_accuracy']:.1%}")
        lines.append(f"  Inverted-U pattern: {optimal.get('is_inverted_u', False)}")
        lines.append(f"  Analysis: {optimal['analysis']}")
    lines.append("")

    # Marginal benefit
    lines.append("MARGINAL BENEFIT OF ADDITIONAL MODELS")
    lines.append("-" * 40)
    marginal = compute_marginal_benefit(df)
    if not marginal.empty:
        for _, row in marginal.iterrows():
            if pd.notna(row["marginal_benefit"]):
                lines.append(
                    f"  Size {int(row['council_size'])-1} → {int(row['council_size'])}: "
                    f"{row['marginal_benefit']:+.1%}"
                )
    lines.append("")

    # Trend test
    lines.append("TREND ANALYSIS")
    lines.append("-" * 40)
    try:
        trend = run_trend_test(df)
        if "error" not in trend:
            lines.append(f"  Spearman correlation: {trend['spearman_correlation']:.3f}")
            lines.append(f"  p-value: {trend['p_value']:.4f}")
            lines.append(f"  Interpretation: {trend['trend']}")
        else:
            lines.append(f"  {trend['error']}")
    except ImportError:
        lines.append("  (scipy not available for trend test)")
    lines.append("")

    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)

    return report


def generate_council_size_curve(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Generate accuracy vs council size curve visualization.

    Args:
        df: DataFrame with experiment results
        output_path: Path to save the chart image
    """
    import matplotlib.pyplot as plt

    ci_df = compute_accuracy_with_ci(df)
    if ci_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot accuracy with confidence interval band
    sizes = ci_df["council_size"].values
    accuracies = ci_df["accuracy"].values
    ci_lower = ci_df["ci_lower"].values
    ci_upper = ci_df["ci_upper"].values

    # Confidence interval band
    ax.fill_between(
        sizes, ci_lower, ci_upper,
        alpha=0.3, color="#3498db",
        label="95% CI"
    )

    # Main line
    ax.plot(
        sizes, accuracies,
        marker="o", markersize=10,
        linewidth=2, color="#2980b9",
        label="Accuracy"
    )

    # Add value labels
    for x, y in zip(sizes, accuracies):
        ax.annotate(
            f"{y:.1%}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )

    # Find and mark optimal
    optimal = find_optimal_size(df)
    if optimal["optimal_size"] is not None:
        optimal_idx = list(sizes).index(optimal["optimal_size"])
        ax.scatter(
            [optimal["optimal_size"]],
            [accuracies[optimal_idx]],
            s=200, c="green", marker="*", zorder=5,
            label=f"Optimal (size={optimal['optimal_size']})"
        )

    ax.set_xlabel("Council Size (number of models)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Accuracy vs Council Size\n(Testing Inverted-U Hypothesis)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xticks(sizes)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

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

    # Council size curve
    curve_path = str(output_path / "council_size_curve.png")
    try:
        generate_council_size_curve(df, curve_path)
        charts.append(curve_path)
        print(f"  Saved: {curve_path}")
    except ImportError:
        print("  (matplotlib not available for chart generation)")

    return charts


def analyze_council_size(
    output_dir: str = "experiments/results_council_size",
    report_path: Optional[str] = None,
    generate_charts_flag: bool = True,
) -> pd.DataFrame:
    """
    Analyze council size experiment results and generate report.

    Args:
        output_dir: Directory containing results
        report_path: Optional path to save report
        generate_charts_flag: Whether to generate chart images

    Returns:
        DataFrame with experiment results
    """
    df = load_results_as_dataframe(output_dir)

    if report_path is None:
        report_path = str(Path(output_dir) / "council_size_report.txt")

    report = generate_report(df, report_path)
    print(report)

    if generate_charts_flag and not df.empty:
        print("\nGenerating charts...")
        generate_charts(df, output_dir)

    return df


if __name__ == "__main__":
    analyze_council_size()
