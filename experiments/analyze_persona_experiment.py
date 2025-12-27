"""Analysis script for persona variant experiment.

Compares persona variant council performance against single model baseline
and prompt variant results.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from backend.persona_variants import BASE_MODEL, PERSONA_MODELS


def load_persona_experiment_results(
    results_dir: str = "experiments/results_persona_variants",
    filename: str = "persona_experiment_results.json",
) -> pd.DataFrame:
    """Load persona experiment results as DataFrame."""
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


def load_prompt_experiment_results(
    results_dir: str = "experiments/results_prompt_variants",
    filename: str = "prompt_experiment_results.json",
) -> pd.DataFrame:
    """Load prompt variant experiment results as DataFrame."""
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


def get_prompt_variant_accuracy(prompt_df: pd.DataFrame) -> Optional[float]:
    """Get overall accuracy from prompt variant experiment."""
    if prompt_df.empty or "is_correct" not in prompt_df.columns:
        return None

    valid_df = prompt_df[prompt_df["is_correct"].notna()]
    if len(valid_df) == 0:
        return None

    return valid_df["is_correct"].mean()


def generate_comparison_report(
    persona_df: pd.DataFrame,
    pilot_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate comparison report between persona council, prompt council, and baseline.

    Args:
        persona_df: Persona experiment results
        pilot_df: Pilot study results (for baseline)
        prompt_df: Prompt variant experiment results
        output_path: Optional path to save report

    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PERSONA VARIANT COUNCIL EXPERIMENT - ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Experiment info
    lines.append("EXPERIMENT CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Base model: {BASE_MODEL}")
    lines.append(f"Personas: {len(PERSONA_MODELS)}")
    for i, persona in enumerate(PERSONA_MODELS, 1):
        lines.append(f"  {i}. {persona}")
    lines.append("")

    # Get baseline accuracy
    baseline = get_baseline_model_accuracy(pilot_df, BASE_MODEL)
    prompt_accuracy = get_prompt_variant_accuracy(prompt_df)

    lines.append("BASELINES")
    lines.append("-" * 40)
    if baseline["overall"] is not None:
        lines.append(f"Single model ({BASE_MODEL}): {baseline['overall']:.1%}")
    else:
        lines.append("Single model: No data available")

    if prompt_accuracy is not None:
        lines.append(f"Prompt variant council: {prompt_accuracy:.1%}")
    else:
        lines.append("Prompt variant council: No data available")
    lines.append("")

    # Persona experiment results
    if persona_df.empty:
        lines.append("PERSONA COUNCIL RESULTS")
        lines.append("-" * 40)
        lines.append("No results yet. Run: python -m experiments.run_persona_experiment")
        lines.append("")
    else:
        # Overall accuracy
        valid_df = persona_df[persona_df["is_correct"].notna()]
        overall_accuracy = valid_df["is_correct"].mean() if len(valid_df) > 0 else 0

        lines.append("PERSONA COUNCIL RESULTS")
        lines.append("-" * 40)
        lines.append(f"Total trials: {len(valid_df)}")
        lines.append(f"Overall accuracy: {overall_accuracy:.1%}")
        lines.append("")

        # By structure
        struct_acc = compute_accuracy_by_structure(persona_df)
        lines.append("Accuracy by Structure:")
        for _, row in struct_acc.iterrows():
            lines.append(f"  {row['structure']}: {row['accuracy']:.1%} ({row['n_correct']}/{row['n_trials']})")
        lines.append("")

        # By benchmark
        bench_acc = compute_accuracy_by_benchmark(persona_df)
        lines.append("Accuracy by Benchmark:")
        for _, row in bench_acc.iterrows():
            lines.append(f"  {row['benchmark']}: {row['accuracy']:.1%} ({row['n_correct']}/{row['n_trials']})")
        lines.append("")

        # Comparison
        lines.append("COMPARISON")
        lines.append("-" * 40)

        if baseline["overall"] is not None:
            diff_baseline = overall_accuracy - baseline["overall"]
            dir_baseline = "+" if diff_baseline >= 0 else ""
            lines.append(f"Single model baseline: {baseline['overall']:.1%}")
            lines.append(f"Persona council:       {overall_accuracy:.1%} ({dir_baseline}{diff_baseline:.1%})")

        if prompt_accuracy is not None:
            diff_prompt = overall_accuracy - prompt_accuracy
            dir_prompt = "+" if diff_prompt >= 0 else ""
            lines.append(f"Prompt variant council: {prompt_accuracy:.1%}")
            lines.append(f"Persona council:        {overall_accuracy:.1%} ({dir_prompt}{diff_prompt:.1%})")

        lines.append("")

        # Finding
        if baseline["overall"] is not None:
            diff = overall_accuracy - baseline["overall"]
            if diff > 0.02:
                lines.append("FINDING: Persona diversity IMPROVES over single model")
            elif diff < -0.02:
                lines.append("FINDING: Persona diversity HURTS compared to single model")
            else:
                lines.append("FINDING: Persona diversity has MINIMAL effect")

            if prompt_accuracy is not None:
                persona_vs_prompt = overall_accuracy - prompt_accuracy
                if persona_vs_prompt > 0.02:
                    lines.append("         Personas work BETTER than prompt variants")
                elif persona_vs_prompt < -0.02:
                    lines.append("         Personas work WORSE than prompt variants")
                else:
                    lines.append("         Personas perform SIMILARLY to prompt variants")

        # Best structure comparison
        if not struct_acc.empty and baseline["overall"] is not None:
            best_struct = struct_acc.iloc[0]
            best_diff = best_struct["accuracy"] - baseline["overall"]
            best_dir = "+" if best_diff >= 0 else ""
            lines.append("")
            lines.append(f"Best structure: {best_struct['structure']}")
            lines.append(f"  Accuracy: {best_struct['accuracy']:.1%} ({best_dir}{best_diff:.1%} vs baseline)")

        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report


def analyze_persona_experiment(
    persona_results_dir: str = "experiments/results_persona_variants",
    pilot_results_dir: str = "experiments/results",
    prompt_results_dir: str = "experiments/results_prompt_variants",
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main analysis function for persona variant experiment.

    Args:
        persona_results_dir: Directory with persona experiment results
        pilot_results_dir: Directory with pilot study results
        prompt_results_dir: Directory with prompt variant results
        output_dir: Optional directory to save analysis outputs

    Returns:
        DataFrame with persona experiment results
    """
    # Load data
    persona_df = load_persona_experiment_results(persona_results_dir)
    pilot_df = load_pilot_results(pilot_results_dir)
    prompt_df = load_prompt_experiment_results(prompt_results_dir)

    # Set output directory
    if output_dir is None:
        output_dir = persona_results_dir

    # Generate and save report
    report_path = str(Path(output_dir) / "analysis_report.txt")
    report = generate_comparison_report(persona_df, pilot_df, prompt_df, report_path)
    print(report)

    return persona_df


if __name__ == "__main__":
    df = analyze_persona_experiment()
