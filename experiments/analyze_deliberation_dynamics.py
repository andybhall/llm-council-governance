"""Analyze deliberation dynamics: mind changes, influence patterns, and agreement shifts."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.governance.utils import extract_final_answer, normalize_answer


def extract_stage_answers(
    stage1_responses: Dict[str, str],
    stage2_data: Dict[str, Any],
) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """
    Extract and normalize answers from stage1 and stage2 (deliberation).

    Args:
        stage1_responses: Raw stage1 responses by model
        stage2_data: Stage2 data containing deliberation_responses

    Returns:
        Tuple of (stage1_answers, stage2_answers) dicts mapping model to normalized answer
    """
    stage1_answers = {}
    stage2_answers = {}

    delib_responses = stage2_data.get("deliberation_responses", {})

    for model in stage1_responses.keys():
        # Stage 1 answer
        raw1 = extract_final_answer(stage1_responses.get(model, ""))
        stage1_answers[model] = normalize_answer(raw1) if raw1 else None

        # Stage 2 (deliberation) answer
        if model in delib_responses:
            raw2 = extract_final_answer(delib_responses.get(model, ""))
            stage2_answers[model] = normalize_answer(raw2) if raw2 else None
        else:
            stage2_answers[model] = None

    return stage1_answers, stage2_answers


def compute_pairwise_agreement(answers: Dict[str, Optional[str]]) -> float:
    """
    Compute pairwise agreement fraction among answers.

    Args:
        answers: Dict mapping model to normalized answer

    Returns:
        Fraction of pairs that agree (0-1)
    """
    valid = [(m, a) for m, a in answers.items() if a is not None]
    if len(valid) < 2:
        return 0.0

    n_pairs = 0
    n_agree = 0
    for i, (_, a1) in enumerate(valid):
        for j, (_, a2) in enumerate(valid):
            if i < j:
                n_pairs += 1
                if a1 == a2:
                    n_agree += 1

    return n_agree / n_pairs if n_pairs > 0 else 0.0


def analyze_single_trial(
    stage1_answers: Dict[str, Optional[str]],
    stage2_answers: Dict[str, Optional[str]],
    expected: Optional[str],
) -> Dict[str, Any]:
    """
    Analyze a single trial's deliberation dynamics.

    Args:
        stage1_answers: Normalized stage1 answers by model
        stage2_answers: Normalized stage2 answers by model
        expected: Expected answer (ground truth)

    Returns:
        Dictionary with per-model metrics and aggregates
    """
    # Normalize expected answer
    expected_norm = normalize_answer(expected) if expected else None

    model_metrics = {}
    total_changes = 0
    fixed = 0
    broke = 0

    for model in stage1_answers.keys():
        a1 = stage1_answers.get(model)
        a2 = stage2_answers.get(model)

        if a1 is None or a2 is None:
            continue

        changed = a1 != a2
        was_correct = a1 == expected_norm if expected_norm else None
        now_correct = a2 == expected_norm if expected_norm else None

        # Check if changed to someone else's stage1 answer
        changed_to_someone_else = False
        influence_sources = []
        if changed:
            total_changes += 1
            for other_model, other_a1 in stage1_answers.items():
                if other_model != model and other_a1 == a2:
                    changed_to_someone_else = True
                    influence_sources.append(other_model)

            # Track fixed/broke
            if was_correct is False and now_correct is True:
                fixed += 1
            elif was_correct is True and now_correct is False:
                broke += 1

        model_metrics[model] = {
            "stage1_answer": a1,
            "stage2_answer": a2,
            "changed_mind": changed,
            "changed_to_someone_else": changed_to_someone_else,
            "influence_sources": influence_sources,
            "was_correct": was_correct,
            "now_correct": now_correct,
            "fixed": was_correct is False and now_correct is True,
            "broke": was_correct is True and now_correct is False,
        }

    agreement_pre = compute_pairwise_agreement(stage1_answers)
    agreement_post = compute_pairwise_agreement(stage2_answers)

    return {
        "model_metrics": model_metrics,
        "agreement_pre": agreement_pre,
        "agreement_post": agreement_post,
        "agreement_change": agreement_post - agreement_pre,
        "total_changes": total_changes,
        "fixed": fixed,
        "broke": broke,
        "net_benefit": fixed - broke,
    }


def compute_influence_matrix(
    trial_analyses: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute influence matrix from multiple trial analyses.

    influence[source, target] = how many times source influenced target to change

    Args:
        trial_analyses: List of analyze_single_trial outputs

    Returns:
        DataFrame with influence matrix (source x target)
    """
    influence_counts = defaultdict(lambda: defaultdict(float))
    all_models = set()

    for trial in trial_analyses:
        for target_model, metrics in trial.get("model_metrics", {}).items():
            all_models.add(target_model)
            if metrics.get("changed_to_someone_else", False):
                sources = metrics.get("influence_sources", [])
                if sources:
                    # Split credit evenly among sources
                    credit = 1.0 / len(sources)
                    for source_model in sources:
                        influence_counts[source_model][target_model] += credit
                        all_models.add(source_model)

    # Convert to DataFrame
    models = sorted(all_models)
    matrix = pd.DataFrame(0.0, index=models, columns=models)
    for source in influence_counts:
        for target, count in influence_counts[source].items():
            matrix.loc[source, target] = count

    return matrix


def analyze_deliberation_dynamics(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze deliberation dynamics across all trials.

    Args:
        df: DataFrame with experiment results
        output_dir: Optional directory to save CSV/JSONL outputs

    Returns:
        Dictionary with aggregate analysis results
    """
    if df.empty:
        return {}

    # Filter to deliberation structures only
    delib_df = df[df["structure"].str.contains("Deliberate", na=False)].copy()
    if delib_df.empty:
        return {}

    trial_analyses = []
    mind_change_rows = []
    question_level_data = []

    for idx, row in delib_df.iterrows():
        stage1 = row.get("stage1_responses", {})
        stage2 = row.get("stage2_data", {})
        expected = row.get("expected", "")
        benchmark = row.get("benchmark", "")
        question_id = row.get("question_id", str(idx))
        structure = row.get("structure", "")
        replication = row.get("replication", 0)

        if not isinstance(stage1, dict) or not isinstance(stage2, dict):
            continue
        if "deliberation_responses" not in stage2:
            continue

        # Extract and analyze
        stage1_ans, stage2_ans = extract_stage_answers(stage1, stage2)
        trial_result = analyze_single_trial(stage1_ans, stage2_ans, expected)
        trial_analyses.append(trial_result)

        # Build per-model rows for CSV
        for model, metrics in trial_result.get("model_metrics", {}).items():
            mind_change_rows.append({
                "benchmark": benchmark,
                "question_id": question_id,
                "structure": structure,
                "replication": replication,
                "model": model,
                "stage1_answer": metrics.get("stage1_answer"),
                "stage2_answer": metrics.get("stage2_answer"),
                "changed_mind": metrics.get("changed_mind"),
                "changed_to_someone_else": metrics.get("changed_to_someone_else"),
                "was_correct": metrics.get("was_correct"),
                "now_correct": metrics.get("now_correct"),
                "fixed": metrics.get("fixed"),
                "broke": metrics.get("broke"),
            })

        # Build question-level JSONL entry
        question_level_data.append({
            "benchmark": benchmark,
            "question_id": question_id,
            "structure": structure,
            "replication": replication,
            "agreement_pre": trial_result["agreement_pre"],
            "agreement_post": trial_result["agreement_post"],
            "total_changes": trial_result["total_changes"],
            "fixed": trial_result["fixed"],
            "broke": trial_result["broke"],
            "model_metrics": trial_result["model_metrics"],
        })

    # Compute influence matrix
    influence_matrix = compute_influence_matrix(trial_analyses)

    # Aggregate statistics
    total_changes = sum(t["total_changes"] for t in trial_analyses)
    total_fixed = sum(t["fixed"] for t in trial_analyses)
    total_broke = sum(t["broke"] for t in trial_analyses)
    avg_agreement_pre = (
        sum(t["agreement_pre"] for t in trial_analyses) / len(trial_analyses)
        if trial_analyses else 0.0
    )
    avg_agreement_post = (
        sum(t["agreement_post"] for t in trial_analyses) / len(trial_analyses)
        if trial_analyses else 0.0
    )

    # Find most influential and most influenced
    influence_given = influence_matrix.sum(axis=1).sort_values(ascending=False)
    influence_received = influence_matrix.sum(axis=0).sort_values(ascending=False)

    result = {
        "n_trials": len(trial_analyses),
        "total_mind_changes": total_changes,
        "total_fixed": total_fixed,
        "total_broke": total_broke,
        "net_benefit": total_fixed - total_broke,
        "avg_agreement_pre": avg_agreement_pre,
        "avg_agreement_post": avg_agreement_post,
        "agreement_increase": avg_agreement_post - avg_agreement_pre,
        "most_influential": influence_given.head(3).to_dict(),
        "most_influenced": influence_received.head(3).to_dict(),
        "influence_matrix": influence_matrix,
        "mind_change_rows": mind_change_rows,
        "question_level_data": question_level_data,
    }

    # Save outputs if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save mind change CSV
        mind_change_df = pd.DataFrame(mind_change_rows)
        mind_change_df.to_csv(output_path / "analysis_mind_change.csv", index=False)

        # Save influence matrix CSV
        influence_matrix.to_csv(output_path / "analysis_influence_matrix.csv")

        # Save question-level JSONL
        with open(output_path / "analysis_question_level_changes.jsonl", "w") as f:
            for entry in question_level_data:
                # Convert model_metrics to serializable format
                entry_copy = entry.copy()
                entry_copy["model_metrics"] = {
                    m: {k: v for k, v in metrics.items() if k != "influence_sources"}
                    for m, metrics in entry_copy.get("model_metrics", {}).items()
                }
                f.write(json.dumps(entry_copy, default=str) + "\n")

    return result


def generate_dynamics_summary(analysis: Dict[str, Any]) -> str:
    """
    Generate a text summary of deliberation dynamics.

    Args:
        analysis: Output from analyze_deliberation_dynamics()

    Returns:
        Formatted text summary
    """
    if not analysis:
        return "No deliberation data available for analysis.\n"

    lines = [
        "DELIBERATION DYNAMICS ANALYSIS",
        "=" * 50,
        "",
        f"Trials analyzed: {analysis.get('n_trials', 0)}",
        "",
        "Agreement Changes:",
        f"  Pre-deliberation:  {analysis.get('avg_agreement_pre', 0):.1%}",
        f"  Post-deliberation: {analysis.get('avg_agreement_post', 0):.1%}",
        f"  Change:            {analysis.get('agreement_increase', 0):+.1%}",
        "",
        "Mind Changes:",
        f"  Total changes: {analysis.get('total_mind_changes', 0)}",
        f"  Fixed (wrong→right): {analysis.get('total_fixed', 0)}",
        f"  Broke (right→wrong): {analysis.get('total_broke', 0)}",
        f"  Net benefit: {analysis.get('net_benefit', 0):+d}",
        "",
        "Most Influential Models (influenced others to change):",
    ]

    for model, score in analysis.get("most_influential", {}).items():
        lines.append(f"  {model}: {score:.1f}")

    lines.append("")
    lines.append("Most Influenced Models (changed to follow others):")

    for model, score in analysis.get("most_influenced", {}).items():
        lines.append(f"  {model}: {score:.1f}")

    lines.append("")
    return "\n".join(lines)
