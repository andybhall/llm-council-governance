"""Tests for prompt experiment analysis."""

import pytest
import json
from pathlib import Path

from experiments.analyze_prompt_experiment import (
    load_prompt_experiment_results,
    load_pilot_results,
    compute_accuracy_by_structure,
    compute_accuracy_by_benchmark,
    compute_accuracy_matrix,
    get_baseline_model_accuracy,
    generate_comparison_report,
)
from backend.prompt_variants import BASE_MODEL


class TestLoadResults:
    """Test result loading functions."""

    def test_load_prompt_results_missing_file(self, tmp_path):
        """Returns empty DataFrame if file doesn't exist."""
        df = load_prompt_experiment_results(str(tmp_path), "missing.json")
        assert df.empty

    def test_load_pilot_results_missing_file(self, tmp_path):
        """Returns empty DataFrame if file doesn't exist."""
        df = load_pilot_results(str(tmp_path), "missing.json")
        assert df.empty

    def test_load_prompt_results_valid(self, tmp_path):
        """Loads valid JSON results."""
        data = [{"benchmark": "GSM8K", "is_correct": True}]
        filepath = tmp_path / "results.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

        df = load_prompt_experiment_results(str(tmp_path), "results.json")
        assert len(df) == 1
        assert df.iloc[0]["benchmark"] == "GSM8K"


class TestComputeAccuracy:
    """Test accuracy computation functions."""

    def test_accuracy_by_structure(self):
        """Computes accuracy per structure."""
        import pandas as pd
        df = pd.DataFrame([
            {"structure": "A", "is_correct": True},
            {"structure": "A", "is_correct": True},
            {"structure": "A", "is_correct": False},
            {"structure": "B", "is_correct": True},
            {"structure": "B", "is_correct": False},
        ])

        result = compute_accuracy_by_structure(df)

        assert len(result) == 2
        a_row = result[result["structure"] == "A"].iloc[0]
        assert a_row["accuracy"] == pytest.approx(2/3)

    def test_accuracy_by_benchmark(self):
        """Computes accuracy per benchmark."""
        import pandas as pd
        df = pd.DataFrame([
            {"benchmark": "GSM8K", "is_correct": True},
            {"benchmark": "GSM8K", "is_correct": True},
            {"benchmark": "TruthfulQA", "is_correct": False},
        ])

        result = compute_accuracy_by_benchmark(df)

        gsm_row = result[result["benchmark"] == "GSM8K"].iloc[0]
        assert gsm_row["accuracy"] == 1.0

    def test_accuracy_matrix(self):
        """Computes structure x benchmark matrix."""
        import pandas as pd
        df = pd.DataFrame([
            {"structure": "A", "benchmark": "GSM8K", "is_correct": True},
            {"structure": "A", "benchmark": "GSM8K", "is_correct": True},
            {"structure": "A", "benchmark": "TruthfulQA", "is_correct": False},
            {"structure": "B", "benchmark": "GSM8K", "is_correct": False},
        ])

        result = compute_accuracy_matrix(df)

        assert result.loc["A", "GSM8K"] == 1.0
        assert result.loc["A", "TruthfulQA"] == 0.0

    def test_empty_dataframe(self):
        """Handles empty DataFrames."""
        import pandas as pd
        df = pd.DataFrame()

        assert compute_accuracy_by_structure(df).empty
        assert compute_accuracy_by_benchmark(df).empty


class TestBaselineAccuracy:
    """Test baseline extraction from pilot data."""

    def test_extracts_baseline_for_model(self):
        """Extracts accuracy for specified model from stage1_responses."""
        import pandas as pd
        df = pd.DataFrame([
            {
                "stage1_responses": {BASE_MODEL: "The answer is 4. FINAL ANSWER: 4"},
                "expected": "4",
                "benchmark": "GSM8K",
            },
            {
                "stage1_responses": {BASE_MODEL: "I think 5. FINAL ANSWER: 5"},
                "expected": "4",
                "benchmark": "GSM8K",
            },
        ])

        result = get_baseline_model_accuracy(df, BASE_MODEL)

        assert result["overall"] == 0.5
        assert result["n_trials"] == 2

    def test_handles_missing_model(self):
        """Returns None if model not in stage1_responses."""
        import pandas as pd
        df = pd.DataFrame([
            {
                "stage1_responses": {"other-model": "answer"},
                "expected": "4",
                "benchmark": "GSM8K",
            },
        ])

        result = get_baseline_model_accuracy(df, BASE_MODEL)

        assert result["overall"] is None

    def test_handles_empty_dataframe(self):
        """Returns None values for empty DataFrame."""
        import pandas as pd
        df = pd.DataFrame()

        result = get_baseline_model_accuracy(df, BASE_MODEL)

        assert result["overall"] is None


class TestComparisonReport:
    """Test report generation."""

    def test_generates_report(self):
        """Generates non-empty report."""
        import pandas as pd

        prompt_df = pd.DataFrame()
        pilot_df = pd.DataFrame()

        report = generate_comparison_report(prompt_df, pilot_df)

        assert "PROMPT VARIANT COUNCIL EXPERIMENT" in report
        assert BASE_MODEL in report

    def test_saves_report_to_file(self, tmp_path):
        """Saves report to specified path."""
        import pandas as pd

        prompt_df = pd.DataFrame()
        pilot_df = pd.DataFrame()
        output_path = str(tmp_path / "report.txt")

        generate_comparison_report(prompt_df, pilot_df, output_path)

        assert Path(output_path).exists()

    def test_shows_comparison_when_data_available(self):
        """Shows comparison when both datasets have data."""
        import pandas as pd

        prompt_df = pd.DataFrame([
            {"structure": "A", "benchmark": "GSM8K", "is_correct": True},
            {"structure": "A", "benchmark": "GSM8K", "is_correct": True},
        ])
        pilot_df = pd.DataFrame([
            {
                "stage1_responses": {BASE_MODEL: "FINAL ANSWER: 4"},
                "expected": "4",
                "benchmark": "GSM8K",
            },
        ])

        report = generate_comparison_report(prompt_df, pilot_df)

        assert "COMPARISON" in report
        assert "100.0%" in report  # prompt council accuracy
