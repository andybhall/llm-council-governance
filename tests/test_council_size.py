"""Tests for council size experiment functionality."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from experiments.analyze_council_size import (
    compute_accuracy_by_council_size,
    compute_accuracy_by_size_and_benchmark,
    compute_accuracy_with_ci,
    compute_marginal_benefit,
    find_optimal_size,
    generate_report,
    load_results_as_dataframe,
)
from experiments.run_council_size import (
    get_model_combinations,
    load_results,
    save_results,
)


class TestModelCombinations:
    """Tests for model combination generation."""

    def test_basic_combination(self):
        """Test getting model combinations for a given size."""
        pool = ["m1", "m2", "m3", "m4"]
        combos = get_model_combinations(pool, 3)
        assert len(combos) == 1
        assert len(combos[0]) == 3
        assert combos[0] == ["m1", "m2", "m3"]

    def test_full_pool(self):
        """Test using full pool."""
        pool = ["m1", "m2", "m3"]
        combos = get_model_combinations(pool, 3)
        assert combos == [["m1", "m2", "m3"]]

    def test_size_exceeds_pool_raises(self):
        """Test that size > pool raises error."""
        pool = ["m1", "m2"]
        with pytest.raises(ValueError):
            get_model_combinations(pool, 3)


class TestResultsSaveLoad:
    """Tests for saving and loading results."""

    def test_save_and_load_results(self):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                {"council_size": 2, "is_correct": True},
                {"council_size": 3, "is_correct": False},
            ]
            save_results(results, tmpdir)
            loaded = load_results(tmpdir)
            assert loaded == results

    def test_load_nonexistent_returns_empty(self):
        """Test loading from nonexistent file returns empty list."""
        loaded = load_results("/nonexistent/path")
        assert loaded == []


class TestAccuracyByCouncilSize:
    """Tests for accuracy computation by council size."""

    def test_basic_accuracy(self):
        """Test basic accuracy computation."""
        df = pd.DataFrame({
            "council_size": [2, 2, 3, 3, 3],
            "is_correct": [True, False, True, True, False],
        })
        acc = compute_accuracy_by_council_size(df)

        size_2 = acc[acc["council_size"] == 2].iloc[0]
        assert size_2["n_trials"] == 2
        assert size_2["n_correct"] == 1
        assert size_2["accuracy"] == 0.5

        size_3 = acc[acc["council_size"] == 3].iloc[0]
        assert size_3["n_trials"] == 3
        assert size_3["n_correct"] == 2
        assert abs(size_3["accuracy"] - 2/3) < 0.001

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()
        acc = compute_accuracy_by_council_size(df)
        assert acc.empty

    def test_filters_null_is_correct(self):
        """Test that null is_correct values are filtered."""
        df = pd.DataFrame({
            "council_size": [2, 2, 2],
            "is_correct": [True, None, False],
        })
        acc = compute_accuracy_by_council_size(df)
        assert acc.iloc[0]["n_trials"] == 2


class TestAccuracyBySizeAndBenchmark:
    """Tests for accuracy by size and benchmark."""

    def test_size_benchmark_combination(self):
        """Test accuracy grouped by size and benchmark."""
        df = pd.DataFrame({
            "council_size": [2, 2, 3, 3],
            "benchmark": ["GSM8K", "TruthfulQA", "GSM8K", "TruthfulQA"],
            "is_correct": [True, False, True, True],
        })
        acc = compute_accuracy_by_size_and_benchmark(df)

        assert len(acc) == 4
        assert "council_size" in acc.columns
        assert "benchmark" in acc.columns
        assert "accuracy" in acc.columns


class TestAccuracyWithCI:
    """Tests for accuracy with confidence intervals."""

    def test_ci_computation(self):
        """Test confidence interval computation."""
        df = pd.DataFrame({
            "council_size": [2] * 100,
            "is_correct": [True] * 80 + [False] * 20,
        })
        ci_df = compute_accuracy_with_ci(df)

        assert len(ci_df) == 1
        assert ci_df.iloc[0]["accuracy"] == 0.8
        assert ci_df.iloc[0]["ci_lower"] < 0.8
        assert ci_df.iloc[0]["ci_upper"] > 0.8

    def test_ci_bounds_valid(self):
        """Test CI bounds are between 0 and 1."""
        df = pd.DataFrame({
            "council_size": [2] * 10,
            "is_correct": [True] * 10,  # 100% accuracy
        })
        ci_df = compute_accuracy_with_ci(df)

        assert ci_df.iloc[0]["ci_lower"] >= 0
        assert ci_df.iloc[0]["ci_upper"] <= 1


class TestFindOptimalSize:
    """Tests for optimal size detection."""

    def test_finds_peak(self):
        """Test finding peak in inverted-U."""
        df = pd.DataFrame({
            "council_size": [2, 2, 3, 3, 3, 3, 4, 4],
            "is_correct": [
                True, False,  # size 2: 50%
                True, True, True, False,  # size 3: 75%
                True, False,  # size 4: 50%
            ],
        })
        optimal = find_optimal_size(df)

        assert optimal["optimal_size"] == 3
        assert optimal["peak_accuracy"] == 0.75
        assert optimal["is_inverted_u"] is True

    def test_monotonic_increasing(self):
        """Test detection of monotonically increasing pattern."""
        df = pd.DataFrame({
            "council_size": [2, 2, 3, 3, 4, 4],
            "is_correct": [
                False, False,  # size 2: 0%
                True, False,  # size 3: 50%
                True, True,  # size 4: 100%
            ],
        })
        optimal = find_optimal_size(df)

        assert optimal["optimal_size"] == 4
        assert optimal["is_inverted_u"] is False

    def test_empty_data(self):
        """Test with empty data."""
        df = pd.DataFrame()
        optimal = find_optimal_size(df)

        assert optimal["optimal_size"] is None


class TestMarginalBenefit:
    """Tests for marginal benefit computation."""

    def test_marginal_benefit_computation(self):
        """Test computing marginal benefit."""
        df = pd.DataFrame({
            "council_size": [2, 2, 3, 3, 4, 4],
            "is_correct": [
                True, False,  # size 2: 50%
                True, True,  # size 3: 100%
                True, False,  # size 4: 50%
            ],
        })
        marginal = compute_marginal_benefit(df)

        assert len(marginal) == 3
        # Size 2 -> 3: +50%
        size_3_row = marginal[marginal["council_size"] == 3].iloc[0]
        assert size_3_row["marginal_benefit"] == 0.5
        # Size 3 -> 4: -50%
        size_4_row = marginal[marginal["council_size"] == 4].iloc[0]
        assert size_4_row["marginal_benefit"] == -0.5


class TestGenerateReport:
    """Tests for report generation."""

    def test_report_generation(self):
        """Test basic report generation."""
        df = pd.DataFrame({
            "council_size": [2, 3, 4],
            "benchmark": ["GSM8K", "GSM8K", "GSM8K"],
            "is_correct": [True, True, False],
        })
        report = generate_report(df)

        assert "COUNCIL SIZE EXPERIMENT" in report
        assert "ACCURACY BY COUNCIL SIZE" in report
        assert "OPTIMAL COUNCIL SIZE" in report

    def test_empty_report(self):
        """Test report with empty data."""
        df = pd.DataFrame()
        report = generate_report(df)

        assert "No results to analyze" in report

    def test_report_saves_to_file(self):
        """Test saving report to file."""
        df = pd.DataFrame({
            "council_size": [2, 3],
            "benchmark": ["GSM8K", "GSM8K"],
            "is_correct": [True, False],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            generate_report(df, str(report_path))
            assert report_path.exists()
            content = report_path.read_text()
            assert "COUNCIL SIZE EXPERIMENT" in content


class TestLoadResultsAsDataframe:
    """Tests for loading results as DataFrame."""

    def test_load_as_dataframe(self):
        """Test loading results as DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                {"council_size": 2, "is_correct": True},
                {"council_size": 3, "is_correct": False},
            ]
            save_results(results, tmpdir)
            df = load_results_as_dataframe(tmpdir)

            assert len(df) == 2
            assert list(df["council_size"]) == [2, 3]

    def test_empty_returns_empty_df(self):
        """Test empty results return empty DataFrame."""
        df = load_results_as_dataframe("/nonexistent")
        assert df.empty


class TestConfigIntegration:
    """Tests for config integration."""

    def test_extended_models_in_config(self):
        """Test that EXTENDED_MODELS is available in config."""
        from backend.config import EXTENDED_MODELS

        assert isinstance(EXTENDED_MODELS, list)
        assert len(EXTENDED_MODELS) >= 4  # Should have at least 4 models

    def test_council_sizes_in_config(self):
        """Test that COUNCIL_SIZES is available in config."""
        from backend.config import COUNCIL_SIZES

        assert isinstance(COUNCIL_SIZES, list)
        assert 2 in COUNCIL_SIZES
        assert len(COUNCIL_SIZES) >= 3
