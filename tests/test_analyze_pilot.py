"""Tests for pilot study analysis script."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from experiments.analyze_pilot import (
    analyze_pilot,
    compute_accuracy_by_benchmark,
    compute_accuracy_by_structure,
    compute_accuracy_matrix,
    compute_accuracy_with_ci,
    compute_timing_summary,
    generate_report,
    load_results_as_dataframe,
    run_chi_square_test,
    run_pairwise_tests,
    wilson_score_interval,
)
from experiments.run_pilot import save_results


def create_sample_results():
    """Create sample experiment results for testing."""
    return [
        {
            "benchmark": "GSM8K",
            "question_id": "q1",
            "structure": "A",
            "replication": 0,
            "is_correct": True,
            "predicted": "42",
            "expected": "42",
            "elapsed_time": 1.5,
        },
        {
            "benchmark": "GSM8K",
            "question_id": "q1",
            "structure": "B",
            "replication": 0,
            "is_correct": False,
            "predicted": "41",
            "expected": "42",
            "elapsed_time": 2.0,
        },
        {
            "benchmark": "GSM8K",
            "question_id": "q2",
            "structure": "A",
            "replication": 0,
            "is_correct": True,
            "predicted": "10",
            "expected": "10",
            "elapsed_time": 1.2,
        },
        {
            "benchmark": "GSM8K",
            "question_id": "q2",
            "structure": "B",
            "replication": 0,
            "is_correct": True,
            "predicted": "10",
            "expected": "10",
            "elapsed_time": 1.8,
        },
        {
            "benchmark": "TruthfulQA",
            "question_id": "t1",
            "structure": "A",
            "replication": 0,
            "is_correct": False,
            "predicted": "B",
            "expected": "A",
            "elapsed_time": 1.0,
        },
        {
            "benchmark": "TruthfulQA",
            "question_id": "t1",
            "structure": "B",
            "replication": 0,
            "is_correct": True,
            "predicted": "A",
            "expected": "A",
            "elapsed_time": 1.5,
        },
    ]


class TestLoadResultsAsDataframe:
    """Tests for load_results_as_dataframe."""

    def test_loads_results(self):
        """Test loading results as DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = create_sample_results()
            save_results(results, tmpdir)

            df = load_results_as_dataframe(tmpdir)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 6

    def test_empty_results(self):
        """Test loading empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = load_results_as_dataframe(tmpdir)

            assert isinstance(df, pd.DataFrame)
            assert df.empty


class TestComputeAccuracyByStructure:
    """Tests for compute_accuracy_by_structure."""

    def test_computes_accuracy(self):
        """Test accuracy computation by structure."""
        df = pd.DataFrame(create_sample_results())

        result = compute_accuracy_by_structure(df)

        assert "structure" in result.columns
        assert "accuracy" in result.columns
        assert len(result) == 2  # Two structures: A and B

    def test_correct_accuracy_values(self):
        """Test that accuracy values are correct."""
        df = pd.DataFrame(create_sample_results())

        result = compute_accuracy_by_structure(df)
        result = result.set_index("structure")

        # Structure A: 2 correct out of 3 (q1, q2 correct; t1 wrong)
        assert result.loc["A", "accuracy"] == pytest.approx(2 / 3)
        # Structure B: 2 correct out of 3 (q1 wrong; q2, t1 correct)
        assert result.loc["B", "accuracy"] == pytest.approx(2 / 3)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        result = compute_accuracy_by_structure(df)

        assert result.empty


class TestComputeAccuracyByBenchmark:
    """Tests for compute_accuracy_by_benchmark."""

    def test_computes_accuracy(self):
        """Test accuracy computation by benchmark."""
        df = pd.DataFrame(create_sample_results())

        result = compute_accuracy_by_benchmark(df)

        assert "benchmark" in result.columns
        assert len(result) == 2  # GSM8K and TruthfulQA

    def test_correct_accuracy_values(self):
        """Test that accuracy values are correct."""
        df = pd.DataFrame(create_sample_results())

        result = compute_accuracy_by_benchmark(df)
        result = result.set_index("benchmark")

        # GSM8K: 3 correct out of 4
        assert result.loc["GSM8K", "accuracy"] == pytest.approx(3 / 4)
        # TruthfulQA: 1 correct out of 2
        assert result.loc["TruthfulQA", "accuracy"] == pytest.approx(1 / 2)


class TestComputeAccuracyMatrix:
    """Tests for compute_accuracy_matrix."""

    def test_returns_pivot_table(self):
        """Test that result is a pivot table."""
        df = pd.DataFrame(create_sample_results())

        result = compute_accuracy_matrix(df)

        assert isinstance(result, pd.DataFrame)
        assert "GSM8K" in result.columns
        assert "TruthfulQA" in result.columns
        assert "A" in result.index
        assert "B" in result.index

    def test_correct_matrix_values(self):
        """Test that matrix values are correct."""
        df = pd.DataFrame(create_sample_results())

        result = compute_accuracy_matrix(df)

        # Structure A, GSM8K: 2/2 = 1.0
        assert result.loc["A", "GSM8K"] == pytest.approx(1.0)
        # Structure B, GSM8K: 1/2 = 0.5
        assert result.loc["B", "GSM8K"] == pytest.approx(0.5)
        # Structure A, TruthfulQA: 0/1 = 0.0
        assert result.loc["A", "TruthfulQA"] == pytest.approx(0.0)
        # Structure B, TruthfulQA: 1/1 = 1.0
        assert result.loc["B", "TruthfulQA"] == pytest.approx(1.0)


class TestWilsonScoreInterval:
    """Tests for wilson_score_interval."""

    def test_zero_trials(self):
        """Test with zero trials."""
        lower, upper = wilson_score_interval(0, 0)
        assert lower == 0.0
        assert upper == 0.0

    def test_all_successes(self):
        """Test with all successes."""
        lower, upper = wilson_score_interval(10, 10)
        assert lower > 0.7
        assert upper == pytest.approx(1.0)

    def test_no_successes(self):
        """Test with no successes."""
        lower, upper = wilson_score_interval(0, 10)
        assert lower == 0.0
        assert upper < 0.3

    def test_half_successes(self):
        """Test with 50% success rate."""
        lower, upper = wilson_score_interval(50, 100)
        assert lower < 0.5
        assert upper > 0.5
        # Should be roughly symmetric around 0.5
        assert abs((lower + upper) / 2 - 0.5) < 0.1

    def test_bounds_in_range(self):
        """Test that bounds are always in [0, 1]."""
        for k in range(11):
            lower, upper = wilson_score_interval(k, 10)
            assert 0 <= lower <= 1
            assert 0 <= upper <= 1
            assert lower <= upper


class TestComputeAccuracyWithCI:
    """Tests for compute_accuracy_with_ci."""

    def test_returns_confidence_intervals(self):
        """Test that CI columns are present."""
        df = pd.DataFrame(create_sample_results())

        result = compute_accuracy_with_ci(df)

        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns
        assert "accuracy" in result.columns

    def test_ci_bounds_valid(self):
        """Test that CI bounds are valid."""
        df = pd.DataFrame(create_sample_results())

        result = compute_accuracy_with_ci(df)

        for _, row in result.iterrows():
            assert row["ci_lower"] <= row["accuracy"]
            assert row["accuracy"] <= row["ci_upper"]
            assert 0 <= row["ci_lower"]
            assert row["ci_upper"] <= 1


class TestRunChiSquareTest:
    """Tests for run_chi_square_test."""

    def test_returns_result(self):
        """Test that chi-square test returns result."""
        df = pd.DataFrame(create_sample_results())

        result = run_chi_square_test(df)

        assert result is not None
        assert "p_value" in result
        assert "interpretation" in result

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        result = run_chi_square_test(df)

        assert result is None

    def test_single_structure(self):
        """Test with single structure."""
        results = [r for r in create_sample_results() if r["structure"] == "A"]
        df = pd.DataFrame(results)

        result = run_chi_square_test(df)

        assert result is None  # Need at least 2 structures


class TestRunPairwiseTests:
    """Tests for run_pairwise_tests."""

    def test_returns_pairwise_results(self):
        """Test that pairwise tests return results."""
        df = pd.DataFrame(create_sample_results())

        results = run_pairwise_tests(df)

        assert len(results) == 1  # One pair: A vs B
        assert "structure_1" in results[0]
        assert "structure_2" in results[0]
        assert "p_value" in results[0]

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        results = run_pairwise_tests(df)

        assert results == []

    def test_bonferroni_correction(self):
        """Test that Bonferroni correction is applied."""
        # Create data with 3 structures for 3 pairwise comparisons
        results = create_sample_results()
        for r in results[:2]:
            new_r = r.copy()
            new_r["structure"] = "C"
            results.append(new_r)
        df = pd.DataFrame(results)

        pairwise = run_pairwise_tests(df)

        # Should have 3 comparisons: A-B, A-C, B-C
        assert len(pairwise) == 3

        # Corrected p-value should be 3x raw p-value (capped at 1.0)
        for result in pairwise:
            expected_corrected = min(result["p_value"] * 3, 1.0)
            assert result["p_value_corrected"] == pytest.approx(expected_corrected)


class TestComputeTimingSummary:
    """Tests for compute_timing_summary."""

    def test_computes_timing(self):
        """Test timing computation."""
        df = pd.DataFrame(create_sample_results())

        result = compute_timing_summary(df)

        assert "mean_time" in result.columns
        assert "std_time" in result.columns
        assert len(result) == 2  # Two structures

    def test_correct_mean_values(self):
        """Test that mean values are correct."""
        df = pd.DataFrame(create_sample_results())

        result = compute_timing_summary(df)
        result = result.set_index("structure")

        # Structure A: (1.5 + 1.2 + 1.0) / 3 = 1.233...
        assert result.loc["A", "mean_time"] == pytest.approx(1.233, rel=0.01)
        # Structure B: (2.0 + 1.8 + 1.5) / 3 = 1.767...
        assert result.loc["B", "mean_time"] == pytest.approx(1.767, rel=0.01)

    def test_no_elapsed_time_column(self):
        """Test with no elapsed_time column."""
        results = create_sample_results()
        for r in results:
            del r["elapsed_time"]
        df = pd.DataFrame(results)

        result = compute_timing_summary(df)

        assert result.empty


class TestGenerateReport:
    """Tests for generate_report."""

    def test_generates_report(self):
        """Test report generation."""
        df = pd.DataFrame(create_sample_results())

        report = generate_report(df)

        assert "ANALYSIS REPORT" in report
        assert "ACCURACY BY STRUCTURE" in report
        assert "ACCURACY BY BENCHMARK" in report

    def test_report_contains_structures(self):
        """Test that report contains structure names."""
        df = pd.DataFrame(create_sample_results())

        report = generate_report(df)

        assert "A:" in report or "Structure A" in report
        assert "B:" in report or "Structure B" in report

    def test_report_contains_benchmarks(self):
        """Test that report contains benchmark names."""
        df = pd.DataFrame(create_sample_results())

        report = generate_report(df)

        assert "GSM8K" in report
        assert "TruthfulQA" in report

    def test_saves_to_file(self):
        """Test that report is saved to file."""
        df = pd.DataFrame(create_sample_results())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            generate_report(df, str(output_path))

            assert output_path.exists()
            content = output_path.read_text()
            assert "ANALYSIS REPORT" in content

    def test_empty_dataframe(self):
        """Test report with empty DataFrame."""
        df = pd.DataFrame()

        report = generate_report(df)

        assert "No results to analyze" in report


class TestAnalyzePilot:
    """Tests for analyze_pilot main function."""

    def test_returns_dataframe(self):
        """Test that analyze_pilot returns DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = create_sample_results()
            save_results(results, tmpdir)

            df = analyze_pilot(tmpdir)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 6

    def test_creates_report_file(self):
        """Test that analyze_pilot creates report file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = create_sample_results()
            save_results(results, tmpdir)

            analyze_pilot(tmpdir)

            report_path = Path(tmpdir) / "analysis_report.txt"
            assert report_path.exists()


class TestAnalyzeImports:
    """Tests for analysis module imports."""

    def test_import_analyze_pilot(self):
        """Test that analyze_pilot can be imported."""
        from experiments.analyze_pilot import analyze_pilot

        assert callable(analyze_pilot)

    def test_import_compute_functions(self):
        """Test that compute functions can be imported."""
        from experiments.analyze_pilot import (
            compute_accuracy_by_benchmark,
            compute_accuracy_by_structure,
            compute_accuracy_matrix,
        )

        assert callable(compute_accuracy_by_structure)
        assert callable(compute_accuracy_by_benchmark)
        assert callable(compute_accuracy_matrix)

    def test_import_generate_report(self):
        """Test that generate_report can be imported."""
        from experiments.analyze_pilot import generate_report

        assert callable(generate_report)
