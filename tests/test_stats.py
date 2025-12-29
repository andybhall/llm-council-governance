"""Tests for statistical utilities."""

import numpy as np
import pandas as pd
import pytest

from experiments.stats import paired_bootstrap_accuracy_diff, run_pairwise_bootstrap_tests


class TestPairedBootstrapAccuracyDiff:
    """Tests for paired_bootstrap_accuracy_diff function."""

    def test_structure_a_beats_b_consistently(self):
        """Test with structure A always correct, B always wrong."""
        # Create a deterministic case where A always beats B
        data = []
        for q_id in range(20):
            # Structure A always correct
            data.append({
                "structure": "A",
                "benchmark": "test",
                "question_id": q_id,
                "is_correct": True,
            })
            # Structure B always wrong
            data.append({
                "structure": "B",
                "benchmark": "test",
                "question_id": q_id,
                "is_correct": False,
            })

        df = pd.DataFrame(data)
        result = paired_bootstrap_accuracy_diff(df, "A", "B", n_boot=1000, seed=42)

        # A - B should be 1.0 (100% - 0%)
        assert result["diff"] == 1.0
        assert result["ci_low"] > 0.5  # CI should be strictly positive
        assert result["ci_high"] == 1.0
        assert result["n_questions"] == 20

    def test_no_difference(self):
        """Test with equal performance."""
        data = []
        for q_id in range(20):
            # Both structures same
            is_correct = q_id % 2 == 0
            data.append({
                "structure": "A",
                "benchmark": "test",
                "question_id": q_id,
                "is_correct": is_correct,
            })
            data.append({
                "structure": "B",
                "benchmark": "test",
                "question_id": q_id,
                "is_correct": is_correct,
            })

        df = pd.DataFrame(data)
        result = paired_bootstrap_accuracy_diff(df, "A", "B", n_boot=1000, seed=42)

        # Difference should be 0
        assert result["diff"] == 0.0
        assert result["ci_low"] == 0.0
        assert result["ci_high"] == 0.0

    def test_structure_b_beats_a(self):
        """Test with B better than A."""
        data = []
        for q_id in range(20):
            # B wins on 75% of questions
            b_correct = q_id % 4 != 0
            data.append({
                "structure": "A",
                "benchmark": "test",
                "question_id": q_id,
                "is_correct": False,
            })
            data.append({
                "structure": "B",
                "benchmark": "test",
                "question_id": q_id,
                "is_correct": b_correct,
            })

        df = pd.DataFrame(data)
        result = paired_bootstrap_accuracy_diff(df, "A", "B", n_boot=1000, seed=42)

        # A - B should be negative (A is worse)
        assert result["diff"] < 0
        assert result["ci_high"] < 0  # CI should be strictly negative

    def test_handles_replications(self):
        """Test that replications are averaged per question."""
        data = []
        for q_id in range(10):
            # 3 replications per question per structure
            for rep in range(3):
                data.append({
                    "structure": "A",
                    "benchmark": "test",
                    "question_id": q_id,
                    "is_correct": True,
                    "replication": rep,
                })
                data.append({
                    "structure": "B",
                    "benchmark": "test",
                    "question_id": q_id,
                    "is_correct": False,
                    "replication": rep,
                })

        df = pd.DataFrame(data)
        result = paired_bootstrap_accuracy_diff(df, "A", "B", n_boot=100, seed=42)

        # Should have 10 questions (not 30)
        assert result["n_questions"] == 10
        assert result["diff"] == 1.0

    def test_handles_multiple_benchmarks(self):
        """Test with questions from multiple benchmarks."""
        data = []
        # 5 questions from GSM8K
        for q_id in range(5):
            data.append({
                "structure": "A", "benchmark": "GSM8K", "question_id": q_id, "is_correct": True
            })
            data.append({
                "structure": "B", "benchmark": "GSM8K", "question_id": q_id, "is_correct": False
            })
        # 5 questions from TruthfulQA
        for q_id in range(5):
            data.append({
                "structure": "A", "benchmark": "TruthfulQA", "question_id": q_id, "is_correct": True
            })
            data.append({
                "structure": "B", "benchmark": "TruthfulQA", "question_id": q_id, "is_correct": False
            })

        df = pd.DataFrame(data)
        result = paired_bootstrap_accuracy_diff(df, "A", "B", n_boot=100, seed=42)

        # Should have 10 total questions
        assert result["n_questions"] == 10

    def test_no_common_questions_returns_default(self):
        """Test handling when structures have no common questions."""
        data = [
            {"structure": "A", "benchmark": "test", "question_id": 0, "is_correct": True},
            {"structure": "B", "benchmark": "test", "question_id": 1, "is_correct": False},
        ]
        df = pd.DataFrame(data)
        result = paired_bootstrap_accuracy_diff(df, "A", "B", n_boot=100, seed=42)

        assert result["n_questions"] == 0
        assert result["diff"] == 0.0
        assert result["p_value"] == 1.0

    def test_deterministic_with_seed(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(None)  # Reset any global state

        data = []
        for q_id in range(50):
            # A better on ~60% of questions
            a_correct = q_id % 5 != 0
            b_correct = q_id % 3 != 0
            data.append({
                "structure": "A", "benchmark": "test", "question_id": q_id, "is_correct": a_correct
            })
            data.append({
                "structure": "B", "benchmark": "test", "question_id": q_id, "is_correct": b_correct
            })

        df = pd.DataFrame(data)

        result1 = paired_bootstrap_accuracy_diff(df, "A", "B", n_boot=1000, seed=42)
        result2 = paired_bootstrap_accuracy_diff(df, "A", "B", n_boot=1000, seed=42)

        assert result1["ci_low"] == result2["ci_low"]
        assert result1["ci_high"] == result2["ci_high"]


class TestRunPairwiseBootstrapTests:
    """Tests for run_pairwise_bootstrap_tests function."""

    def test_compares_all_structures_to_baseline(self):
        """Test that all structures are compared to baseline."""
        data = []
        for q_id in range(10):
            for struct in ["baseline", "A", "B", "C"]:
                data.append({
                    "structure": struct,
                    "benchmark": "test",
                    "question_id": q_id,
                    "is_correct": struct != "baseline",
                })

        df = pd.DataFrame(data)
        results = run_pairwise_bootstrap_tests(df, baseline="baseline", n_boot=100, seed=42)

        # Should have 3 rows (A, B, C vs baseline)
        assert len(results) == 3
        assert set(results["structure"]) == {"A", "B", "C"}
        assert all(results["baseline"] == "baseline")

    def test_respects_structures_argument(self):
        """Test that structures argument filters comparisons."""
        data = []
        for q_id in range(10):
            for struct in ["baseline", "A", "B", "C"]:
                data.append({
                    "structure": struct,
                    "benchmark": "test",
                    "question_id": q_id,
                    "is_correct": True,
                })

        df = pd.DataFrame(data)
        results = run_pairwise_bootstrap_tests(
            df, baseline="baseline", structures=["A", "B"], n_boot=100, seed=42
        )

        # Should only have A and B
        assert len(results) == 2
        assert set(results["structure"]) == {"A", "B"}

    def test_result_columns(self):
        """Test that result DataFrame has expected columns."""
        data = []
        for q_id in range(10):
            for struct in ["baseline", "A"]:
                data.append({
                    "structure": struct,
                    "benchmark": "test",
                    "question_id": q_id,
                    "is_correct": True,
                })

        df = pd.DataFrame(data)
        results = run_pairwise_bootstrap_tests(df, baseline="baseline", n_boot=100, seed=42)

        expected_columns = {"structure", "baseline", "diff", "ci_low", "ci_high", "p_value", "n_questions"}
        assert set(results.columns) == expected_columns
