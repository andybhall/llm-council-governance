"""Tests for persona variant experiment runner and analysis."""

import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from experiments.run_persona_experiment import (
    save_results,
    load_results,
    get_completed_keys,
    create_persona_patches,
    run_persona_experiment,
)
from experiments.analyze_persona_experiment import (
    load_persona_experiment_results,
    compute_accuracy_by_structure,
    compute_accuracy_by_benchmark,
    get_prompt_variant_accuracy,
    generate_comparison_report,
)
from backend.persona_variants import PERSONA_MODELS, BASE_MODEL


class TestExperimentHelperFunctions:
    """Test helper functions for experiment management."""

    def test_get_completed_keys(self):
        """Test completed key extraction from results."""
        results = [
            {
                "benchmark": "GSM8K",
                "question_id": "q1",
                "structure": "Structure B",
                "replication": 0,
            },
            {
                "benchmark": "TruthfulQA",
                "question_id": "q2",
                "structure": "Structure A",
                "replication": 1,
            },
        ]

        keys = get_completed_keys(results)

        assert len(keys) == 2
        assert ("GSM8K", "q1", "Structure B", 0) in keys
        assert ("TruthfulQA", "q2", "Structure A", 1) in keys

    def test_save_and_load_results(self, tmp_path):
        """Test result serialization round-trip."""
        results = [{"test": "data", "number": 42}]
        output_dir = str(tmp_path)

        save_results(results, output_dir, "test.json")
        loaded = load_results(output_dir, "test.json")

        assert loaded == results

    def test_load_results_missing_file(self, tmp_path):
        """Test loading from non-existent file returns empty list."""
        loaded = load_results(str(tmp_path), "nonexistent.json")
        assert loaded == []

    def test_save_creates_directory(self, tmp_path):
        """Test save creates output directory if needed."""
        output_dir = str(tmp_path / "nested" / "dir")
        results = [{"test": "data"}]

        save_results(results, output_dir, "test.json")
        loaded = load_results(output_dir, "test.json")

        assert loaded == results


class TestCreatePersonaPatches:
    """Test the patching mechanism."""

    def test_creates_patches_for_all_structures(self):
        """Verify patches are created for all governance structures."""
        patches = create_persona_patches()

        # Should have 2 patches per structure (query_model + query_models_parallel)
        # For 4 structures (A, B, C, D): 8 patches
        assert len(patches) == 8

    def test_patches_are_applicable(self):
        """Verify patches can be started and stopped."""
        patches = create_persona_patches()

        # Start all patches
        for p in patches:
            p.start()

        # Stop all patches
        for p in patches:
            p.stop()

        # Should not raise any exceptions


class TestRunPersonaExperiment:
    """Test the main experiment runner."""

    @pytest.fixture
    def mock_question(self):
        """Create a mock question."""
        question = MagicMock()
        question.id = "q1"
        question.text = "What is 2+2?"
        question.ground_truth = "4"
        return question

    @pytest.fixture
    def mock_benchmark(self, mock_question):
        """Create mock benchmark."""
        benchmark = MagicMock()
        benchmark.name = "TestBenchmark"
        benchmark.load_questions.return_value = [mock_question]
        benchmark.evaluate.return_value = MagicMock(
            is_correct=True,
            predicted="4",
            expected="4"
        )
        return benchmark

    @pytest.fixture
    def mock_structure(self):
        """Create mock governance structure."""
        mock_result = MagicMock()
        mock_result.final_answer = "4"
        mock_result.stage1_responses = {"m1": "resp1"}
        mock_result.stage2_data = {}
        mock_result.stage3_data = {}
        mock_result.metadata = {}

        structure = MagicMock()
        structure.name = "Test Structure"
        structure.run = AsyncMock(return_value=mock_result)
        return structure

    @pytest.mark.asyncio
    async def test_experiment_runs_successfully(
        self, mock_benchmark, mock_structure, tmp_path
    ):
        """Test experiment completes without errors."""
        results_df = await run_persona_experiment(
            structures=[mock_structure],
            benchmarks=[mock_benchmark],
            n_questions=1,
            n_replications=1,
            output_dir=str(tmp_path),
            resume=False,
            verbose=False,
        )

        assert len(results_df) == 1
        assert results_df.iloc[0]["is_correct"] == True
        assert results_df.iloc[0]["base_model"] == BASE_MODEL

    @pytest.mark.asyncio
    async def test_experiment_saves_results(
        self, mock_benchmark, mock_structure, tmp_path
    ):
        """Test results are saved to disk."""
        await run_persona_experiment(
            structures=[mock_structure],
            benchmarks=[mock_benchmark],
            n_questions=1,
            n_replications=1,
            output_dir=str(tmp_path),
            resume=False,
            verbose=False,
        )

        # Results should be saved
        loaded = load_results(str(tmp_path))
        assert len(loaded) == 1

    @pytest.mark.asyncio
    async def test_experiment_includes_persona_info(
        self, mock_benchmark, mock_structure, tmp_path
    ):
        """Test results include persona information."""
        await run_persona_experiment(
            structures=[mock_structure],
            benchmarks=[mock_benchmark],
            n_questions=1,
            n_replications=1,
            output_dir=str(tmp_path),
            resume=False,
            verbose=False,
        )

        loaded = load_results(str(tmp_path))
        assert loaded[0]["base_model"] == BASE_MODEL
        assert loaded[0]["personas"] == PERSONA_MODELS

    @pytest.mark.asyncio
    async def test_experiment_handles_errors(
        self, mock_benchmark, tmp_path
    ):
        """Test experiment handles structure errors gracefully."""
        failing_structure = MagicMock()
        failing_structure.name = "Failing Structure"
        failing_structure.run = AsyncMock(side_effect=Exception("Test error"))

        await run_persona_experiment(
            structures=[failing_structure],
            benchmarks=[mock_benchmark],
            n_questions=1,
            n_replications=1,
            output_dir=str(tmp_path),
            resume=False,
            verbose=False,
        )

        loaded = load_results(str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0]["is_correct"] is None
        assert "Test error" in loaded[0]["error"]


class TestAnalysisFunctions:
    """Test analysis helper functions."""

    def test_compute_accuracy_by_structure(self):
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

    def test_compute_accuracy_by_benchmark(self):
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

    def test_get_prompt_variant_accuracy(self):
        """Gets accuracy from prompt experiment."""
        import pandas as pd
        df = pd.DataFrame([
            {"is_correct": True},
            {"is_correct": True},
            {"is_correct": False},
        ])

        result = get_prompt_variant_accuracy(df)
        assert result == pytest.approx(2/3)

    def test_get_prompt_variant_accuracy_empty(self):
        """Returns None for empty DataFrame."""
        import pandas as pd
        df = pd.DataFrame()

        result = get_prompt_variant_accuracy(df)
        assert result is None


class TestComparisonReport:
    """Test report generation."""

    def test_generates_report(self):
        """Generates non-empty report."""
        import pandas as pd

        persona_df = pd.DataFrame()
        pilot_df = pd.DataFrame()
        prompt_df = pd.DataFrame()

        report = generate_comparison_report(persona_df, pilot_df, prompt_df)

        assert "PERSONA VARIANT COUNCIL EXPERIMENT" in report
        assert BASE_MODEL in report

    def test_saves_report_to_file(self, tmp_path):
        """Saves report to specified path."""
        import pandas as pd

        persona_df = pd.DataFrame()
        pilot_df = pd.DataFrame()
        prompt_df = pd.DataFrame()
        output_path = str(tmp_path / "report.txt")

        generate_comparison_report(persona_df, pilot_df, prompt_df, output_path)

        assert Path(output_path).exists()

    def test_shows_comparison_when_data_available(self):
        """Shows comparison when datasets have data."""
        import pandas as pd

        persona_df = pd.DataFrame([
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
        prompt_df = pd.DataFrame([
            {"is_correct": True},
            {"is_correct": False},
        ])

        report = generate_comparison_report(persona_df, pilot_df, prompt_df)

        assert "COMPARISON" in report
        assert "100.0%" in report  # persona council accuracy


class TestRunPersonaPilot:
    """Test the pilot experiment function."""

    def test_imports_work(self):
        """Verify run_persona_pilot can be imported."""
        from experiments.run_persona_experiment import run_persona_pilot
        assert run_persona_pilot is not None

    def test_persona_models_used(self):
        """Verify persona pilot uses PERSONA_MODELS."""
        assert len(PERSONA_MODELS) == 4
        assert BASE_MODEL == "google/gemma-2-9b-it"
