"""Tests for prompt variant experiment runner."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from experiments.run_prompt_experiment import (
    save_results,
    load_results,
    get_completed_keys,
    create_variant_patches,
    run_prompt_experiment,
)
from backend.prompt_variants import VARIANT_MODELS, BASE_MODEL


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


class TestCreateVariantPatches:
    """Test the patching mechanism."""

    def test_creates_patches_for_all_structures(self):
        """Verify patches are created for all governance structures."""
        patches = create_variant_patches()

        # Should have:
        # - 1 base class patch for query_models_parallel (Stage 1)
        # - 1 struct_a query_model + 1 struct_a query_models_parallel (Stage 2)
        # - 1 query_model patch for each of B, C, D
        # Total: 6 patches
        assert len(patches) == 6

    def test_patches_are_applicable(self):
        """Verify patches can be started and stopped."""
        patches = create_variant_patches()

        # Start all patches
        for p in patches:
            p.start()

        # Stop all patches
        for p in patches:
            p.stop()

        # Should not raise any exceptions


class TestRunPromptExperiment:
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
        results_df = await run_prompt_experiment(
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
        await run_prompt_experiment(
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
    async def test_experiment_resumes_correctly(
        self, mock_benchmark, mock_structure, tmp_path
    ):
        """Test resumption skips completed trials."""
        # Pre-populate with existing result
        existing = [{
            "benchmark": "TestBenchmark",
            "question_id": "q1",
            "structure": "Test Structure",
            "replication": 0,
            "is_correct": True,
        }]
        save_results(existing, str(tmp_path))

        await run_prompt_experiment(
            structures=[mock_structure],
            benchmarks=[mock_benchmark],
            n_questions=1,
            n_replications=1,
            output_dir=str(tmp_path),
            resume=True,
            verbose=False,
        )

        # Should not run again (still just 1 result)
        mock_structure.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_experiment_includes_variant_info(
        self, mock_benchmark, mock_structure, tmp_path
    ):
        """Test results include prompt variant information."""
        await run_prompt_experiment(
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
        assert loaded[0]["prompt_variants"] == VARIANT_MODELS

    @pytest.mark.asyncio
    async def test_experiment_handles_errors(
        self, mock_benchmark, tmp_path
    ):
        """Test experiment handles structure errors gracefully."""
        failing_structure = MagicMock()
        failing_structure.name = "Failing Structure"
        failing_structure.run = AsyncMock(side_effect=Exception("Test error"))

        await run_prompt_experiment(
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

    @pytest.mark.asyncio
    async def test_experiment_multiple_replications(
        self, mock_benchmark, mock_structure, tmp_path
    ):
        """Test experiment runs multiple replications."""
        await run_prompt_experiment(
            structures=[mock_structure],
            benchmarks=[mock_benchmark],
            n_questions=1,
            n_replications=3,
            output_dir=str(tmp_path),
            resume=False,
            verbose=False,
        )

        loaded = load_results(str(tmp_path))
        assert len(loaded) == 3

        # Verify different replication numbers
        reps = [r["replication"] for r in loaded]
        assert sorted(reps) == [0, 1, 2]


class TestRunPromptPilot:
    """Test the pilot experiment function."""

    def test_imports_work(self):
        """Verify run_prompt_pilot can be imported."""
        from experiments.run_prompt_experiment import run_prompt_pilot
        assert run_prompt_pilot is not None

    def test_variant_models_used(self):
        """Verify prompt pilot uses VARIANT_MODELS."""
        # Just verify the constants are correct
        assert len(VARIANT_MODELS) == 4
        assert BASE_MODEL == "google/gemma-2-9b-it"
