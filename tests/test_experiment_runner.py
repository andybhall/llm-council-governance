"""Tests for experiment runner."""

import json
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest

from backend.evaluation.base import Benchmark, EvalResult, Question
from backend.governance.base import CouncilResult, GovernanceStructure
from experiments.run_pilot import (
    get_completed_keys,
    load_results,
    run_experiment,
    run_single_trial,
    save_results,
)


# Mock implementations for testing


class MockGovernanceStructure(GovernanceStructure):
    """Mock governance structure for testing."""

    def __init__(
        self,
        council_models: List[str],
        chairman_model: str,
        name_override: str = "MockStructure",
        answer: str = "42",
    ):
        super().__init__(council_models, chairman_model)
        self._name = name_override
        self._answer = answer

    @property
    def name(self) -> str:
        return self._name

    async def run(self, query: str) -> CouncilResult:
        return CouncilResult(
            final_answer=f"FINAL ANSWER: {self._answer}",
            stage1_responses={model: f"Response from {model}" for model in self.council_models},
            stage2_data={"mock": "data"},
            stage3_data={"mock": "synthesis"},
            metadata={"mock": True},
        )


class MockBenchmark(Benchmark):
    """Mock benchmark for testing."""

    def __init__(self, name: str = "MockBenchmark", n_questions: int = 3):
        self._name = name
        self._n_questions = n_questions

    @property
    def name(self) -> str:
        return self._name

    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        count = n if n is not None else self._n_questions
        return [
            Question(
                id=f"{self._name.lower()}_{i}",
                text=f"Question {i}?",
                ground_truth="42",
                metadata={"index": i},
            )
            for i in range(min(count, self._n_questions))
        ]

    def evaluate(self, question: Question, response: str) -> EvalResult:
        # Extract answer from response
        predicted = response.replace("FINAL ANSWER: ", "").strip()
        is_correct = predicted == question.ground_truth

        return EvalResult(
            question_id=question.id,
            is_correct=is_correct,
            predicted=predicted,
            expected=question.ground_truth,
        )


class TestSaveLoadResults:
    """Tests for save_results and load_results."""

    def test_save_and_load_results(self):
        """Test that results can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                {"benchmark": "test", "question_id": "q1", "is_correct": True},
                {"benchmark": "test", "question_id": "q2", "is_correct": False},
            ]

            save_results(results, tmpdir)
            loaded = load_results(tmpdir)

            assert loaded == results

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_results(tmpdir)
            assert loaded == []

    def test_save_creates_directory(self):
        """Test that save_results creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "path"
            results = [{"test": "data"}]

            save_results(results, str(nested_dir))

            assert nested_dir.exists()
            assert (nested_dir / "pilot_results.json").exists()

    def test_custom_filename(self):
        """Test saving with custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [{"test": "data"}]

            save_results(results, tmpdir, filename="custom.json")
            loaded = load_results(tmpdir, filename="custom.json")

            assert loaded == results


class TestGetCompletedKeys:
    """Tests for get_completed_keys."""

    def test_empty_results(self):
        """Test with empty results."""
        assert get_completed_keys([]) == set()

    def test_extracts_keys(self):
        """Test that keys are correctly extracted."""
        results = [
            {
                "benchmark": "GSM8K",
                "question_id": "q1",
                "structure": "A",
                "replication": 0,
            },
            {
                "benchmark": "GSM8K",
                "question_id": "q1",
                "structure": "A",
                "replication": 1,
            },
        ]

        keys = get_completed_keys(results)

        assert ("GSM8K", "q1", "A", 0) in keys
        assert ("GSM8K", "q1", "A", 1) in keys
        assert len(keys) == 2


class TestRunSingleTrial:
    """Tests for run_single_trial."""

    @pytest.mark.asyncio
    async def test_returns_correct_result(self):
        """Test that run_single_trial returns expected fields."""
        structure = MockGovernanceStructure(["m1", "m2"], "chairman")
        benchmark = MockBenchmark()
        question = benchmark.load_questions()[0]

        result = await run_single_trial(structure, question, benchmark)

        assert "is_correct" in result
        assert "predicted" in result
        assert "expected" in result
        assert "final_answer" in result
        assert "elapsed_time" in result
        assert "stage1_responses" in result

    @pytest.mark.asyncio
    async def test_evaluates_correctly(self):
        """Test that evaluation produces correct result."""
        structure = MockGovernanceStructure(["m1"], "chairman", answer="42")
        benchmark = MockBenchmark()
        question = Question(id="q1", text="What?", ground_truth="42")

        result = await run_single_trial(structure, question, benchmark)

        assert result["is_correct"] is True
        assert result["predicted"] == "42"
        assert result["expected"] == "42"

    @pytest.mark.asyncio
    async def test_elapsed_time_recorded(self):
        """Test that elapsed time is recorded."""
        structure = MockGovernanceStructure(["m1"], "chairman")
        benchmark = MockBenchmark()
        question = benchmark.load_questions()[0]

        result = await run_single_trial(structure, question, benchmark)

        assert result["elapsed_time"] >= 0


class TestRunExperiment:
    """Tests for run_experiment."""

    @pytest.mark.asyncio
    async def test_runs_all_combinations(self):
        """Test that all structure x question x replication combinations are run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structures = [
                MockGovernanceStructure(["m1"], "c", name_override="A"),
                MockGovernanceStructure(["m1"], "c", name_override="B"),
            ]
            benchmarks = [MockBenchmark(n_questions=2)]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=2,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # 2 structures x 2 questions x 2 replications = 8 trials
            assert len(df) == 8

    @pytest.mark.asyncio
    async def test_returns_dataframe(self):
        """Test that result is a DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structures = [MockGovernanceStructure(["m1"], "c")]
            benchmarks = [MockBenchmark(n_questions=1)]

            result = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            assert isinstance(result, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_dataframe_has_required_columns(self):
        """Test that DataFrame has all required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structures = [MockGovernanceStructure(["m1"], "c")]
            benchmarks = [MockBenchmark(n_questions=1)]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            required_columns = [
                "benchmark",
                "question_id",
                "structure",
                "replication",
                "is_correct",
                "predicted",
                "expected",
            ]
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.asyncio
    async def test_saves_results_incrementally(self):
        """Test that results are saved after each trial."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structures = [MockGovernanceStructure(["m1"], "c")]
            benchmarks = [MockBenchmark(n_questions=2)]

            await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Check that file exists and has results
            results = load_results(tmpdir)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_n_questions_limits_questions(self):
        """Test that n_questions parameter limits question count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structures = [MockGovernanceStructure(["m1"], "c")]
            benchmarks = [MockBenchmark(n_questions=10)]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=2,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            assert len(df) == 2


class TestResumeExperiment:
    """Tests for experiment resumption."""

    @pytest.mark.asyncio
    async def test_resume_skips_completed(self):
        """Test that resume skips already completed trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structures = [MockGovernanceStructure(["m1"], "c", name_override="A")]
            benchmarks = [MockBenchmark(name="TestBench", n_questions=2)]

            # Run first time
            await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Modify results file to simulate partial completion
            results = load_results(tmpdir)
            assert len(results) == 2

            # Run again with resume - should not add new results
            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=1,
                output_dir=tmpdir,
                resume=True,
                verbose=False,
            )

            # Should still have only 2 results (no duplicates)
            assert len(df) == 2

    @pytest.mark.asyncio
    async def test_resume_false_starts_fresh(self):
        """Test that resume=False starts fresh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structures = [MockGovernanceStructure(["m1"], "c")]
            benchmarks = [MockBenchmark(n_questions=1)]

            # Run first time
            await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Run again without resume
            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Should have just 1 result (fresh start)
            assert len(df) == 1


class TestMultipleBenchmarks:
    """Tests for running with multiple benchmarks."""

    @pytest.mark.asyncio
    async def test_multiple_benchmarks(self):
        """Test running experiment with multiple benchmarks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structures = [MockGovernanceStructure(["m1"], "c")]
            benchmarks = [
                MockBenchmark(name="Bench1", n_questions=2),
                MockBenchmark(name="Bench2", n_questions=3),
            ]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # 1 structure x (2 + 3) questions x 1 replication = 5 trials
            assert len(df) == 5

            # Check both benchmarks present
            assert set(df["benchmark"].unique()) == {"Bench1", "Bench2"}


class TestExperimentImports:
    """Tests for experiment module imports."""

    def test_import_run_experiment(self):
        """Test that run_experiment can be imported."""
        from experiments.run_pilot import run_experiment

        assert callable(run_experiment)

    def test_import_save_load(self):
        """Test that save/load functions can be imported."""
        from experiments.run_pilot import load_results, save_results

        assert callable(save_results)
        assert callable(load_results)

    def test_import_run_pilot(self):
        """Test that run_pilot can be imported."""
        from experiments.run_pilot import run_pilot

        assert callable(run_pilot)
