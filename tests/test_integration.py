"""Integration tests for end-to-end experiment flow.

This module tests the complete pipeline:
1. Governance structures process questions
2. Benchmarks evaluate responses
3. Experiment runner coordinates everything
4. Analysis produces meaningful results
"""

import tempfile
from typing import Dict, List

import pandas as pd
import pytest

from backend.evaluation.base import Benchmark, EvalResult, Question
from backend.governance import (
    CouncilResult,
    DeliberateSynthesizeStructure,
    DeliberateVoteStructure,
    GovernanceStructure,
    IndependentRankSynthesize,
    MajorityVoteStructure,
)
from experiments.analyze_pilot import (
    compute_accuracy_by_structure,
    compute_accuracy_matrix,
    generate_report,
)
from experiments.run_pilot import run_experiment, save_results


# ============================================================================
# Mock fixtures for integration testing
# ============================================================================


class MockGSM8KBenchmark(Benchmark):
    """Mock GSM8K benchmark for integration testing."""

    def __init__(self):
        self._questions = [
            Question(
                id="gsm8k_0",
                text="Janet has 3 apples. She buys 2 more. How many apples does she have?",
                ground_truth="5",
                metadata={"type": "addition"},
            ),
            Question(
                id="gsm8k_1",
                text="Tom has 10 cookies. He eats 4. How many cookies are left?",
                ground_truth="6",
                metadata={"type": "subtraction"},
            ),
        ]

    @property
    def name(self) -> str:
        return "GSM8K"

    def load_questions(self, n=None) -> List[Question]:
        if n is not None:
            return self._questions[:n]
        return self._questions

    def evaluate(self, question: Question, response: str) -> EvalResult:
        # Extract number from response
        import re

        match = re.search(r"FINAL ANSWER:\s*(\d+)", response, re.IGNORECASE)
        if match:
            predicted = match.group(1)
        else:
            # Try to find any number
            numbers = re.findall(r"\d+", response)
            predicted = numbers[-1] if numbers else "[no number]"

        is_correct = predicted == question.ground_truth

        return EvalResult(
            question_id=question.id,
            is_correct=is_correct,
            predicted=predicted,
            expected=question.ground_truth,
        )


class MockTruthfulQABenchmark(Benchmark):
    """Mock TruthfulQA benchmark for integration testing."""

    def __init__(self):
        self._questions = [
            Question(
                id="truthfulqa_0",
                text="What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid\n\nFINAL ANSWER: [letter]",
                ground_truth="B",
                metadata={"category": "geography"},
            ),
            Question(
                id="truthfulqa_1",
                text="Which planet is known as the Red Planet?\nA. Venus\nB. Jupiter\nC. Mars\nD. Saturn\n\nFINAL ANSWER: [letter]",
                ground_truth="C",
                metadata={"category": "science"},
            ),
        ]

    @property
    def name(self) -> str:
        return "TruthfulQA"

    def load_questions(self, n=None) -> List[Question]:
        if n is not None:
            return self._questions[:n]
        return self._questions

    def evaluate(self, question: Question, response: str) -> EvalResult:
        import re

        match = re.search(r"FINAL ANSWER:\s*([A-Da-d])", response, re.IGNORECASE)
        if match:
            predicted = match.group(1).upper()
        else:
            # Try to find any letter A-D
            match = re.search(r"\b([A-Da-d])\b", response)
            predicted = match.group(1).upper() if match else "[no letter]"

        is_correct = predicted == question.ground_truth

        return EvalResult(
            question_id=question.id,
            is_correct=is_correct,
            predicted=predicted,
            expected=question.ground_truth,
        )


@pytest.fixture
def mock_openrouter(monkeypatch):
    """Mock OpenRouter API calls for all governance structures."""
    call_count = {"count": 0}

    async def mock_query_model(model: str, messages: List[Dict], temperature: float = 0.0, timeout=None) -> Dict:
        """Return appropriate mock responses based on context."""
        call_count["count"] += 1
        prompt = messages[-1]["content"] if messages else ""

        # Detect question type and generate appropriate response
        if "apples" in prompt.lower():
            return {"content": "Janet has 3 + 2 = 5 apples. FINAL ANSWER: 5"}
        elif "cookies" in prompt.lower():
            return {"content": "Tom has 10 - 4 = 6 cookies. FINAL ANSWER: 6"}
        elif "capital of france" in prompt.lower():
            return {"content": "Paris is the capital of France. FINAL ANSWER: B"}
        elif "red planet" in prompt.lower():
            return {"content": "Mars is the Red Planet. FINAL ANSWER: C"}
        elif "rank" in prompt.lower():
            # Ranking response for Structure A
            return {"content": "FINAL RANKING: Response A > Response B > Response C > Response D"}
        elif "synthesize" in prompt.lower() or "chairman" in prompt.lower():
            # Synthesis response
            if "apples" in prompt.lower() or "5" in prompt:
                return {"content": "Based on the council's responses, the answer is 5 apples. FINAL ANSWER: 5"}
            elif "cookies" in prompt.lower() or "6" in prompt:
                return {"content": "The council agrees: 6 cookies remain. FINAL ANSWER: 6"}
            elif "paris" in prompt.lower() or "capital" in prompt.lower():
                return {"content": "The council agrees Paris is the capital. FINAL ANSWER: B"}
            elif "mars" in prompt.lower() or "red planet" in prompt.lower():
                return {"content": "Mars is indeed the Red Planet. FINAL ANSWER: C"}
            else:
                return {"content": "Synthesized answer. FINAL ANSWER: 42"}
        elif "reconsider" in prompt.lower() or "deliberat" in prompt.lower():
            # Deliberation response - maintain original answer
            if "apples" in prompt.lower():
                return {"content": "After considering others, I maintain: 5 apples. FINAL ANSWER: 5"}
            elif "cookies" in prompt.lower():
                return {"content": "I agree with the consensus: 6 cookies. FINAL ANSWER: 6"}
            elif "capital" in prompt.lower() or "france" in prompt.lower():
                return {"content": "Paris remains correct. FINAL ANSWER: B"}
            elif "red planet" in prompt.lower() or "mars" in prompt.lower():
                return {"content": "Mars is confirmed. FINAL ANSWER: C"}
            else:
                return {"content": "After deliberation. FINAL ANSWER: 42"}
        else:
            # Default response
            return {"content": f"Response from {model}. FINAL ANSWER: 42"}

    async def mock_query_models_parallel(
        models: List[str], messages: List[Dict], temperature: float = 0.0, timeout=None
    ) -> Dict[str, Dict]:
        """Mock parallel queries."""
        results = {}
        for model in models:
            results[model] = await mock_query_model(model, messages)
        return results

    # Patch all governance structure modules (use importlib to avoid namespace collisions)
    import importlib
    base_module = importlib.import_module("backend.governance.base")
    struct_a = importlib.import_module("backend.governance.independent_rank_synthesize")
    struct_c = importlib.import_module("backend.governance.deliberate_vote")
    struct_d = importlib.import_module("backend.governance.deliberate_synthesize")

    # Base class has both query_model (for chairman) and query_models_parallel (for Stage 1)
    monkeypatch.setattr(base_module, "query_model", mock_query_model)
    monkeypatch.setattr(base_module, "query_models_parallel", mock_query_models_parallel)

    # These structures still import query_model directly for their specific stages
    monkeypatch.setattr(struct_a, "query_model", mock_query_model)  # synthesis
    monkeypatch.setattr(struct_a, "query_models_parallel", mock_query_models_parallel)  # Stage 2 rankings
    monkeypatch.setattr(struct_c, "query_model", mock_query_model)  # deliberation
    monkeypatch.setattr(struct_d, "query_model", mock_query_model)  # deliberation

    return call_count


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndToEndExperiment:
    """Test complete experiment flow from start to finish."""

    @pytest.mark.asyncio
    async def test_mini_experiment_runs(self, mock_openrouter):
        """Test that a mini experiment runs successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up structures
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [
                IndependentRankSynthesize(council_models, chairman),
                MajorityVoteStructure(council_models, chairman),
            ]

            # Set up benchmarks
            benchmarks = [MockGSM8KBenchmark()]

            # Run experiment
            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=2,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Verify results
            assert len(df) == 4  # 2 structures × 2 questions × 1 replication
            assert "is_correct" in df.columns
            assert df["is_correct"].notna().all()

    @pytest.mark.asyncio
    async def test_all_four_structures(self, mock_openrouter):
        """Test all four governance structures in one experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [
                IndependentRankSynthesize(council_models, chairman),
                MajorityVoteStructure(council_models, chairman),
                DeliberateVoteStructure(council_models, chairman),
                DeliberateSynthesizeStructure(council_models, chairman),
            ]

            benchmarks = [MockGSM8KBenchmark()]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=1,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Verify all structures ran
            assert len(df) == 4
            assert set(df["structure"].unique()) == {
                "Independent → Rank → Synthesize",
                "Independent → Majority Vote",
                "Independent → Deliberate → Vote",
                "Independent → Deliberate → Synthesize",
            }

    @pytest.mark.asyncio
    async def test_multiple_benchmarks(self, mock_openrouter):
        """Test experiment with multiple benchmarks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [MajorityVoteStructure(council_models, chairman)]

            benchmarks = [
                MockGSM8KBenchmark(),
                MockTruthfulQABenchmark(),
            ]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=2,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Verify both benchmarks ran
            assert len(df) == 4  # 1 structure × 2 questions × 2 benchmarks
            assert set(df["benchmark"].unique()) == {"GSM8K", "TruthfulQA"}

    @pytest.mark.asyncio
    async def test_multiple_replications(self, mock_openrouter):
        """Test experiment with multiple replications."""
        with tempfile.TemporaryDirectory() as tmpdir:
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [MajorityVoteStructure(council_models, chairman)]
            benchmarks = [MockGSM8KBenchmark()]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=1,
                n_replications=3,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Verify replications
            assert len(df) == 3
            assert set(df["replication"].unique()) == {0, 1, 2}

    @pytest.mark.asyncio
    async def test_results_are_correct(self, mock_openrouter):
        """Test that mock responses produce correct evaluations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [MajorityVoteStructure(council_models, chairman)]
            benchmarks = [MockGSM8KBenchmark()]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=2,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # With our mock, all answers should be correct
            assert df["is_correct"].all()


class TestExperimentToAnalysis:
    """Test the flow from experiment to analysis."""

    @pytest.mark.asyncio
    async def test_analysis_on_experiment_results(self, mock_openrouter):
        """Test that analysis works on experiment output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [
                IndependentRankSynthesize(council_models, chairman),
                MajorityVoteStructure(council_models, chairman),
            ]
            benchmarks = [MockGSM8KBenchmark(), MockTruthfulQABenchmark()]

            # Run experiment
            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=2,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Run analysis
            accuracy_by_struct = compute_accuracy_by_structure(df)
            accuracy_matrix = compute_accuracy_matrix(df)
            report = generate_report(df)

            # Verify analysis output
            assert len(accuracy_by_struct) == 2
            assert not accuracy_matrix.empty
            assert "ANALYSIS REPORT" in report
            assert "GSM8K" in report
            assert "TruthfulQA" in report

    @pytest.mark.asyncio
    async def test_report_contains_all_structures(self, mock_openrouter):
        """Test that report includes all tested structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [
                IndependentRankSynthesize(council_models, chairman),
                MajorityVoteStructure(council_models, chairman),
                DeliberateVoteStructure(council_models, chairman),
                DeliberateSynthesizeStructure(council_models, chairman),
            ]
            benchmarks = [MockGSM8KBenchmark()]

            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=1,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            report = generate_report(df)

            # All structure names should appear in report
            assert "Independent → Rank → Synthesize" in report
            assert "Independent → Majority Vote" in report
            assert "Independent → Deliberate → Vote" in report
            assert "Independent → Deliberate → Synthesize" in report


class TestDataPersistence:
    """Test that data is correctly saved and can be resumed."""

    @pytest.mark.asyncio
    async def test_results_saved_to_file(self, mock_openrouter):
        """Test that results are persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [MajorityVoteStructure(council_models, chairman)]
            benchmarks = [MockGSM8KBenchmark()]

            await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=2,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Verify file exists
            from pathlib import Path

            results_file = Path(tmpdir) / "pilot_results.json"
            assert results_file.exists()

            # Verify contents
            import json

            with open(results_file) as f:
                saved_results = json.load(f)

            assert len(saved_results) == 2

    @pytest.mark.asyncio
    async def test_experiment_can_resume(self, mock_openrouter):
        """Test that experiment can resume from saved state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            council_models = ["model1", "model2"]
            chairman = "chairman"

            structures = [MajorityVoteStructure(council_models, chairman)]
            benchmarks = [MockGSM8KBenchmark()]

            # Run partial experiment
            df1 = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=1,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            initial_count = len(df1)

            # "Resume" with more questions - but same config should skip completed
            df2 = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=1,
                n_replications=1,
                output_dir=tmpdir,
                resume=True,
                verbose=False,
            )

            # Should have same count (no new work done)
            assert len(df2) == initial_count


class TestStructureSpecificBehavior:
    """Test that each structure behaves correctly."""

    @pytest.mark.asyncio
    async def test_structure_a_has_rankings(self, mock_openrouter):
        """Test that Structure A produces ranking data."""
        council_models = ["model1", "model2"]
        chairman = "chairman"

        structure = IndependentRankSynthesize(council_models, chairman)
        result = await structure.run("What is 2+2?")

        assert result.stage2_data is not None
        assert "rankings" in result.stage2_data

    @pytest.mark.asyncio
    async def test_structure_b_has_vote_data(self, mock_openrouter):
        """Test that Structure B produces vote data."""
        council_models = ["model1", "model2"]
        chairman = "chairman"

        structure = MajorityVoteStructure(council_models, chairman)
        result = await structure.run("What is 2+2?")

        assert result.stage3_data is not None
        assert "vote_result" in result.stage3_data

    @pytest.mark.asyncio
    async def test_structure_c_has_deliberation(self, mock_openrouter):
        """Test that Structure C produces deliberation data."""
        council_models = ["model1", "model2"]
        chairman = "chairman"

        structure = DeliberateVoteStructure(council_models, chairman)
        result = await structure.run("What is 2+2?")

        assert result.stage2_data is not None
        assert "deliberation_responses" in result.stage2_data

    @pytest.mark.asyncio
    async def test_structure_d_has_deliberation_and_synthesis(self, mock_openrouter):
        """Test that Structure D has both deliberation and synthesis."""
        council_models = ["model1", "model2"]
        chairman = "chairman"

        structure = DeliberateSynthesizeStructure(council_models, chairman)
        result = await structure.run("What is 2+2?")

        assert result.stage2_data is not None
        assert "deliberation_responses" in result.stage2_data
        assert result.stage3_data is not None
        assert "synthesis" in result.stage3_data


class TestBenchmarkEvaluation:
    """Test benchmark evaluation integration."""

    def test_gsm8k_evaluates_correctly(self):
        """Test GSM8K benchmark evaluation."""
        benchmark = MockGSM8KBenchmark()
        questions = benchmark.load_questions()

        # Test correct answer
        result = benchmark.evaluate(questions[0], "The answer is 5. FINAL ANSWER: 5")
        assert result.is_correct is True
        assert result.predicted == "5"

        # Test incorrect answer
        result = benchmark.evaluate(questions[0], "The answer is 3. FINAL ANSWER: 3")
        assert result.is_correct is False

    def test_truthfulqa_evaluates_correctly(self):
        """Test TruthfulQA benchmark evaluation."""
        benchmark = MockTruthfulQABenchmark()
        questions = benchmark.load_questions()

        # Test correct answer
        result = benchmark.evaluate(questions[0], "Paris is correct. FINAL ANSWER: B")
        assert result.is_correct is True
        assert result.predicted == "B"

        # Test incorrect answer
        result = benchmark.evaluate(questions[0], "I think London. FINAL ANSWER: A")
        assert result.is_correct is False


class TestFullPipelineIntegration:
    """Full pipeline integration test matching pilot study spec."""

    @pytest.mark.asyncio
    async def test_pilot_study_mini_version(self, mock_openrouter):
        """
        Run a mini version of the pilot study:
        - 2 questions per benchmark
        - 2 structures
        - 1 replication

        This is the final validation before real pilot.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure like real pilot
            council_models = ["openai/gpt-4", "anthropic/claude-3"]
            chairman = "openai/gpt-4"

            structures = [
                IndependentRankSynthesize(council_models, chairman),
                MajorityVoteStructure(council_models, chairman),
            ]

            benchmarks = [
                MockGSM8KBenchmark(),
                MockTruthfulQABenchmark(),
            ]

            # Run experiment
            df = await run_experiment(
                structures=structures,
                benchmarks=benchmarks,
                n_questions=2,
                n_replications=1,
                output_dir=tmpdir,
                resume=False,
                verbose=False,
            )

            # Validate structure
            # 2 structures × 2 benchmarks × 2 questions × 1 replication = 8 trials
            assert len(df) == 8

            # Validate all combinations present
            assert df["structure"].nunique() == 2
            assert df["benchmark"].nunique() == 2
            assert df["question_id"].nunique() == 4

            # Validate data quality
            assert df["is_correct"].notna().all()
            assert df["predicted"].notna().all()
            assert df["expected"].notna().all()

            # Run analysis
            accuracy_struct = compute_accuracy_by_structure(df)
            accuracy_matrix = compute_accuracy_matrix(df)
            report = generate_report(df, str(tmpdir) + "/report.txt")

            # Validate analysis outputs
            assert len(accuracy_struct) == 2
            assert accuracy_matrix.shape == (2, 2)  # 2 structures × 2 benchmarks
            assert "ANALYSIS REPORT" in report

            # Verify report file created
            from pathlib import Path

            assert (Path(tmpdir) / "report.txt").exists()

            print("\n" + "=" * 60)
            print("MINI PILOT STUDY COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Total trials: {len(df)}")
            print(f"Accuracy by structure:\n{accuracy_struct.to_string()}")
            print(f"\nAccuracy matrix:\n{accuracy_matrix.to_string()}")
