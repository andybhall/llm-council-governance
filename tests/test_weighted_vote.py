"""Tests for weighted voting functionality and Structure E."""

import json
import tempfile
from pathlib import Path

import pytest

from backend.governance.structure_e import WeightedMajorityVote
from backend.governance.utils import (
    smart_weighted_majority_vote,
    weighted_majority_vote,
    weighted_majority_vote_letter,
    weighted_majority_vote_numeric,
)


class TestWeightedMajorityVote:
    """Tests for weighted_majority_vote function."""

    def test_basic_weighted_vote(self):
        """Test basic weighted voting with clear winner."""
        answers = ["A", "B", "B"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 0.9, "m2": 0.5, "m3": 0.5}
        # m1 votes A with weight 0.9 = 0.9
        # m2+m3 vote B with weight 0.5+0.5 = 1.0
        # B wins
        result = weighted_majority_vote(answers, models, weights)
        assert result == "B"

    def test_high_weight_beats_count(self):
        """Test that high weight can beat raw count."""
        answers = ["A", "B", "B", "B"]
        models = ["expert", "m1", "m2", "m3"]
        weights = {"expert": 3.0, "m1": 0.3, "m2": 0.3, "m3": 0.3}
        # expert votes A with weight 3.0
        # m1+m2+m3 vote B with weight 0.9 total
        # A wins despite fewer votes
        result = weighted_majority_vote(answers, models, weights)
        assert result == "A"

    def test_equal_weights_acts_like_majority(self):
        """Test that equal weights behaves like regular majority vote."""
        answers = ["A", "A", "B"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = weighted_majority_vote(answers, models, weights)
        assert result == "A"

    def test_default_weight_of_one(self):
        """Test that missing weights default to 1.0."""
        answers = ["A", "B", "B"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 2.5}  # m2 and m3 will get default 1.0
        # m1 votes A with weight 2.5
        # m2+m3 vote B with weight 2.0 total
        # A wins
        result = weighted_majority_vote(answers, models, weights)
        assert result == "A"

    def test_tiebreaker_with_weights(self):
        """Test tiebreaker when weighted votes are equal."""
        answers = ["A", "B"]
        models = ["m1", "m2"]
        weights = {"m1": 1.0, "m2": 1.0}
        result = weighted_majority_vote(answers, models, weights, tiebreaker="B")
        assert result == "B"

    def test_empty_answers(self):
        """Test with empty answers list."""
        result = weighted_majority_vote([], [], {})
        assert result == ""

    def test_none_answers_filtered(self):
        """Test that None answers are filtered out."""
        answers = ["A", None, "A"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = weighted_majority_vote(answers, models, weights)
        assert result == "A"

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError):
            weighted_majority_vote(["A", "B"], ["m1"], {})


class TestWeightedMajorityVoteNumeric:
    """Tests for weighted_majority_vote_numeric function."""

    def test_numeric_weighted_vote(self):
        """Test weighted voting with numeric answers."""
        answers = ["42", "43", "43"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 2.0, "m2": 0.5, "m3": 0.5}
        # 42: weight 2.0, 43: weight 1.0
        result = weighted_majority_vote_numeric(answers, models, weights)
        assert result == "42"

    def test_numeric_normalization(self):
        """Test that numeric values are normalized for comparison."""
        answers = ["42", "42.0", "$43"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        # 42 and 42.0 are the same value = weight 2.0
        result = weighted_majority_vote_numeric(answers, models, weights)
        assert result in ["42", "42.0"]

    def test_empty_returns_none(self):
        """Test empty list returns None."""
        result = weighted_majority_vote_numeric([], [], {})
        assert result is None


class TestWeightedMajorityVoteLetter:
    """Tests for weighted_majority_vote_letter function."""

    def test_letter_weighted_vote(self):
        """Test weighted voting with letter answers."""
        answers = ["A", "B", "B"]
        models = ["expert", "m1", "m2"]
        weights = {"expert": 3.0, "m1": 0.4, "m2": 0.4}
        result = weighted_majority_vote_letter(answers, models, weights)
        assert result == "A"

    def test_case_insensitive(self):
        """Test that letter comparison is case insensitive."""
        answers = ["a", "A", "B"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = weighted_majority_vote_letter(answers, models, weights)
        assert result == "A"

    def test_returns_uppercase(self):
        """Test that result is uppercase."""
        answers = ["a", "a", "b"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = weighted_majority_vote_letter(answers, models, weights)
        assert result == "A"


class TestSmartWeightedMajorityVote:
    """Tests for smart_weighted_majority_vote function."""

    def test_auto_detect_numeric(self):
        """Test auto-detection of numeric answers."""
        answers = ["42", "42.0", "43"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = smart_weighted_majority_vote(answers, models, weights)
        assert result in ["42", "42.0"]

    def test_auto_detect_letter(self):
        """Test auto-detection of letter answers."""
        answers = ["A", "a", "B"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = smart_weighted_majority_vote(answers, models, weights)
        assert result == "A"

    def test_fallback_to_string(self):
        """Test fallback to string comparison for mixed types."""
        answers = ["yes", "YES", "no"]
        models = ["m1", "m2", "m3"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        result = smart_weighted_majority_vote(answers, models, weights)
        assert result.lower() == "yes"

    def test_filters_none_answers(self):
        """Test that None answers are filtered before processing."""
        answers = [None, "42", "42", None]
        models = ["m1", "m2", "m3", "m4"]
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0, "m4": 1.0}
        result = smart_weighted_majority_vote(answers, models, weights)
        assert result == "42"

    def test_empty_after_filtering_returns_empty(self):
        """Test that all-None answers return empty string."""
        answers = [None, None]
        models = ["m1", "m2"]
        weights = {"m1": 1.0, "m2": 1.0}
        result = smart_weighted_majority_vote(answers, models, weights)
        assert result == ""


class TestWeightedMajorityVoteStructure:
    """Tests for WeightedMajorityVote governance structure."""

    @pytest.fixture
    def mock_openrouter(self, monkeypatch):
        """Mock openrouter API calls for testing."""
        call_log = []

        async def mock_query_model(model, messages):
            call_log.append({"type": "single", "model": model, "messages": messages})
            return {"content": "The chairman thinks 4. FINAL ANSWER: 4"}

        async def mock_query_models_parallel(models, messages):
            call_log.append({"type": "parallel", "models": models, "messages": messages})
            # expert (high weight) says 5, others say 4
            results = {}
            for model in models:
                if "expert" in model:
                    results[model] = {"content": f"{model} says FINAL ANSWER: 5"}
                else:
                    results[model] = {"content": f"{model} says FINAL ANSWER: 4"}
            return results

        import backend.governance.structure_e as struct_e_module

        monkeypatch.setattr(struct_e_module, "query_model", mock_query_model)
        monkeypatch.setattr(struct_e_module, "query_models_parallel", mock_query_models_parallel)

        return call_log

    def test_structure_name(self):
        """Test that structure has correct name."""
        structure = WeightedMajorityVote(
            council_models=["m1", "m2"],
            chairman_model="chair",
            weights={"m1": 1.0, "m2": 1.0},
        )
        assert structure.name == "Independent → Weighted Majority Vote"

    def test_loads_weights_from_dict(self):
        """Test loading weights from provided dictionary."""
        weights = {"m1": 0.8, "m2": 0.6}
        structure = WeightedMajorityVote(
            council_models=["m1", "m2"],
            chairman_model="chair",
            weights=weights,
        )
        assert structure.weights == weights

    def test_loads_weights_from_file(self):
        """Test loading weights from JSON file."""
        weights = {"m1": 0.75, "m2": 0.85}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(weights, f)
            weights_path = f.name

        try:
            structure = WeightedMajorityVote(
                council_models=["m1", "m2"],
                chairman_model="chair",
                weights_file=weights_path,
            )
            assert structure.weights == weights
        finally:
            Path(weights_path).unlink()

    def test_default_weights_if_no_file(self):
        """Test that default weights are used if no file exists."""
        structure = WeightedMajorityVote(
            council_models=["m1", "m2"],
            chairman_model="chair",
            weights_file="/nonexistent/path.json",
        )
        # Should default to 1.0 for each model
        assert structure.weights == {"m1": 1.0, "m2": 1.0}

    @pytest.mark.asyncio
    async def test_structure_e_runs(self, mock_openrouter):
        """Verify Structure E executes successfully."""
        structure = WeightedMajorityVote(
            council_models=["model1", "model2", "expert"],
            chairman_model="chairman",
            weights={"expert": 3.0, "model1": 1.0, "model2": 1.0},
        )
        result = await structure.run("What is 2+2?")

        assert result.final_answer is not None
        assert len(result.stage1_responses) == 3

    @pytest.mark.asyncio
    async def test_high_weight_model_wins(self, mock_openrouter):
        """Test that high-weight model's answer wins."""
        structure = WeightedMajorityVote(
            council_models=["model1", "model2", "expert"],
            chairman_model="chairman",
            weights={"expert": 10.0, "model1": 1.0, "model2": 1.0},
        )
        result = await structure.run("What is 2+2?")

        # Expert with weight 10 says 5, others say 4
        # Expert should win despite being outnumbered
        assert result.final_answer == "5"

    @pytest.mark.asyncio
    async def test_stage2_includes_weights(self, mock_openrouter):
        """Test that stage2 data includes weights used."""
        structure = WeightedMajorityVote(
            council_models=["model1", "expert"],
            chairman_model="chairman",
            weights={"expert": 2.0, "model1": 1.0},
        )
        result = await structure.run("What is 2+2?")

        assert "weights_used" in result.stage2_data
        assert result.stage2_data["weights_used"]["expert"] == 2.0
        assert result.stage2_data["weights_used"]["model1"] == 1.0

    @pytest.mark.asyncio
    async def test_api_call_sequence(self, mock_openrouter):
        """Verify correct sequence of API calls."""
        structure = WeightedMajorityVote(
            council_models=["model1", "model2"],
            chairman_model="chairman",
            weights={"model1": 1.0, "model2": 1.0},
        )
        await structure.run("What is 2+2?")

        # Should have: Stage 1 (parallel), Stage 3 chairman (single)
        assert len(mock_openrouter) == 2
        assert mock_openrouter[0]["type"] == "parallel"
        assert mock_openrouter[1]["type"] == "single"


class TestWeightedVoteImports:
    """Test that weighted vote functions can be imported from governance package."""

    def test_import_weighted_vote_structure(self):
        """Test importing WeightedMajorityVote from package."""
        from backend.governance import WeightedMajorityVote

        assert WeightedMajorityVote is not None

    def test_import_weighted_vote_function(self):
        """Test importing weighted_majority_vote from package."""
        from backend.governance import weighted_majority_vote

        assert callable(weighted_majority_vote)

    def test_import_smart_weighted_vote(self):
        """Test importing smart_weighted_majority_vote from package."""
        from backend.governance import smart_weighted_majority_vote

        assert callable(smart_weighted_majority_vote)
