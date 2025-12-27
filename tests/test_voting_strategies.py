"""Tests for VotingStrategy abstraction and implementations."""

import json
import tempfile
from pathlib import Path

import pytest

from backend.governance.voting import (
    VotingStrategy,
    MajorityVoteStrategy,
    WeightedMajorityVoteStrategy,
    OracleWeightedVoteStrategy,
    create_voting_strategies,
)


class TestMajorityVoteStrategy:
    """Tests for MajorityVoteStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = MajorityVoteStrategy()
        assert strategy.name == "Majority Vote"
        assert strategy.short_name == "MV"

    def test_clear_winner(self):
        """Test with clear majority winner."""
        strategy = MajorityVoteStrategy()
        stage1 = {
            "model1": "I think the answer is FINAL ANSWER: A",
            "model2": "The result is FINAL ANSWER: A",
            "model3": "After analysis: FINAL ANSWER: A",
            "model4": "My conclusion: FINAL ANSWER: B",
        }
        result = strategy.vote(stage1)
        assert result == "A"

    def test_tie_with_tiebreaker(self):
        """Test tie resolution with chairman tiebreaker."""
        strategy = MajorityVoteStrategy()
        stage1 = {
            "model1": "FINAL ANSWER: A",
            "model2": "FINAL ANSWER: A",
            "model3": "FINAL ANSWER: B",
            "model4": "FINAL ANSWER: B",
        }
        result = strategy.vote(stage1, chairman_answer="B")
        assert result == "B"

    def test_tie_without_tiebreaker(self):
        """Test tie resolution without tiebreaker."""
        strategy = MajorityVoteStrategy()
        stage1 = {
            "model1": "FINAL ANSWER: A",
            "model2": "FINAL ANSWER: A",
            "model3": "FINAL ANSWER: B",
            "model4": "FINAL ANSWER: B",
        }
        result = strategy.vote(stage1)
        # Should return one of the tied answers (deterministic order)
        assert result in ["A", "B"]

    def test_numeric_answers(self):
        """Test with numeric answers."""
        strategy = MajorityVoteStrategy()
        stage1 = {
            "model1": "FINAL ANSWER: 42",
            "model2": "FINAL ANSWER: 42",
            "model3": "FINAL ANSWER: 43",
            "model4": "FINAL ANSWER: 42",
        }
        result = strategy.vote(stage1)
        assert result == "42"

    def test_no_valid_answers(self):
        """Test when no answers can be extracted."""
        strategy = MajorityVoteStrategy()
        stage1 = {
            "model1": "I don't know the answer",
            "model2": "This is confusing",
        }
        result = strategy.vote(stage1, chairman_answer="fallback")
        assert result == "fallback"

    def test_empty_responses(self):
        """Test with empty responses dict."""
        strategy = MajorityVoteStrategy()
        result = strategy.vote({}, chairman_answer="default")
        assert result == "default"

    def test_extract_answers(self):
        """Test answer extraction helper."""
        strategy = MajorityVoteStrategy()
        stage1 = {
            "model1": "FINAL ANSWER: A",
            "model2": "No valid answer here",
            "model3": "FINAL ANSWER: B",
        }
        extracted = strategy.extract_answers(stage1)
        assert extracted["model1"] == "A"
        assert extracted["model2"] is None
        assert extracted["model3"] == "B"

    def test_get_valid_answers(self):
        """Test valid answer filtering helper."""
        strategy = MajorityVoteStrategy()
        stage1 = {
            "model1": "FINAL ANSWER: A",
            "model2": "No valid answer",
            "model3": "FINAL ANSWER: B",
        }
        answers, models = strategy.get_valid_answers(stage1)
        assert len(answers) == 2
        assert len(models) == 2
        assert "A" in answers
        assert "B" in answers


class TestWeightedMajorityVoteStrategy:
    """Tests for WeightedMajorityVoteStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = WeightedMajorityVoteStrategy()
        assert strategy.name == "Weighted Majority Vote"
        assert strategy.short_name == "WMV"

    def test_equal_weights_matches_majority(self):
        """Test that equal weights produces same result as majority vote."""
        mv_strategy = MajorityVoteStrategy()
        wv_strategy = WeightedMajorityVoteStrategy(
            weights={"m1": 1.0, "m2": 1.0, "m3": 1.0, "m4": 1.0}
        )

        stage1 = {
            "m1": "FINAL ANSWER: A",
            "m2": "FINAL ANSWER: A",
            "m3": "FINAL ANSWER: B",
            "m4": "FINAL ANSWER: B",
        }

        mv_result = mv_strategy.vote(stage1, chairman_answer="A")
        wv_result = wv_strategy.vote(stage1, chairman_answer="A")
        assert mv_result == wv_result

    def test_high_weight_wins(self):
        """Test that high-weight model's vote wins."""
        strategy = WeightedMajorityVoteStrategy(
            weights={"expert": 10.0, "m1": 1.0, "m2": 1.0, "m3": 1.0}
        )

        stage1 = {
            "expert": "FINAL ANSWER: A",
            "m1": "FINAL ANSWER: B",
            "m2": "FINAL ANSWER: B",
            "m3": "FINAL ANSWER: B",
        }

        result = strategy.vote(stage1)
        # Expert with weight 10 beats 3 models with weight 1 each
        assert result == "A"

    def test_default_weight(self):
        """Test that missing models get default weight of 1.0."""
        strategy = WeightedMajorityVoteStrategy(
            weights={"m1": 5.0}  # Only m1 has explicit weight
        )

        stage1 = {
            "m1": "FINAL ANSWER: A",
            "m2": "FINAL ANSWER: B",  # Gets default 1.0
            "m3": "FINAL ANSWER: B",  # Gets default 1.0
        }

        result = strategy.vote(stage1)
        # m1 (5.0) beats m2+m3 (2.0)
        assert result == "A"

    def test_load_weights_from_file(self):
        """Test loading weights from JSON file."""
        weights = {"m1": 0.9, "m2": 0.7}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(weights, f)
            weights_path = f.name

        try:
            strategy = WeightedMajorityVoteStrategy(weights_file=weights_path)
            assert strategy.weights == weights
        finally:
            Path(weights_path).unlink()

    def test_missing_weights_file(self):
        """Test handling of missing weights file."""
        strategy = WeightedMajorityVoteStrategy(
            weights_file="/nonexistent/path.json"
        )
        # Should use empty dict (all models get default 1.0)
        assert strategy.weights == {}

    def test_get_vote_details(self):
        """Test vote details for analysis."""
        strategy = WeightedMajorityVoteStrategy(
            weights={"m1": 0.8, "m2": 0.6}
        )

        stage1 = {
            "m1": "FINAL ANSWER: A",
            "m2": "FINAL ANSWER: A",
            "m3": "FINAL ANSWER: B",
        }

        details = strategy.get_vote_details(stage1)

        assert "extracted_answers" in details
        assert details["extracted_answers"]["m1"] == "A"

        assert "weights_used" in details
        assert details["weights_used"]["m1"] == 0.8
        assert details["weights_used"]["m2"] == 0.6
        assert details["weights_used"]["m3"] == 1.0  # default

        assert "vote_totals" in details
        # A: 0.8 + 0.6 = 1.4, B: 1.0
        assert details["vote_totals"]["A"] == pytest.approx(1.4)
        assert details["vote_totals"]["B"] == pytest.approx(1.0)

    def test_tiebreaker_with_weights(self):
        """Test tiebreaker when weighted votes are equal."""
        strategy = WeightedMajorityVoteStrategy(
            weights={"m1": 1.0, "m2": 1.0}
        )

        stage1 = {
            "m1": "FINAL ANSWER: A",
            "m2": "FINAL ANSWER: B",
        }

        result = strategy.vote(stage1, chairman_answer="B")
        assert result == "B"


class TestOracleWeightedVoteStrategy:
    """Tests for OracleWeightedVoteStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = OracleWeightedVoteStrategy()
        assert strategy.name == "Oracle Weighted Vote"
        assert strategy.short_name == "Oracle"

    def test_oracle_uses_correct_models(self):
        """Test that oracle weights correct models higher."""
        correctness = {
            "q1": {"m1": True, "m2": False, "m3": True, "m4": False}
        }
        strategy = OracleWeightedVoteStrategy(correctness_map=correctness)
        strategy.set_question("q1")

        stage1 = {
            "m1": "FINAL ANSWER: A",  # correct model
            "m2": "FINAL ANSWER: B",  # incorrect model
            "m3": "FINAL ANSWER: A",  # correct model
            "m4": "FINAL ANSWER: B",  # incorrect model
        }

        result = strategy.vote(stage1)
        # Correct models (m1, m3) vote A with weight 1.0 each
        # Incorrect models (m2, m4) vote B with weight 0.0 each
        assert result == "A"

    def test_oracle_without_data_uses_equal(self):
        """Test fallback to equal weights without oracle data."""
        strategy = OracleWeightedVoteStrategy()
        strategy.set_question("unknown_question")

        stage1 = {
            "m1": "FINAL ANSWER: A",
            "m2": "FINAL ANSWER: A",
            "m3": "FINAL ANSWER: B",
        }

        result = strategy.vote(stage1)
        # Should act like majority vote
        assert result == "A"


class TestCreateVotingStrategies:
    """Tests for create_voting_strategies helper."""

    def test_creates_standard_strategies(self):
        """Test creating standard set of strategies."""
        strategies = create_voting_strategies()
        assert len(strategies) == 2

        names = [s.name for s in strategies]
        assert "Majority Vote" in names
        assert "Weighted Majority Vote" in names

    def test_includes_oracle_when_requested(self):
        """Test including oracle strategy."""
        strategies = create_voting_strategies(include_oracle=True)
        assert len(strategies) == 3

        names = [s.name for s in strategies]
        assert "Oracle Weighted Vote" in names

    def test_passes_weights_file(self):
        """Test that weights file is passed to weighted strategy."""
        weights = {"m1": 0.9}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(weights, f)
            weights_path = f.name

        try:
            strategies = create_voting_strategies(weights_file=weights_path)
            weighted = next(
                s for s in strategies
                if isinstance(s, WeightedMajorityVoteStrategy)
            )
            assert weighted.weights == weights
        finally:
            Path(weights_path).unlink()


class TestVotingStrategyEquivalence:
    """Tests verifying equivalence between weighted and unweighted with equal weights."""

    @pytest.fixture
    def equal_weight_strategies(self):
        """Create MV and WMV with equal weights."""
        models = ["m1", "m2", "m3", "m4"]
        return (
            MajorityVoteStrategy(),
            WeightedMajorityVoteStrategy(
                weights={m: 1.0 for m in models}
            ),
        )

    def test_equivalence_clear_winner(self, equal_weight_strategies):
        """Test equivalence with clear winner."""
        mv, wv = equal_weight_strategies
        stage1 = {
            "m1": "FINAL ANSWER: A",
            "m2": "FINAL ANSWER: A",
            "m3": "FINAL ANSWER: A",
            "m4": "FINAL ANSWER: B",
        }
        assert mv.vote(stage1) == wv.vote(stage1)

    def test_equivalence_tie_with_tiebreaker(self, equal_weight_strategies):
        """Test equivalence with tie and tiebreaker."""
        mv, wv = equal_weight_strategies
        stage1 = {
            "m1": "FINAL ANSWER: A",
            "m2": "FINAL ANSWER: A",
            "m3": "FINAL ANSWER: B",
            "m4": "FINAL ANSWER: B",
        }
        assert mv.vote(stage1, "A") == wv.vote(stage1, "A")
        assert mv.vote(stage1, "B") == wv.vote(stage1, "B")

    def test_equivalence_numeric(self, equal_weight_strategies):
        """Test equivalence with numeric answers."""
        mv, wv = equal_weight_strategies
        stage1 = {
            "m1": "FINAL ANSWER: 42",
            "m2": "FINAL ANSWER: 42",
            "m3": "FINAL ANSWER: 99",
            "m4": "FINAL ANSWER: 99",
        }
        assert mv.vote(stage1, "42") == wv.vote(stage1, "42")

    def test_equivalence_with_none_answers(self, equal_weight_strategies):
        """Test equivalence when some answers can't be extracted."""
        mv, wv = equal_weight_strategies
        stage1 = {
            "m1": "FINAL ANSWER: A",
            "m2": "No answer here",
            "m3": "FINAL ANSWER: A",
            "m4": "FINAL ANSWER: B",
        }
        assert mv.vote(stage1) == wv.vote(stage1)
