"""Tests for deliberation dynamics analysis module."""

import pandas as pd
import pytest

from experiments.analyze_deliberation_dynamics import (
    analyze_deliberation_dynamics,
    analyze_single_trial,
    compute_influence_matrix,
    compute_pairwise_agreement,
    extract_stage_answers,
    generate_dynamics_summary,
)


class TestExtractStageAnswers:
    """Tests for extract_stage_answers function."""

    def test_extracts_both_stages(self):
        """Test extraction from stage1 and stage2 responses."""
        stage1 = {
            "model1": "I think 42. FINAL ANSWER: 42",
            "model2": "My answer is 7. FINAL ANSWER: 7",
        }
        stage2 = {
            "deliberation_responses": {
                "model1": "After discussion, FINAL ANSWER: 7",
                "model2": "I stick with FINAL ANSWER: 7",
            }
        }

        s1, s2 = extract_stage_answers(stage1, stage2)

        assert s1["model1"] == "42"
        assert s1["model2"] == "7"
        assert s2["model1"] == "7"
        assert s2["model2"] == "7"

    def test_handles_missing_deliberation(self):
        """Test when deliberation_responses is missing."""
        stage1 = {"model1": "FINAL ANSWER: 42"}
        stage2 = {}

        s1, s2 = extract_stage_answers(stage1, stage2)

        assert s1["model1"] == "42"
        assert s2["model1"] is None

    def test_normalizes_answers(self):
        """Test normalization of answers."""
        stage1 = {"model1": "FINAL ANSWER: A", "model2": "FINAL ANSWER: b"}
        stage2 = {
            "deliberation_responses": {
                "model1": "FINAL ANSWER: B",
                "model2": "FINAL ANSWER: B",
            }
        }

        s1, s2 = extract_stage_answers(stage1, stage2)

        # normalize_answer lowercases everything
        assert s1["model1"] == "a"
        assert s1["model2"] == "b"
        assert s2["model1"] == "b"
        assert s2["model2"] == "b"


class TestComputePairwiseAgreement:
    """Tests for compute_pairwise_agreement function."""

    def test_full_agreement(self):
        """Test when all models agree."""
        answers = {"m1": "42", "m2": "42", "m3": "42"}
        assert compute_pairwise_agreement(answers) == 1.0

    def test_no_agreement(self):
        """Test when no models agree."""
        answers = {"m1": "1", "m2": "2", "m3": "3"}
        assert compute_pairwise_agreement(answers) == 0.0

    def test_partial_agreement(self):
        """Test partial agreement (2 of 3 agree)."""
        answers = {"m1": "42", "m2": "42", "m3": "7"}
        # 3 pairs: (m1,m2) agree, (m1,m3) disagree, (m2,m3) disagree
        # 1/3 = 0.333...
        assert abs(compute_pairwise_agreement(answers) - 1 / 3) < 0.01

    def test_handles_none_values(self):
        """Test that None values are excluded."""
        answers = {"m1": "42", "m2": None, "m3": "42"}
        # Only m1 and m3 count, they agree
        assert compute_pairwise_agreement(answers) == 1.0

    def test_single_answer(self):
        """Test with only one valid answer."""
        answers = {"m1": "42", "m2": None}
        assert compute_pairwise_agreement(answers) == 0.0

    def test_empty_answers(self):
        """Test with empty answers."""
        assert compute_pairwise_agreement({}) == 0.0


class TestAnalyzeSingleTrial:
    """Tests for analyze_single_trial function."""

    def test_detects_mind_change(self):
        """Test detection of mind changes."""
        stage1 = {"m1": "42", "m2": "7"}
        stage2 = {"m1": "7", "m2": "7"}

        result = analyze_single_trial(stage1, stage2, "42")

        assert result["model_metrics"]["m1"]["changed_mind"] is True
        assert result["model_metrics"]["m2"]["changed_mind"] is False

    def test_detects_changed_to_someone_else(self):
        """Test detection of adopting another's answer."""
        stage1 = {"m1": "42", "m2": "7", "m3": "7"}
        stage2 = {"m1": "7", "m2": "7", "m3": "7"}

        result = analyze_single_trial(stage1, stage2, "42")

        metrics = result["model_metrics"]["m1"]
        assert metrics["changed_to_someone_else"] is True
        assert "m2" in metrics["influence_sources"] or "m3" in metrics["influence_sources"]

    def test_detects_fixed(self):
        """Test detection of fixing wrong answer."""
        stage1 = {"m1": "wrong"}
        stage2 = {"m1": "correct"}

        result = analyze_single_trial(stage1, stage2, "correct")

        assert result["model_metrics"]["m1"]["fixed"] is True
        assert result["model_metrics"]["m1"]["broke"] is False
        assert result["fixed"] == 1
        assert result["broke"] == 0

    def test_detects_broke(self):
        """Test detection of breaking correct answer."""
        stage1 = {"m1": "correct"}
        stage2 = {"m1": "wrong"}

        result = analyze_single_trial(stage1, stage2, "correct")

        assert result["model_metrics"]["m1"]["fixed"] is False
        assert result["model_metrics"]["m1"]["broke"] is True
        assert result["fixed"] == 0
        assert result["broke"] == 1

    def test_computes_agreement_change(self):
        """Test agreement pre/post computation."""
        stage1 = {"m1": "42", "m2": "7", "m3": "7"}
        stage2 = {"m1": "7", "m2": "7", "m3": "7"}

        result = analyze_single_trial(stage1, stage2, None)

        # Pre: 2/3 pairs agree (m2-m3)
        # Post: all agree
        assert result["agreement_pre"] < result["agreement_post"]
        assert result["agreement_post"] == 1.0


class TestComputeInfluenceMatrix:
    """Tests for compute_influence_matrix function."""

    def test_tracks_influence(self):
        """Test influence tracking."""
        trial1 = {
            "model_metrics": {
                "m1": {
                    "changed_mind": True,
                    "changed_to_someone_else": True,
                    "influence_sources": ["m2"],
                },
                "m2": {
                    "changed_mind": False,
                    "changed_to_someone_else": False,
                    "influence_sources": [],
                },
            }
        }

        matrix = compute_influence_matrix([trial1])

        assert matrix.loc["m2", "m1"] == 1.0
        assert matrix.loc["m1", "m2"] == 0.0

    def test_splits_credit_evenly(self):
        """Test credit splitting when multiple sources."""
        trial = {
            "model_metrics": {
                "m1": {
                    "changed_mind": True,
                    "changed_to_someone_else": True,
                    "influence_sources": ["m2", "m3"],
                },
            }
        }

        matrix = compute_influence_matrix([trial])

        assert matrix.loc["m2", "m1"] == 0.5
        assert matrix.loc["m3", "m1"] == 0.5

    def test_accumulates_across_trials(self):
        """Test accumulation across multiple trials."""
        trials = [
            {
                "model_metrics": {
                    "m1": {
                        "changed_to_someone_else": True,
                        "influence_sources": ["m2"],
                    },
                }
            },
            {
                "model_metrics": {
                    "m1": {
                        "changed_to_someone_else": True,
                        "influence_sources": ["m2"],
                    },
                }
            },
        ]

        matrix = compute_influence_matrix(trials)

        assert matrix.loc["m2", "m1"] == 2.0

    def test_empty_trials(self):
        """Test with no trials."""
        matrix = compute_influence_matrix([])
        assert matrix.empty


class TestAnalyzeDeliberationDynamics:
    """Tests for the main analyze_deliberation_dynamics function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with deliberation data."""
        return pd.DataFrame([
            {
                "structure": "Deliberate → Vote",
                "benchmark": "GSM8K",
                "question_id": "q1",
                "replication": 0,
                "expected": "42",
                "stage1_responses": {
                    "m1": "FINAL ANSWER: 7",
                    "m2": "FINAL ANSWER: 42",
                },
                "stage2_data": {
                    "deliberation_responses": {
                        "m1": "FINAL ANSWER: 42",
                        "m2": "FINAL ANSWER: 42",
                    }
                },
            },
            {
                "structure": "Deliberate → Vote",
                "benchmark": "GSM8K",
                "question_id": "q2",
                "replication": 0,
                "expected": "10",
                "stage1_responses": {
                    "m1": "FINAL ANSWER: 10",
                    "m2": "FINAL ANSWER: 10",
                },
                "stage2_data": {
                    "deliberation_responses": {
                        "m1": "FINAL ANSWER: 5",
                        "m2": "FINAL ANSWER: 10",
                    }
                },
            },
        ])

    def test_analyzes_deliberation_trials(self, sample_df):
        """Test analysis of deliberation trials."""
        result = analyze_deliberation_dynamics(sample_df)

        assert result["n_trials"] == 2
        assert result["total_mind_changes"] == 2  # m1 changed in both

    def test_counts_fixed_broke(self, sample_df):
        """Test fixed/broke counting."""
        result = analyze_deliberation_dynamics(sample_df)

        # q1: m1 was wrong (7) → correct (42) = fixed
        # q2: m1 was correct (10) → wrong (5) = broke
        assert result["total_fixed"] == 1
        assert result["total_broke"] == 1
        assert result["net_benefit"] == 0

    def test_computes_influence(self, sample_df):
        """Test influence matrix computation."""
        result = analyze_deliberation_dynamics(sample_df)

        # m2 influenced m1 in q1 (m1 adopted m2's answer 42)
        matrix = result["influence_matrix"]
        assert matrix.loc["m2", "m1"] >= 1.0

    def test_filters_non_deliberation(self):
        """Test that non-deliberation structures are filtered."""
        df = pd.DataFrame([
            {
                "structure": "Majority Vote",
                "benchmark": "GSM8K",
                "question_id": "q1",
                "stage1_responses": {"m1": "FINAL ANSWER: 42"},
                "stage2_data": {},
            }
        ])

        result = analyze_deliberation_dynamics(df)

        assert result == {}

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        result = analyze_deliberation_dynamics(pd.DataFrame())
        assert result == {}

    def test_generates_output_rows(self, sample_df):
        """Test that output data structures are generated."""
        result = analyze_deliberation_dynamics(sample_df)

        assert len(result["mind_change_rows"]) > 0
        assert len(result["question_level_data"]) == 2

    def test_saves_outputs_to_dir(self, sample_df, tmp_path):
        """Test saving outputs to directory."""
        result = analyze_deliberation_dynamics(sample_df, output_dir=str(tmp_path))

        assert (tmp_path / "analysis_mind_change.csv").exists()
        assert (tmp_path / "analysis_influence_matrix.csv").exists()
        assert (tmp_path / "analysis_question_level_changes.jsonl").exists()


class TestGenerateDynamicsSummary:
    """Tests for generate_dynamics_summary function."""

    def test_generates_summary(self):
        """Test summary generation."""
        analysis = {
            "n_trials": 10,
            "avg_agreement_pre": 0.5,
            "avg_agreement_post": 0.8,
            "agreement_increase": 0.3,
            "total_mind_changes": 15,
            "total_fixed": 8,
            "total_broke": 3,
            "net_benefit": 5,
            "most_influential": {"m1": 5.0, "m2": 3.0},
            "most_influenced": {"m3": 4.0, "m1": 2.0},
        }

        summary = generate_dynamics_summary(analysis)

        assert "DELIBERATION DYNAMICS" in summary
        assert "Trials analyzed: 10" in summary
        assert "Net benefit: +5" in summary

    def test_handles_empty_analysis(self):
        """Test with empty analysis."""
        summary = generate_dynamics_summary({})
        assert "No deliberation data" in summary
