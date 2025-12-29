"""Tests for governance utility functions."""

import pytest

from backend.governance.utils import (
    extract_final_answer,
    extract_final_answer_with_fallback,
    majority_vote,
    majority_vote_normalized,
    normalize_answer,
    smart_majority_vote,
)


class TestExtractFinalAnswer:
    """Tests for extract_final_answer function."""

    def test_extract_final_answer_simple(self):
        """Test extracting a simple final answer."""
        response = "Let me think... The answer is 4. FINAL ANSWER: 4"
        assert extract_final_answer(response) == "4"

    def test_extract_final_answer_not_found(self):
        """Test when no final answer pattern is found."""
        response = "The answer is 4."
        assert extract_final_answer(response) is None

    def test_extract_final_answer_multiword(self):
        """Test extracting a multi-word answer."""
        response = "After analysis. FINAL ANSWER: The answer is forty-two"
        assert extract_final_answer(response) == "The answer is forty-two"

    def test_extract_final_answer_case_insensitive(self):
        """Test that matching is case insensitive."""
        response = "final answer: yes"
        assert extract_final_answer(response) == "yes"

    def test_extract_final_answer_with_newline(self):
        """Test answer extraction stops at newline."""
        response = "FINAL ANSWER: 42\nSome additional text"
        assert extract_final_answer(response) == "42"

    def test_extract_final_answer_at_end(self):
        """Test answer at end of response."""
        response = "The computation shows FINAL ANSWER: 100"
        assert extract_final_answer(response) == "100"

    def test_extract_final_answer_strips_whitespace(self):
        """Test that whitespace is stripped from answer."""
        response = "FINAL ANSWER:   spaced answer   "
        assert extract_final_answer(response) == "spaced answer"

    def test_extract_final_answer_empty_response(self):
        """Test with empty response."""
        assert extract_final_answer("") is None


class TestExtractFinalAnswerWithFallback:
    """Tests for extract_final_answer_with_fallback function."""

    def test_with_final_answer_pattern(self):
        """Test that it extracts when pattern is present."""
        response = "Let me think. FINAL ANSWER: 42"
        assert extract_final_answer_with_fallback(response) == "42"

    def test_fallback_to_last_sentence(self):
        """Test fallback to last sentence when pattern not found."""
        response = "First sentence. Second sentence. The answer is 42"
        assert extract_final_answer_with_fallback(response) == "The answer is 42"

    def test_fallback_single_sentence(self):
        """Test fallback with single sentence."""
        response = "The answer is definitely 42"
        assert extract_final_answer_with_fallback(response) == "The answer is definitely 42"

    def test_empty_response(self):
        """Test with empty response."""
        assert extract_final_answer_with_fallback("") == ""

    def test_whitespace_only(self):
        """Test with whitespace only."""
        assert extract_final_answer_with_fallback("   ") == ""


class TestMajorityVote:
    """Tests for majority_vote function."""

    def test_clear_winner(self):
        """Test with a clear majority winner."""
        assert majority_vote(["A", "A", "B", "A"]) == "A"

    def test_tie_with_tiebreaker(self):
        """Test tie resolution with tiebreaker."""
        assert majority_vote(["A", "A", "B", "B"], tiebreaker="B") == "B"

    def test_tie_without_tiebreaker(self):
        """Test tie without tiebreaker returns first winner."""
        result = majority_vote(["A", "A", "B", "B"])
        assert result in ["A", "B"]

    def test_single_answer(self):
        """Test with single answer."""
        assert majority_vote(["A"]) == "A"

    def test_all_same(self):
        """Test when all answers are the same."""
        assert majority_vote(["X", "X", "X"]) == "X"

    def test_tiebreaker_not_in_winners(self):
        """Test tiebreaker that's not among winners."""
        result = majority_vote(["A", "A", "B", "B"], tiebreaker="C")
        assert result in ["A", "B"]

    def test_empty_list(self):
        """Test with empty list."""
        assert majority_vote([]) == ""

    def test_three_way_tie(self):
        """Test three-way tie."""
        result = majority_vote(["A", "B", "C"])
        assert result in ["A", "B", "C"]


class TestNormalizeAnswer:
    """Tests for normalize_answer function."""

    def test_lowercase(self):
        """Test that answer is lowercased."""
        assert normalize_answer("YES") == "yes"

    def test_strip_whitespace(self):
        """Test that whitespace is stripped."""
        assert normalize_answer("  answer  ") == "answer"

    def test_remove_trailing_punctuation(self):
        """Test removal of trailing punctuation."""
        assert normalize_answer("answer.") == "answer"
        assert normalize_answer("answer!") == "answer"
        assert normalize_answer("answer?") == "answer"

    def test_multiple_trailing_punctuation(self):
        """Test removal of multiple trailing punctuation."""
        assert normalize_answer("answer...") == "answer"

    def test_preserves_internal_punctuation(self):
        """Test that internal punctuation is preserved."""
        assert normalize_answer("it's fine") == "it's fine"

    def test_empty_string(self):
        """Test with empty string."""
        assert normalize_answer("") == ""

    def test_none_like_behavior(self):
        """Test with empty/falsy input."""
        assert normalize_answer("") == ""


class TestMajorityVoteNormalized:
    """Tests for majority_vote_normalized function."""

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        result = majority_vote_normalized(["Yes", "YES", "no"])
        assert result.lower() == "yes"

    def test_punctuation_normalized(self):
        """Test that punctuation is normalized for matching."""
        result = majority_vote_normalized(["42", "42.", "41"])
        assert result in ["42", "42."]

    def test_returns_original_form(self):
        """Test that original form is returned, not normalized."""
        result = majority_vote_normalized(["YES", "Yes", "no"])
        # Should return one of the original forms
        assert result in ["YES", "Yes"]

    def test_with_tiebreaker(self):
        """Test tiebreaker with normalized matching."""
        result = majority_vote_normalized(["A", "a", "B", "b"], tiebreaker="B")
        assert result.lower() == "b"

    def test_empty_list(self):
        """Test with empty list."""
        assert majority_vote_normalized([]) == ""

    def test_whitespace_normalized(self):
        """Test that whitespace is normalized."""
        result = majority_vote_normalized(["answer", " answer ", "other"])
        assert result.strip() == "answer"


class TestUtilsImports:
    """Test that utils can be imported from governance package."""

    def test_import_extract_final_answer(self):
        """Test importing extract_final_answer from package."""
        from backend.governance import extract_final_answer

        assert callable(extract_final_answer)

    def test_import_majority_vote(self):
        """Test importing majority_vote from package."""
        from backend.governance import majority_vote

        assert callable(majority_vote)

    def test_import_all_utils(self):
        """Test importing all utility functions from package."""
        from backend.governance import (
            extract_final_answer,
            extract_final_answer_with_fallback,
            majority_vote,
            majority_vote_normalized,
            normalize_answer,
        )

        assert all(
            callable(f)
            for f in [
                extract_final_answer,
                extract_final_answer_with_fallback,
                majority_vote,
                majority_vote_normalized,
                normalize_answer,
            ]
        )


class TestSmartMajorityVote:
    """Tests for smart_majority_vote function."""

    def test_numeric_answers_normalized(self):
        """Test that numeric answers are compared by value."""
        # "42" and "42.0" should be treated as the same vote
        result = smart_majority_vote(["42", "42.0", "43"])
        assert result in ["42", "42.0"]

    def test_numeric_with_dollar_sign(self):
        """Test that dollar signs are handled in numeric answers."""
        result = smart_majority_vote(["$42", "42", "$43"])
        assert result in ["$42", "42"]

    def test_numeric_with_commas(self):
        """Test that commas are handled in numeric answers."""
        result = smart_majority_vote(["1,000", "1000", "2000"])
        assert result in ["1,000", "1000"]

    def test_letter_answers_case_insensitive(self):
        """Test that letter answers are compared case-insensitively."""
        result = smart_majority_vote(["A", "a", "B"])
        assert result.upper() == "A"

    def test_letter_answers_uppercase_result(self):
        """Test that letter voting returns uppercase."""
        result = smart_majority_vote(["a", "a", "b"])
        assert result == "A"

    def test_string_answers_normalized(self):
        """Test that string answers are compared with normalization."""
        result = smart_majority_vote(["Yes", "YES", "no"])
        assert result.lower() == "yes"

    def test_empty_list(self):
        """Test with empty list."""
        assert smart_majority_vote([]) == ""

    def test_single_answer(self):
        """Test with single answer."""
        assert smart_majority_vote(["42"]) == "42"

    def test_tiebreaker_numeric(self):
        """Test tiebreaker with numeric answers."""
        result = smart_majority_vote(["42", "43"], tiebreaker="43")
        assert result == "43"

    def test_tiebreaker_letter(self):
        """Test tiebreaker with letter answers."""
        result = smart_majority_vote(["A", "B"], tiebreaker="B")
        assert result == "B"

    def test_tiebreaker_string(self):
        """Test tiebreaker with string answers."""
        result = smart_majority_vote(["yes", "no"], tiebreaker="no")
        assert result.lower() == "no"

    def test_mixed_numeric_and_text_falls_back(self):
        """Test that mixed types fall back to string comparison."""
        # "42" parses as numeric, but "forty-two" doesn't
        result = smart_majority_vote(["42", "42", "forty-two"])
        # Should fall back to normalized string matching
        assert result == "42"


class TestComputeVoteMetadata:
    """Tests for compute_vote_metadata function."""

    def test_basic_metadata(self):
        """Test basic vote metadata computation."""
        from backend.governance.utils import compute_vote_metadata

        extracted = {"m1": "4", "m2": "4", "m3": "5"}
        winner, metadata = compute_vote_metadata(extracted)

        assert winner == "4"
        assert metadata["raw_answers"] == {"m1": "4", "m2": "4", "m3": "5"}
        assert metadata["normalized_answers"] == {"m1": "4", "m2": "4", "m3": "5"}
        assert metadata["vote_counts"]["4"] == 2
        assert metadata["vote_counts"]["5"] == 1
        assert metadata["is_tie"] is False
        assert metadata["tiebreaker_used"] is False

    def test_tie_detected(self):
        """Test that ties are correctly detected."""
        from backend.governance.utils import compute_vote_metadata

        extracted = {"m1": "A", "m2": "B"}
        winner, metadata = compute_vote_metadata(extracted)

        assert metadata["is_tie"] is True
        assert metadata["vote_counts"]["a"] == 1
        assert metadata["vote_counts"]["b"] == 1

    def test_tiebreaker_used(self):
        """Test that tiebreaker usage is tracked."""
        from backend.governance.utils import compute_vote_metadata

        extracted = {"m1": "A", "m2": "B"}
        winner, metadata = compute_vote_metadata(extracted, tiebreaker="B")

        assert metadata["is_tie"] is True
        assert metadata["tiebreaker_used"] is True
        assert winner == "B"

    def test_normalization_applied(self):
        """Test that '4', '4.', and ' 4' normalize identically."""
        from backend.governance.utils import compute_vote_metadata

        extracted = {"m1": "4", "m2": "4.", "m3": " 4"}
        winner, metadata = compute_vote_metadata(extracted)

        # All should normalize to "4"
        assert metadata["vote_counts"]["4"] == 3
        assert metadata["is_tie"] is False

    def test_none_answers_handled(self):
        """Test handling of None answers."""
        from backend.governance.utils import compute_vote_metadata

        extracted = {"m1": "4", "m2": None, "m3": "4"}
        winner, metadata = compute_vote_metadata(extracted)

        assert winner == "4"
        assert metadata["raw_answers"]["m2"] is None
        assert metadata["normalized_answers"]["m2"] is None
        assert metadata["vote_counts"]["4"] == 2

    def test_empty_extracted_answers(self):
        """Test handling of all None answers."""
        from backend.governance.utils import compute_vote_metadata

        extracted = {"m1": None, "m2": None}
        winner, metadata = compute_vote_metadata(extracted)

        assert winner == ""
        assert metadata["vote_counts"] == {}
        assert metadata["is_tie"] is False


class TestComputeWeightedVoteMetadata:
    """Tests for compute_weighted_vote_metadata function."""

    def test_basic_weighted_metadata(self):
        """Test basic weighted vote metadata computation."""
        from backend.governance.utils import compute_weighted_vote_metadata

        extracted = {"m1": "4", "m2": "4", "m3": "5"}
        weights = {"m1": 1.0, "m2": 1.0, "m3": 1.0}
        winner, metadata = compute_weighted_vote_metadata(extracted, weights)

        assert winner == "4"
        assert metadata["vote_counts"]["4"] == 2.0
        assert metadata["vote_counts"]["5"] == 1.0
        assert metadata["is_tie"] is False

    def test_high_weight_wins(self):
        """Test that higher weight can overcome count."""
        from backend.governance.utils import compute_weighted_vote_metadata

        extracted = {"m1": "4", "m2": "4", "m3": "5"}
        weights = {"m1": 1.0, "m2": 1.0, "m3": 10.0}  # m3 has high weight
        winner, metadata = compute_weighted_vote_metadata(extracted, weights)

        assert winner == "5"
        assert metadata["vote_counts"]["4"] == 2.0
        assert metadata["vote_counts"]["5"] == 10.0

    def test_weighted_tie(self):
        """Test weighted tie detection."""
        from backend.governance.utils import compute_weighted_vote_metadata

        extracted = {"m1": "A", "m2": "B"}
        weights = {"m1": 1.0, "m2": 1.0}
        winner, metadata = compute_weighted_vote_metadata(extracted, weights)

        assert metadata["is_tie"] is True
        assert metadata["vote_counts"]["a"] == 1.0
        assert metadata["vote_counts"]["b"] == 1.0
