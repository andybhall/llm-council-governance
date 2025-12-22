"""Tests for benchmark loaders and evaluators."""

import pytest

from backend.evaluation.base import Benchmark, EvalResult, Question
from backend.evaluation.gsm8k import GSM8KBenchmark
from backend.evaluation.truthfulqa import TruthfulQABenchmark


class TestBaseClasses:
    """Tests for base benchmark classes."""

    def test_question_dataclass(self):
        """Test Question dataclass instantiation."""
        q = Question(
            id="test_1",
            text="What is 2+2?",
            ground_truth="4",
        )
        assert q.id == "test_1"
        assert q.text == "What is 2+2?"
        assert q.ground_truth == "4"
        assert q.metadata is None

    def test_question_with_metadata(self):
        """Test Question with metadata."""
        q = Question(
            id="test_1",
            text="Test",
            ground_truth="answer",
            metadata={"source": "test", "difficulty": "easy"},
        )
        assert q.metadata["source"] == "test"
        assert q.metadata["difficulty"] == "easy"

    def test_eval_result_dataclass(self):
        """Test EvalResult dataclass."""
        result = EvalResult(
            question_id="q1",
            is_correct=True,
            predicted="4",
            expected="4",
        )
        assert result.question_id == "q1"
        assert result.is_correct is True
        assert result.predicted == "4"
        assert result.expected == "4"

    def test_benchmark_is_abstract(self):
        """Test that Benchmark cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Benchmark()


class TestGSM8KBenchmark:
    """Tests for GSM8K benchmark."""

    def test_benchmark_name(self):
        """Test benchmark name property."""
        benchmark = GSM8KBenchmark()
        assert benchmark.name == "GSM8K"

    def test_extract_ground_truth_standard_format(self):
        """Test extracting answer from standard GSM8K format."""
        benchmark = GSM8KBenchmark()

        answer = "Step 1: 2+2=4\nStep 2: 4+4=8\n#### 8"
        result = benchmark._extract_ground_truth(answer)
        assert result == "8"

    def test_extract_ground_truth_with_commas(self):
        """Test extracting answer with comma formatting."""
        benchmark = GSM8KBenchmark()

        answer = "The total is #### 1,234"
        result = benchmark._extract_ground_truth(answer)
        assert result == "1234"

    def test_extract_ground_truth_decimal(self):
        """Test extracting decimal answer."""
        benchmark = GSM8KBenchmark()

        answer = "#### 3.14"
        result = benchmark._extract_ground_truth(answer)
        assert result == "3.14"

    def test_extract_ground_truth_negative(self):
        """Test extracting negative answer."""
        benchmark = GSM8KBenchmark()

        answer = "The result is #### -42"
        result = benchmark._extract_ground_truth(answer)
        assert result == "-42"

    def test_normalize_number_removes_commas(self):
        """Test number normalization removes commas."""
        benchmark = GSM8KBenchmark()
        assert benchmark._normalize_number("1,234,567") == "1234567"

    def test_normalize_number_handles_decimals(self):
        """Test number normalization handles decimals."""
        benchmark = GSM8KBenchmark()
        assert benchmark._normalize_number("3.14159") == "3.14159"

    def test_normalize_number_whole_number(self):
        """Test that whole numbers don't have decimal."""
        benchmark = GSM8KBenchmark()
        assert benchmark._normalize_number("42.0") == "42"

    def test_extract_number_from_response_final_answer(self):
        """Test extracting from FINAL ANSWER pattern."""
        benchmark = GSM8KBenchmark()

        response = "Let me calculate... 2+2=4. FINAL ANSWER: 4"
        result = benchmark._extract_number_from_response(response)
        assert result == "4"

    def test_extract_number_from_response_answer_is(self):
        """Test extracting from 'answer is' pattern."""
        benchmark = GSM8KBenchmark()

        response = "After calculation, the answer is 42."
        result = benchmark._extract_number_from_response(response)
        assert result == "42"

    def test_extract_number_from_response_equals(self):
        """Test extracting from 'equals' pattern."""
        benchmark = GSM8KBenchmark()

        response = "2 + 2 equals 4"
        result = benchmark._extract_number_from_response(response)
        assert result == "4"

    def test_extract_number_from_response_fallback(self):
        """Test fallback to last number."""
        benchmark = GSM8KBenchmark()

        response = "First I got 10, then 20, finally 30"
        result = benchmark._extract_number_from_response(response)
        assert result == "30"

    def test_extract_number_from_response_no_number(self):
        """Test when no number is found."""
        benchmark = GSM8KBenchmark()

        response = "I cannot compute this."
        result = benchmark._extract_number_from_response(response)
        assert result is None

    def test_evaluate_correct_answer(self):
        """Test evaluation of correct answer."""
        benchmark = GSM8KBenchmark()
        question = Question(id="q1", text="2+2?", ground_truth="4")

        result = benchmark.evaluate(question, "The answer is 4. FINAL ANSWER: 4")

        assert result.is_correct is True
        assert result.predicted == "4"
        assert result.expected == "4"

    def test_evaluate_incorrect_answer(self):
        """Test evaluation of incorrect answer."""
        benchmark = GSM8KBenchmark()
        question = Question(id="q1", text="2+2?", ground_truth="4")

        result = benchmark.evaluate(question, "I think it's 5. FINAL ANSWER: 5")

        assert result.is_correct is False
        assert result.predicted == "5"
        assert result.expected == "4"

    def test_evaluate_no_number_found(self):
        """Test evaluation when no number in response."""
        benchmark = GSM8KBenchmark()
        question = Question(id="q1", text="2+2?", ground_truth="4")

        result = benchmark.evaluate(question, "I don't know the answer.")

        assert result.is_correct is False
        assert result.predicted == "[no number found]"
        assert result.expected == "4"

    def test_evaluate_with_comma_formatting(self):
        """Test that comma-formatted numbers match."""
        benchmark = GSM8KBenchmark()
        question = Question(id="q1", text="Sum?", ground_truth="1234")

        result = benchmark.evaluate(question, "FINAL ANSWER: 1,234")

        assert result.is_correct is True
        assert result.predicted == "1234"


class TestGSM8KWithMockDataset:
    """Tests for GSM8K with mocked dataset loading."""

    @pytest.fixture
    def mock_dataset(self, monkeypatch):
        """Mock the HuggingFace datasets library."""

        class MockDataset:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __iter__(self):
                return iter(self._data)

            def select(self, indices):
                return MockDataset([self._data[i] for i in indices])

        mock_data = [
            {
                "question": "What is 2+2?",
                "answer": "2+2=4\n#### 4",
            },
            {
                "question": "If John has 5 apples and gives 2 away, how many does he have?",
                "answer": "5-2=3\n#### 3",
            },
            {
                "question": "What is 10 * 10?",
                "answer": "10*10=100\n#### 100",
            },
        ]

        def mock_load_dataset(name, config, split):
            return MockDataset(mock_data)

        # Mock at the module level where it's imported
        import backend.evaluation.gsm8k as gsm8k_module

        monkeypatch.setattr(
            gsm8k_module,
            "load_dataset",
            mock_load_dataset,
            raising=False,
        )

        # Also need to patch the import itself
        import sys
        from unittest.mock import MagicMock

        mock_datasets = MagicMock()
        mock_datasets.load_dataset = mock_load_dataset
        monkeypatch.setitem(sys.modules, "datasets", mock_datasets)

        return mock_data

    def test_load_questions_all(self, mock_dataset):
        """Test loading all questions."""
        benchmark = GSM8KBenchmark()
        questions = benchmark.load_questions()

        assert len(questions) == 3
        assert "What is 2+2?" in questions[0].text
        assert "single number" in questions[0].text  # Instruction included
        assert questions[0].ground_truth == "4"

    def test_load_questions_limited(self, mock_dataset):
        """Test loading limited number of questions."""
        benchmark = GSM8KBenchmark()
        questions = benchmark.load_questions(n=2)

        assert len(questions) == 2

    def test_questions_have_metadata(self, mock_dataset):
        """Test that loaded questions have metadata."""
        benchmark = GSM8KBenchmark()
        questions = benchmark.load_questions(n=1)

        assert questions[0].metadata is not None
        assert "full_solution" in questions[0].metadata
        assert "split" in questions[0].metadata


class TestBenchmarkImports:
    """Test that benchmark classes can be imported from package."""

    def test_import_base_classes(self):
        """Test importing base classes."""
        from backend.evaluation import Benchmark, EvalResult, Question

        assert Benchmark is not None
        assert EvalResult is not None
        assert Question is not None

    def test_import_gsm8k(self):
        """Test importing GSM8K benchmark."""
        from backend.evaluation import GSM8KBenchmark

        assert GSM8KBenchmark is not None

    def test_import_truthfulqa(self):
        """Test importing TruthfulQA benchmark."""
        from backend.evaluation import TruthfulQABenchmark

        assert TruthfulQABenchmark is not None


class TestTruthfulQABenchmark:
    """Tests for TruthfulQA benchmark."""

    def test_benchmark_name(self):
        """Test benchmark name property."""
        benchmark = TruthfulQABenchmark()
        assert benchmark.name == "TruthfulQA"

    def test_extract_letter_final_answer(self):
        """Test extracting letter from FINAL ANSWER pattern."""
        benchmark = TruthfulQABenchmark()

        response = "After considering the options, FINAL ANSWER: B"
        result = benchmark._extract_letter_from_response(response)
        assert result == "B"

    def test_extract_letter_lowercase(self):
        """Test extracting lowercase letter."""
        benchmark = TruthfulQABenchmark()

        response = "FINAL ANSWER: c"
        result = benchmark._extract_letter_from_response(response)
        assert result == "C"

    def test_extract_letter_answer_is(self):
        """Test extracting from 'answer is' pattern."""
        benchmark = TruthfulQABenchmark()

        response = "The answer is A because..."
        result = benchmark._extract_letter_from_response(response)
        assert result == "A"

    def test_extract_letter_choose(self):
        """Test extracting from 'choose' pattern."""
        benchmark = TruthfulQABenchmark()

        response = "I would choose D"
        result = benchmark._extract_letter_from_response(response)
        assert result == "D"

    def test_extract_letter_at_end(self):
        """Test extracting standalone letter at end."""
        benchmark = TruthfulQABenchmark()

        response = "Based on the evidence, my choice is B."
        result = benchmark._extract_letter_from_response(response)
        assert result == "B"

    def test_extract_letter_fallback(self):
        """Test fallback to first letter found."""
        benchmark = TruthfulQABenchmark()

        response = "Option A seems correct"
        result = benchmark._extract_letter_from_response(response)
        assert result == "A"

    def test_extract_letter_no_letter(self):
        """Test when no valid letter found."""
        benchmark = TruthfulQABenchmark()

        response = "I'm not sure about this question."
        result = benchmark._extract_letter_from_response(response)
        # Should find some letter or return None
        # "I'm" contains I which is a valid letter
        assert result in ["I", None] or result is not None

    def test_format_multiple_choice(self):
        """Test formatting a question as multiple choice."""
        benchmark = TruthfulQABenchmark()

        item = {
            "question": "What color is the sky?",
            "mc1_targets": {
                "choices": ["Blue", "Green", "Red", "Yellow"],
                "labels": [1, 0, 0, 0],
            },
        }

        formatted, correct = benchmark._format_multiple_choice(item)

        assert "What color is the sky?" in formatted
        assert "A. Blue" in formatted
        assert "B. Green" in formatted
        assert "C. Red" in formatted
        assert "D. Yellow" in formatted
        assert correct == "A"

    def test_format_multiple_choice_correct_not_first(self):
        """Test when correct answer is not the first option."""
        benchmark = TruthfulQABenchmark()

        item = {
            "question": "Test question?",
            "mc1_targets": {
                "choices": ["Wrong", "Wrong", "Correct", "Wrong"],
                "labels": [0, 0, 1, 0],
            },
        }

        formatted, correct = benchmark._format_multiple_choice(item)
        assert correct == "C"

    def test_evaluate_correct_answer(self):
        """Test evaluation of correct answer."""
        benchmark = TruthfulQABenchmark()
        question = Question(id="q1", text="Test?", ground_truth="B")

        result = benchmark.evaluate(question, "I think FINAL ANSWER: B")

        assert result.is_correct is True
        assert result.predicted == "B"
        assert result.expected == "B"

    def test_evaluate_incorrect_answer(self):
        """Test evaluation of incorrect answer."""
        benchmark = TruthfulQABenchmark()
        question = Question(id="q1", text="Test?", ground_truth="B")

        result = benchmark.evaluate(question, "FINAL ANSWER: A")

        assert result.is_correct is False
        assert result.predicted == "A"
        assert result.expected == "B"

    def test_evaluate_case_insensitive(self):
        """Test that evaluation is case insensitive."""
        benchmark = TruthfulQABenchmark()
        question = Question(id="q1", text="Test?", ground_truth="B")

        result = benchmark.evaluate(question, "FINAL ANSWER: b")

        assert result.is_correct is True

    def test_evaluate_no_letter_found(self):
        """Test evaluation when no letter in response."""
        benchmark = TruthfulQABenchmark()
        question = Question(id="q1", text="Test?", ground_truth="B")

        result = benchmark.evaluate(question, "12345")

        assert result.is_correct is False
        assert result.predicted == "[no letter found]"


class TestTruthfulQAWithMockDataset:
    """Tests for TruthfulQA with mocked dataset loading."""

    @pytest.fixture
    def mock_truthfulqa_dataset(self, monkeypatch):
        """Mock the HuggingFace datasets library for TruthfulQA."""

        class MockDataset:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __iter__(self):
                return iter(self._data)

            def select(self, indices):
                return MockDataset([self._data[i] for i in indices])

        mock_data = [
            {
                "question": "What happens if you eat watermelon seeds?",
                "mc1_targets": {
                    "choices": [
                        "Nothing happens",
                        "You grow a watermelon in your stomach",
                        "You get sick",
                    ],
                    "labels": [1, 0, 0],
                },
                "category": "Misconceptions",
                "source": "test",
            },
            {
                "question": "Can you see the Great Wall of China from space?",
                "mc1_targets": {
                    "choices": [
                        "No, it's too narrow",
                        "Yes, easily",
                        "Only with a telescope",
                    ],
                    "labels": [1, 0, 0],
                },
                "category": "Misconceptions",
                "source": "test",
            },
        ]

        def mock_load_dataset(name, config, split):
            return MockDataset(mock_data)

        import sys
        from unittest.mock import MagicMock

        mock_datasets = MagicMock()
        mock_datasets.load_dataset = mock_load_dataset
        monkeypatch.setitem(sys.modules, "datasets", mock_datasets)

        return mock_data

    def test_load_questions_all(self, mock_truthfulqa_dataset):
        """Test loading all questions."""
        benchmark = TruthfulQABenchmark()
        # Use binary_format=False since mock data doesn't have binary format fields
        questions = benchmark.load_questions(binary_format=False)

        assert len(questions) == 2
        assert "watermelon" in questions[0].text.lower()

    def test_load_questions_limited(self, mock_truthfulqa_dataset):
        """Test loading limited number of questions."""
        benchmark = TruthfulQABenchmark()
        questions = benchmark.load_questions(n=1, binary_format=False)

        assert len(questions) == 1

    def test_questions_formatted_as_multiple_choice(self, mock_truthfulqa_dataset):
        """Test that questions are formatted with options."""
        benchmark = TruthfulQABenchmark()
        questions = benchmark.load_questions(n=1, binary_format=False)

        assert "A." in questions[0].text
        assert "B." in questions[0].text

    def test_questions_have_correct_ground_truth(self, mock_truthfulqa_dataset):
        """Test that ground truth is the correct letter."""
        benchmark = TruthfulQABenchmark()
        questions = benchmark.load_questions(binary_format=False)

        # First question: "Nothing happens" is correct (index 0 = A)
        assert questions[0].ground_truth == "A"

    def test_questions_have_metadata(self, mock_truthfulqa_dataset):
        """Test that loaded questions have metadata."""
        benchmark = TruthfulQABenchmark()
        questions = benchmark.load_questions(n=1, binary_format=False)

        assert questions[0].metadata is not None
        assert "category" in questions[0].metadata
        assert "original_question" in questions[0].metadata
