"""GSM8K benchmark loader and evaluator."""

import re
from typing import List, Optional

from backend.evaluation.base import Benchmark, EvalResult, Question


class GSM8KBenchmark(Benchmark):
    """
    GSM8K: Grade School Math 8K benchmark.

    Tests arithmetic reasoning with grade school math word problems.
    Ground truth answers are numerical.
    """

    def __init__(self, split: str = "test"):
        """
        Initialize GSM8K benchmark.

        Args:
            split: Dataset split to use ("train" or "test")
        """
        self.split = split
        self._dataset = None

    @property
    def name(self) -> str:
        return "GSM8K"

    def _load_dataset(self):
        """Lazy load the dataset from HuggingFace."""
        if self._dataset is None:
            try:
                from datasets import load_dataset

                self._dataset = load_dataset("gsm8k", "main", split=self.split)
            except Exception as e:
                raise RuntimeError(f"Failed to load GSM8K dataset: {e}")
        return self._dataset

    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        """
        Load questions from GSM8K.

        Args:
            n: Number of questions to load. If None, load all.

        Returns:
            List of Question objects
        """
        dataset = self._load_dataset()

        if n is not None:
            dataset = dataset.select(range(min(n, len(dataset))))

        questions = []
        for i, item in enumerate(dataset):
            # Extract the numerical answer from the solution
            # GSM8K format: solution text ending with "#### <number>"
            ground_truth = self._extract_ground_truth(item["answer"])

            # Format question with instruction for numeric answer
            formatted_text = f"""{item["question"]}

Solve this step by step. Your answer should be a single number."""

            questions.append(
                Question(
                    id=f"gsm8k_{i}",
                    text=formatted_text,
                    ground_truth=ground_truth,
                    metadata={
                        "full_solution": item["answer"],
                        "split": self.split,
                        "original_question": item["question"],
                    },
                )
            )

        return questions

    def _extract_ground_truth(self, answer: str) -> str:
        """
        Extract the numerical answer from GSM8K answer format.

        GSM8K answers end with "#### <number>"
        """
        # Look for #### followed by the answer
        match = re.search(r"####\s*(.+?)$", answer.strip())
        if match:
            return self._normalize_number(match.group(1).strip())

        # Fallback: try to find the last number in the text
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
        if numbers:
            return self._normalize_number(numbers[-1])

        return answer.strip()

    def _normalize_number(self, num_str: str) -> str:
        """Normalize a number string for comparison."""
        # Remove commas
        num_str = num_str.replace(",", "")

        # Try to convert to float and back to handle formatting
        try:
            num = float(num_str)
            # Return as integer if it's a whole number
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            return num_str

    def _extract_number_from_response(self, response: str) -> Optional[str]:
        """
        Extract a numerical answer from a model response.

        Tries multiple patterns:
        1. "FINAL ANSWER: [optional $]<number>"
        2. "answer is [optional $]<number>"
        3. Last number in the response
        """
        # Try FINAL ANSWER pattern first - allow optional $ prefix
        match = re.search(
            r"FINAL ANSWER:\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
            response,
            re.IGNORECASE,
        )
        if match:
            return self._normalize_number(match.group(1))

        # Try to find any number after "answer is" or similar - allow optional $
        match = re.search(
            r"(?:the answer is|answer is|equals?)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
            response,
            re.IGNORECASE,
        )
        if match:
            return self._normalize_number(match.group(1))

        # Fallback: find the last number in the response (strip any $ prefix)
        numbers = re.findall(r"\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", response)
        if numbers:
            return self._normalize_number(numbers[-1])

        return None

    def evaluate(self, question: Question, response: str) -> EvalResult:
        """
        Evaluate a response against the ground truth.

        Args:
            question: The question that was answered
            response: The model's response

        Returns:
            EvalResult with correctness assessment
        """
        predicted = self._extract_number_from_response(response)

        if predicted is None:
            return EvalResult(
                question_id=question.id,
                is_correct=False,
                predicted="[no number found]",
                expected=question.ground_truth,
            )

        # Normalize both for comparison
        expected = self._normalize_number(question.ground_truth) if question.ground_truth else None
        is_correct = predicted == expected if expected else None

        return EvalResult(
            question_id=question.id,
            is_correct=is_correct,
            predicted=predicted,
            expected=expected,
        )
