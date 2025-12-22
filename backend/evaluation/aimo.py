"""AIMO (AI Mathematical Olympiad) benchmark - Level 5 competition math."""

import re
from typing import List, Optional

from backend.evaluation.base import Benchmark, EvalResult, Question


class AIMOBenchmark(Benchmark):
    """
    AIMO Level 5: Competition mathematics problems.

    These are olympiad-level math problems requiring sophisticated reasoning.
    Answers are integers.
    """

    def __init__(self):
        self._dataset = None

    @property
    def name(self) -> str:
        return "AIMO"

    def _load_dataset(self):
        """Lazy load the dataset from HuggingFace."""
        if self._dataset is None:
            try:
                from datasets import load_dataset

                self._dataset = load_dataset(
                    "AI-MO/aimo-validation-math-level-5", split="train"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load AIMO dataset: {e}")
        return self._dataset

    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        """
        Load questions from AIMO.

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
            # Format the problem with instructions
            problem_text = f"""{item['problem']}

Solve this competition math problem step by step. Show your reasoning clearly.
The answer is an integer. End with: FINAL ANSWER: [integer]"""

            questions.append(
                Question(
                    id=f"aimo_{i}",
                    text=problem_text,
                    ground_truth=str(item["answer"]),
                    metadata={
                        "original_id": item.get("id"),
                        "difficulty": "Level 5 (Olympiad)",
                    },
                )
            )

        return questions

    def _extract_integer_from_response(self, response: str) -> Optional[str]:
        """
        Extract an integer answer from a model response.
        """
        # Try FINAL ANSWER pattern first
        match = re.search(
            r"FINAL ANSWER:\s*(-?\d+)", response, re.IGNORECASE
        )
        if match:
            return match.group(1)

        # Try "answer is" pattern
        match = re.search(
            r"(?:answer is|equals?|=)\s*(-?\d+)\s*$",
            response,
            re.IGNORECASE | re.MULTILINE,
        )
        if match:
            return match.group(1)

        # Try boxed format common in math: \boxed{answer}
        match = re.search(r"\\boxed\{(-?\d+)\}", response)
        if match:
            return match.group(1)

        # Fallback: find the last standalone integer
        matches = re.findall(r"(?<![.\d])(-?\d+)(?![.\d])", response)
        if matches:
            return matches[-1]

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
        predicted = self._extract_integer_from_response(response)

        if predicted is None:
            return EvalResult(
                question_id=question.id,
                is_correct=False,
                predicted="[no integer found]",
                expected=question.ground_truth,
            )

        is_correct = predicted == question.ground_truth

        return EvalResult(
            question_id=question.id,
            is_correct=is_correct,
            predicted=predicted,
            expected=question.ground_truth,
        )
