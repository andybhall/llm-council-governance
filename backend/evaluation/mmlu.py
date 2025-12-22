"""MMLU (Massive Multitask Language Understanding) benchmark - College level subjects."""

import re
from typing import List, Optional

from backend.evaluation.base import Benchmark, EvalResult, Question


class MMLUBenchmark(Benchmark):
    """
    MMLU College-level subjects: Challenging multiple choice questions.

    Supports various subjects including:
    - college_mathematics
    - college_physics
    - college_chemistry
    - abstract_algebra
    - etc.
    """

    def __init__(self, subject: str = "college_mathematics"):
        """
        Initialize MMLU benchmark.

        Args:
            subject: MMLU subject to use (e.g., 'college_mathematics', 'college_physics')
        """
        self.subject = subject
        self._dataset = None

    @property
    def name(self) -> str:
        # Convert subject name to readable format
        readable = self.subject.replace("_", " ").title()
        return f"MMLU-{readable}"

    def _load_dataset(self):
        """Lazy load the dataset from HuggingFace."""
        if self._dataset is None:
            try:
                from datasets import load_dataset

                self._dataset = load_dataset("cais/mmlu", self.subject, split="test")
            except Exception as e:
                raise RuntimeError(f"Failed to load MMLU {self.subject} dataset: {e}")
        return self._dataset

    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        """
        Load questions from MMLU.

        Args:
            n: Number of questions to load. If None, load all.

        Returns:
            List of Question objects formatted as multiple choice
        """
        dataset = self._load_dataset()

        if n is not None:
            dataset = dataset.select(range(min(n, len(dataset))))

        questions = []
        for i, item in enumerate(dataset):
            # Format as multiple choice question
            formatted_text, correct_answer = self._format_multiple_choice(item, i)

            questions.append(
                Question(
                    id=f"mmlu_{self.subject}_{i}",
                    text=formatted_text,
                    ground_truth=correct_answer,
                    metadata={
                        "subject": item.get("subject", self.subject),
                    },
                )
            )

        return questions

    def _format_multiple_choice(self, item: dict, idx: int) -> tuple[str, str]:
        """
        Format an MMLU item as a multiple choice question.

        Args:
            item: Dataset item with question, choices, answer
            idx: Question index

        Returns:
            Tuple of (formatted question text, correct answer letter)
        """
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]  # This is an integer index

        # Build formatted question with lettered options
        letters = "ABCD"
        options_text = []
        for i, choice in enumerate(choices):
            if i < len(letters):
                options_text.append(f"{letters[i]}. {choice}")

        formatted_question = f"""{question}

{chr(10).join(options_text)}

This is a challenging college-level question. Think through it carefully step by step.
Choose the best answer. State your choice as a single letter (A, B, C, or D).
End with: FINAL ANSWER: [letter]"""

        correct_letter = letters[answer_idx] if answer_idx < len(letters) else "A"

        return formatted_question, correct_letter

    def _extract_letter_from_response(self, response: str) -> Optional[str]:
        """
        Extract a letter answer from a model response.
        """
        # Try FINAL ANSWER pattern first
        match = re.search(
            r"FINAL ANSWER:\s*([A-Da-d])\b", response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Try "answer is [letter]" pattern
        match = re.search(
            r"(?:answer is|choose|select)\s*([A-Da-d])\b", response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Try to find a standalone letter at end
        match = re.search(r"\b([A-Da-d])\s*[.)]?\s*$", response.strip())
        if match:
            return match.group(1).upper()

        # Fallback: find first letter A-D that appears
        match = re.search(r"\b([A-Da-d])\b", response)
        if match:
            return match.group(1).upper()

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
        predicted = self._extract_letter_from_response(response)

        if predicted is None:
            return EvalResult(
                question_id=question.id,
                is_correct=False,
                predicted="[no letter found]",
                expected=question.ground_truth,
            )

        is_correct = predicted.upper() == question.ground_truth.upper()

        return EvalResult(
            question_id=question.id,
            is_correct=is_correct,
            predicted=predicted,
            expected=question.ground_truth,
        )
