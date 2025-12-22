"""MMLU-Pro benchmark - Harder version of MMLU with 10 answer choices."""

import re
from typing import List, Optional

from backend.evaluation.base import Benchmark, EvalResult, Question


class MMLUProBenchmark(Benchmark):
    """
    MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.

    Key differences from standard MMLU:
    - 10 answer choices (A-J) instead of 4
    - More challenging, reasoning-focused questions
    - ~12,000 questions across 14 domains
    - GPT-4o scores ~72%, Claude-3-Sonnet ~55% (vs 85-90% on standard MMLU)

    Supports filtering by category:
    - math, physics, chemistry, law, engineering, economics, health,
      psychology, business, biology, philosophy, computer science, history, other
    """

    VALID_CATEGORIES = [
        "math", "physics", "chemistry", "law", "engineering", "economics",
        "health", "psychology", "business", "biology", "philosophy",
        "computer science", "history", "other"
    ]

    def __init__(self, category: Optional[str] = None):
        """
        Initialize MMLU-Pro benchmark.

        Args:
            category: Optional category to filter by (e.g., 'math', 'physics').
                     If None, uses all categories.
        """
        self.category = category
        self._dataset = None

    @property
    def name(self) -> str:
        if self.category:
            readable = self.category.replace("_", " ").title()
            return f"MMLU-Pro-{readable}"
        return "MMLU-Pro"

    def _load_dataset(self):
        """Lazy load the dataset from HuggingFace."""
        if self._dataset is None:
            try:
                from datasets import load_dataset

                self._dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

                # Filter by category if specified
                if self.category:
                    self._dataset = self._dataset.filter(
                        lambda x: x['category'] == self.category
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to load MMLU-Pro dataset: {e}")
        return self._dataset

    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        """
        Load questions from MMLU-Pro.

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
            # Format as multiple choice question with 10 options
            formatted_text, correct_answer = self._format_multiple_choice(item)

            questions.append(
                Question(
                    id=f"mmlu_pro_{item['question_id']}",
                    text=formatted_text,
                    ground_truth=correct_answer,
                    metadata={
                        "category": item.get("category", "unknown"),
                        "source": item.get("src", "unknown"),
                    },
                )
            )

        return questions

    def _format_multiple_choice(self, item: dict) -> tuple[str, str]:
        """
        Format an MMLU-Pro item as a multiple choice question.

        Args:
            item: Dataset item with question, options, answer, answer_index

        Returns:
            Tuple of (formatted question text, correct answer letter)
        """
        question = item["question"]
        options = item["options"]
        answer_index = item["answer_index"]

        # Build formatted question with lettered options (A-J)
        letters = "ABCDEFGHIJ"
        options_text = []
        for i, choice in enumerate(options):
            if i < len(letters):
                options_text.append(f"{letters[i]}. {choice}")

        formatted_question = f"""{question}

{chr(10).join(options_text)}

This is a challenging question requiring careful reasoning. Think through it step by step.
Choose the best answer from the options above. Your answer should be a single letter (A through J)."""

        correct_letter = letters[answer_index] if answer_index < len(letters) else "A"

        return formatted_question, correct_letter

    def _extract_letter_from_response(self, response: str) -> Optional[str]:
        """
        Extract a letter answer (A-J) from a model response.
        """
        # Try FINAL ANSWER pattern first (most reliable)
        match = re.search(
            r"FINAL ANSWER:\s*([A-Ja-j])\b", response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Try "answer is [letter]" pattern
        match = re.search(
            r"(?:the answer is|answer is|answer:)\s*([A-Ja-j])\b", response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Try "option [letter]" or "choice [letter]" pattern
        match = re.search(
            r"(?:correct option is|option|choice)\s*([A-Ja-j])\b", response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Try to find a standalone letter at end of response
        match = re.search(r"\b([A-Ja-j])\s*[.)]?\s*$", response.strip())
        if match:
            return match.group(1).upper()

        # Try "[letter]." at the very end (common format)
        match = re.search(r"\b([A-Ja-j])\.\s*$", response.strip())
        if match:
            return match.group(1).upper()

        # Last resort: find the last standalone letter A-J (not part of a word)
        # Use findall to get the last one
        matches = re.findall(r"(?<![a-zA-Z])([A-Ja-j])(?![a-zA-Z])", response)
        if matches:
            return matches[-1].upper()

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
