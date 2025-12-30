"""TruthfulQA benchmark loader and evaluator."""

import hashlib
import random
import re
from typing import List, Optional, Tuple

from backend.evaluation.base import Benchmark, EvalResult, Question


class TruthfulQABenchmark(Benchmark):
    """
    TruthfulQA: Measuring How Models Mimic Human Falsehoods.

    Tests whether models give truthful answers rather than common misconceptions.
    Uses the multiple-choice format for easy evaluation.
    """

    def __init__(self, split: str = "validation"):
        """
        Initialize TruthfulQA benchmark.

        Args:
            split: Dataset split to use ("validation" - TruthfulQA only has validation)
        """
        self.split = split
        self._dataset_mc = None  # multiple_choice config
        self._dataset_gen = None  # generation config

    @property
    def name(self) -> str:
        return "TruthfulQA"

    def _load_dataset(self, config: str = "multiple_choice"):
        """
        Lazy load the dataset from HuggingFace.

        Args:
            config: Dataset config - "multiple_choice" or "generation"
        """
        if config == "generation":
            if self._dataset_gen is None:
                try:
                    from datasets import load_dataset
                    self._dataset_gen = load_dataset(
                        "truthful_qa", "generation", split=self.split
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load TruthfulQA generation dataset: {e}")
            return self._dataset_gen
        else:
            if self._dataset_mc is None:
                try:
                    from datasets import load_dataset
                    self._dataset_mc = load_dataset(
                        "truthful_qa", "multiple_choice", split=self.split
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load TruthfulQA dataset: {e}")
            return self._dataset_mc

    def load_questions(
        self, n: Optional[int] = None, binary_format: bool = True
    ) -> List[Question]:
        """
        Load questions from TruthfulQA.

        Args:
            n: Number of questions to load. If None, load all.
            binary_format: If True, use binary A/B format with randomized order
                (recommended, prevents position bias). If False, use legacy MC1 format.

        Returns:
            List of Question objects formatted as multiple choice
        """
        # Binary format uses "generation" config which has best_answer/incorrect_answers
        # Legacy MC format uses "multiple_choice" config which has mc1_targets
        config = "generation" if binary_format else "multiple_choice"
        dataset = self._load_dataset(config)

        if n is not None:
            dataset = dataset.select(range(min(n, len(dataset))))

        questions = []
        for i, item in enumerate(dataset):
            if binary_format:
                # Use new binary A/B format with randomized order
                formatted_text, correct_answer, rand_info = self._format_binary_choice(
                    item
                )
                metadata = {
                    "original_question": item["question"],
                    "category": item.get("category", "unknown"),
                    "source": item.get("source", "unknown"),
                    "split": self.split,
                    "format": "binary",
                    "randomization": rand_info,
                }
            else:
                # Legacy MC1 format (fixed order - subject to position bias)
                formatted_text, correct_answer = self._format_multiple_choice(item)
                metadata = {
                    "original_question": item["question"],
                    "category": item.get("category", "unknown"),
                    "source": item.get("source", "unknown"),
                    "split": self.split,
                    "format": "mc1",
                }

            questions.append(
                Question(
                    id=f"truthfulqa_{i}",
                    text=formatted_text,
                    ground_truth=correct_answer,
                    metadata=metadata,
                )
            )

        return questions

    def _format_binary_choice(
        self, item: dict, seed: Optional[int] = None
    ) -> Tuple[str, str, dict]:
        """
        Format a TruthfulQA item as binary A/B choice with randomized order.

        Uses the improved binary format recommended by TruthfulQA authors:
        - Only two options: best correct answer vs best incorrect answer
        - Randomized order to prevent position bias

        See: https://www.alignmentforum.org/posts/Bunfwz6JsNd44kgLT/new-improved-multiple-choice-truthfulqa

        Args:
            item: Dataset item with question, best_answer, incorrect_answers
            seed: Random seed for reproducibility (defaults to hash of question)

        Returns:
            Tuple of (formatted question text, correct answer letter, randomization_info)
        """
        question = item["question"]
        best_answer = item.get("best_answer", "")
        incorrect_answers = item.get("incorrect_answers", [])

        # Get the best incorrect answer (first one in the list)
        best_incorrect = incorrect_answers[0] if incorrect_answers else ""

        # Use deterministic hash for reproducibility across Python sessions
        # (Python's built-in hash() is randomized per PEP 456)
        if seed is None:
            seed = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16)

        rng = random.Random(seed)

        # Randomize order: 50% chance best_answer is A, 50% chance it's B
        if rng.random() < 0.5:
            option_a = best_answer
            option_b = best_incorrect
            correct_letter = "A"
        else:
            option_a = best_incorrect
            option_b = best_answer
            correct_letter = "B"

        formatted_question = f"""{question}

A. {option_a}
B. {option_b}

Choose the best answer. Your answer should be a single letter (A or B)."""

        randomization_info = {
            "seed": seed,
            "correct_is_a": correct_letter == "A",
            "option_a_text": option_a,
            "option_b_text": option_b,
        }

        return formatted_question, correct_letter, randomization_info

    def _format_multiple_choice(self, item: dict) -> tuple[str, str]:
        """
        Format a TruthfulQA item as a multiple choice question.

        Args:
            item: Dataset item with question and mc1_targets/mc2_targets

        Returns:
            Tuple of (formatted question text, correct answer letter)
        """
        question = item["question"]

        # TruthfulQA has mc1_targets (single correct) and mc2_targets (multiple correct)
        # We use mc1_targets for simpler evaluation
        targets = item.get("mc1_targets", item.get("mc2_targets", {}))

        choices = targets.get("choices", [])
        labels = targets.get("labels", [])

        # Find the correct answer (label == 1)
        correct_idx = None
        for idx, label in enumerate(labels):
            if label == 1:
                correct_idx = idx
                break

        # Build formatted question with lettered options
        letters = "ABCDEFGHIJ"
        options_text = []
        for idx, choice in enumerate(choices):
            if idx < len(letters):
                options_text.append(f"{letters[idx]}. {choice}")

        formatted_question = f"""{question}

{chr(10).join(options_text)}

Choose the best answer. Your answer should be a single letter (A, B, C, etc.)."""

        correct_letter = letters[correct_idx] if correct_idx is not None else "A"

        return formatted_question, correct_letter

    def _extract_letter_from_response(
        self, response: str, binary_only: bool = False
    ) -> Optional[str]:
        """
        Extract a letter answer from a model response.

        Tries multiple patterns:
        1. "FINAL ANSWER: [letter]"
        2. Standalone letter at end
        3. First letter mentioned

        Args:
            response: Model response text
            binary_only: If True, only accept A or B (for binary format)
        """
        # Define the letter pattern based on format
        letter_pattern = r"[A-Ba-b]" if binary_only else r"[A-Ja-j]"

        # Try FINAL ANSWER pattern first
        match = re.search(
            rf"FINAL ANSWER:\s*({letter_pattern})\b", response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Try "answer is [letter]" pattern
        match = re.search(
            rf"(?:answer is|choose|select)\s*({letter_pattern})\b",
            response,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

        # Try to find a standalone letter (likely the answer)
        # Look for pattern like "A." or "(A)" or just "A" at word boundary
        match = re.search(rf"\b({letter_pattern})\s*[.)]?\s*$", response.strip())
        if match:
            return match.group(1).upper()

        # Fallback: find first valid letter that appears as a standalone word
        match = re.search(rf"\b({letter_pattern})\b", response)
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
        # Use binary_only extraction if the question was formatted as binary
        binary_only = (
            question.metadata.get("format") == "binary"
            if question.metadata
            else False
        )
        predicted = self._extract_letter_from_response(response, binary_only=binary_only)

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
