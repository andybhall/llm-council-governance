"""Base classes for benchmark evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Question:
    """A single question from a benchmark."""

    id: str
    text: str
    ground_truth: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class EvalResult:
    """Result of evaluating a response against ground truth."""

    question_id: str
    is_correct: Optional[bool]
    predicted: str
    expected: Optional[str]


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this benchmark."""
        pass

    @abstractmethod
    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        """
        Load questions from the benchmark.

        Args:
            n: Optional number of questions to load. If None, load all.

        Returns:
            List of Question objects
        """
        pass

    @abstractmethod
    def evaluate(self, question: Question, response: str) -> EvalResult:
        """
        Evaluate a response against ground truth.

        Args:
            question: The question that was answered
            response: The model's response/answer

        Returns:
            EvalResult with correctness assessment
        """
        pass
