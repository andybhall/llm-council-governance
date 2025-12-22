"""Base classes for governance structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class CouncilResult:
    """Result from running a governance structure."""

    final_answer: str
    stage1_responses: Dict[str, str]  # model -> response
    stage2_data: Optional[Any] = None  # Structure-specific
    stage3_data: Optional[Any] = None  # Structure-specific
    metadata: Optional[Dict[str, Any]] = None  # Timings, token counts, etc.


class GovernanceStructure(ABC):
    """Abstract base class for governance structures."""

    def __init__(self, council_models: List[str], chairman_model: str):
        self.council_models = council_models
        self.chairman_model = chairman_model

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this structure."""
        pass

    @abstractmethod
    async def run(self, query: str) -> CouncilResult:
        """Execute the governance process and return result."""
        pass
