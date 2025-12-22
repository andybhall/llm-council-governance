"""Tests for governance base classes."""

import pytest

from backend.governance.base import CouncilResult, GovernanceStructure


def test_council_result_dataclass():
    """Verify CouncilResult can be instantiated with required fields."""
    result = CouncilResult(
        final_answer="42",
        stage1_responses={"model1": "response1"},
    )
    assert result.final_answer == "42"
    assert result.stage1_responses == {"model1": "response1"}


def test_council_result_optional_fields():
    """Verify CouncilResult optional fields default to None."""
    result = CouncilResult(
        final_answer="42",
        stage1_responses={"model1": "response1"},
    )
    assert result.stage2_data is None
    assert result.stage3_data is None
    assert result.metadata is None


def test_council_result_with_all_fields():
    """Verify CouncilResult works with all fields populated."""
    result = CouncilResult(
        final_answer="42",
        stage1_responses={"model1": "response1", "model2": "response2"},
        stage2_data={"rankings": [1, 2]},
        stage3_data={"synthesis": "combined answer"},
        metadata={"duration_ms": 1500, "tokens": 200},
    )
    assert result.final_answer == "42"
    assert len(result.stage1_responses) == 2
    assert result.stage2_data["rankings"] == [1, 2]
    assert result.stage3_data["synthesis"] == "combined answer"
    assert result.metadata["duration_ms"] == 1500


def test_governance_structure_is_abstract():
    """Verify GovernanceStructure cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GovernanceStructure(
            council_models=["model1", "model2"],
            chairman_model="chairman",
        )


def test_governance_structure_subclass():
    """Verify GovernanceStructure can be subclassed properly."""

    class ConcreteStructure(GovernanceStructure):
        @property
        def name(self) -> str:
            return "Test Structure"

        async def run(self, query: str) -> CouncilResult:
            return CouncilResult(
                final_answer="test answer",
                stage1_responses={m: f"response from {m}" for m in self.council_models},
            )

    structure = ConcreteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )

    assert structure.name == "Test Structure"
    assert structure.council_models == ["model1", "model2"]
    assert structure.chairman_model == "chairman"


@pytest.mark.asyncio
async def test_governance_structure_run():
    """Verify subclass run() method works correctly."""

    class ConcreteStructure(GovernanceStructure):
        @property
        def name(self) -> str:
            return "Test Structure"

        async def run(self, query: str) -> CouncilResult:
            return CouncilResult(
                final_answer="42",
                stage1_responses={m: f"{m} says 42" for m in self.council_models},
                metadata={"query": query},
            )

    structure = ConcreteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )

    result = await structure.run("What is the answer?")

    assert result.final_answer == "42"
    assert len(result.stage1_responses) == 2
    assert result.metadata["query"] == "What is the answer?"


def test_governance_structure_imports_from_package():
    """Verify classes can be imported from governance package."""
    from backend.governance import CouncilResult, GovernanceStructure

    assert CouncilResult is not None
    assert GovernanceStructure is not None
