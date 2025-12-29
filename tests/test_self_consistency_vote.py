"""Tests for Self-Consistency Vote Structure."""

import importlib

import pytest

from backend.governance import SelfConsistencyVoteStructure


@pytest.fixture
def mock_openrouter(monkeypatch):
    """Mock openrouter API calls for testing."""
    call_log = []

    async def mock_query_model(model, messages, temperature=0.0, timeout=None):
        call_log.append({
            "model": model,
            "messages": messages,
            "temperature": temperature,
        })
        # Return varying answers based on call count
        call_num = len(call_log)
        if call_num <= 3:
            return {"content": f"Sample {call_num} says FINAL ANSWER: 4"}
        elif call_num <= 5:
            return {"content": f"Sample {call_num} says FINAL ANSWER: 5"}
        else:
            return {"content": f"Sample {call_num} says FINAL ANSWER: 4"}

    # Patch openrouter at all necessary locations
    openrouter_module = importlib.import_module("backend.openrouter")
    monkeypatch.setattr(openrouter_module, "query_model", mock_query_model)

    sc_module = importlib.import_module("backend.governance.self_consistency_vote")
    monkeypatch.setattr(sc_module, "query_model", mock_query_model)

    return call_log


@pytest.mark.asyncio
async def test_self_consistency_runs(mock_openrouter):
    """Verify self-consistency structure executes successfully."""
    structure = SelfConsistencyVoteStructure(
        base_model="test-model",
        n_samples=5,
        temperature=0.7,
    )
    result = await structure.run("What is 2+2?")

    assert result.final_answer is not None
    assert len(result.stage1_responses) == 5


@pytest.mark.asyncio
async def test_self_consistency_name():
    """Verify structure has correct name."""
    structure = SelfConsistencyVoteStructure(
        base_model="test-model",
        n_samples=5,
    )
    assert structure.name == "Self-Consistency Vote"


@pytest.mark.asyncio
async def test_self_consistency_calls_n_samples_times(mock_openrouter):
    """Verify exactly n_samples API calls are made."""
    n_samples = 7
    structure = SelfConsistencyVoteStructure(
        base_model="test-model",
        n_samples=n_samples,
        temperature=0.7,
    )
    await structure.run("What is 2+2?")

    assert len(mock_openrouter) == n_samples


@pytest.mark.asyncio
async def test_self_consistency_uses_temperature(mock_openrouter):
    """Verify temperature is passed to API calls."""
    temperature = 0.8
    structure = SelfConsistencyVoteStructure(
        base_model="test-model",
        n_samples=3,
        temperature=temperature,
    )
    await structure.run("What is 2+2?")

    assert all(call["temperature"] == temperature for call in mock_openrouter)


@pytest.mark.asyncio
async def test_self_consistency_uses_base_model(mock_openrouter):
    """Verify base_model is used for all calls."""
    base_model = "my-special-model"
    structure = SelfConsistencyVoteStructure(
        base_model=base_model,
        n_samples=3,
        temperature=0.7,
    )
    await structure.run("What is 2+2?")

    assert all(call["model"] == base_model for call in mock_openrouter)


@pytest.mark.asyncio
async def test_self_consistency_majority_vote(mock_openrouter):
    """Verify majority answer wins."""
    structure = SelfConsistencyVoteStructure(
        base_model="test-model",
        n_samples=5,
        temperature=0.7,
    )
    result = await structure.run("What is 2+2?")

    # Based on mock: 3 samples say 4, 2 samples say 5
    assert result.final_answer == "4"


@pytest.mark.asyncio
async def test_self_consistency_stage2_data(mock_openrouter):
    """Verify stage2_data includes expected fields."""
    structure = SelfConsistencyVoteStructure(
        base_model="test-model",
        n_samples=5,
        temperature=0.7,
    )
    result = await structure.run("What is 2+2?")

    assert "extracted_answers" in result.stage2_data
    assert "base_model" in result.stage2_data
    assert "n_samples" in result.stage2_data
    assert "temperature" in result.stage2_data

    assert result.stage2_data["base_model"] == "test-model"
    assert result.stage2_data["n_samples"] == 5
    assert result.stage2_data["temperature"] == 0.7


@pytest.mark.asyncio
async def test_self_consistency_vote_metadata(mock_openrouter):
    """Verify vote metadata is present in stage3_data."""
    structure = SelfConsistencyVoteStructure(
        base_model="test-model",
        n_samples=5,
        temperature=0.7,
    )
    result = await structure.run("What is 2+2?")

    stage3 = result.stage3_data
    assert "vote_result" in stage3
    assert "raw_answers" in stage3
    assert "normalized_answers" in stage3
    assert "vote_counts" in stage3
    assert "is_tie" in stage3
    assert "winning_answer" in stage3
    assert "tiebreaker_used" in stage3


@pytest.mark.asyncio
async def test_self_consistency_default_params():
    """Verify default parameters are set correctly."""
    structure = SelfConsistencyVoteStructure()

    assert structure.base_model == "google/gemini-2.0-flash-001"
    assert structure.n_samples == 11
    assert structure.temperature == 0.7


@pytest.mark.asyncio
async def test_self_consistency_accepts_council_params():
    """Verify council_models and chairman_model are accepted for API compatibility."""
    # Should not raise
    structure = SelfConsistencyVoteStructure(
        base_model="test-model",
        n_samples=5,
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    assert structure.base_model == "test-model"


def test_self_consistency_imports_from_package():
    """Verify SelfConsistencyVoteStructure can be imported from governance package."""
    from backend.governance import SelfConsistencyVoteStructure

    assert SelfConsistencyVoteStructure is not None


def test_self_consistency_in_structure_registry():
    """Verify structure is registered in STRUCTURES dict."""
    from backend.governance import STRUCTURES, get_structure

    assert "self_consistency" in STRUCTURES
    assert get_structure("self_consistency") == SelfConsistencyVoteStructure
