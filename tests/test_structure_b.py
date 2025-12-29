"""Tests for Structure B: Independent → Majority Vote."""

import importlib

import pytest

from backend.governance import MajorityVoteStructure


@pytest.fixture
def mock_openrouter(monkeypatch):
    """Mock openrouter API calls for testing."""
    call_log = []

    async def mock_query_model(model, messages, temperature=0.0, timeout=None):
        call_log.append({"type": "single", "model": model, "messages": messages})
        # Chairman response
        return {"content": "The chairman thinks 4. FINAL ANSWER: 4"}

    async def mock_query_models_parallel(models, messages, temperature=0.0, timeout=None):
        call_log.append({"type": "parallel", "models": models, "messages": messages})
        # Return different answers - 4 wins by majority
        results = {}
        for i, model in enumerate(models):
            if i % 3 == 0:
                results[model] = {"content": f"{model} says FINAL ANSWER: 5"}
            else:
                results[model] = {"content": f"{model} says FINAL ANSWER: 4"}
        return results

    # Patch at the actual implementation locations (use importlib to avoid namespace collision)
    base_module = importlib.import_module("backend.governance.base")

    # Both query_model and query_models_parallel are now in base class
    monkeypatch.setattr(base_module, "query_model", mock_query_model)
    monkeypatch.setattr(base_module, "query_models_parallel", mock_query_models_parallel)

    return call_log


@pytest.fixture
def mock_openrouter_tie(monkeypatch):
    """Mock openrouter with tied votes."""
    call_log = []

    async def mock_query_model(model, messages, temperature=0.0, timeout=None):
        call_log.append({"type": "single", "model": model, "messages": messages})
        return {"content": "Chairman picks B. FINAL ANSWER: B"}

    async def mock_query_models_parallel(models, messages, temperature=0.0, timeout=None):
        call_log.append({"type": "parallel", "models": models, "messages": messages})
        # Return tied votes
        results = {}
        for i, model in enumerate(models):
            if i % 2 == 0:
                results[model] = {"content": f"{model}: FINAL ANSWER: A"}
            else:
                results[model] = {"content": f"{model}: FINAL ANSWER: B"}
        return results

    # Patch at the actual implementation locations (use importlib to avoid namespace collision)
    base_module = importlib.import_module("backend.governance.base")

    # Both query_model and query_models_parallel are now in base class
    monkeypatch.setattr(base_module, "query_model", mock_query_model)
    monkeypatch.setattr(base_module, "query_models_parallel", mock_query_models_parallel)

    return call_log


@pytest.mark.asyncio
async def test_structure_b_runs(mock_openrouter):
    """Verify Structure B executes successfully."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.final_answer is not None
    assert len(result.stage1_responses) == 3


@pytest.mark.asyncio
async def test_structure_b_name():
    """Verify Structure B has correct name."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    assert structure.name == "Independent → Majority Vote"


@pytest.mark.asyncio
async def test_structure_b_majority_wins(mock_openrouter):
    """Verify majority answer wins."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    # Based on mock: model1 says 5, model2 and model3 say 4
    assert result.final_answer == "4"


@pytest.mark.asyncio
async def test_structure_b_extracts_answers(mock_openrouter):
    """Verify answers are extracted from all responses."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    extracted = result.stage2_data["extracted_answers"]
    assert len(extracted) == 3
    assert all(ans is not None for ans in extracted.values())


@pytest.mark.asyncio
async def test_structure_b_tiebreaker(mock_openrouter_tie):
    """Verify chairman acts as tiebreaker."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2"],  # Even number for tie
        chairman_model="chairman",
    )
    result = await structure.run("A or B?")

    # Chairman says B, so B should win the tie
    assert result.final_answer == "B"
    assert result.stage3_data["chairman_tiebreaker"] == "B"


@pytest.mark.asyncio
async def test_structure_b_api_call_sequence(mock_openrouter):
    """Verify correct sequence of API calls."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    await structure.run("What is 2+2?")

    # Should have: Stage 1 (parallel), Stage 3 chairman (single)
    assert len(mock_openrouter) == 2
    assert mock_openrouter[0]["type"] == "parallel"  # Stage 1
    assert mock_openrouter[1]["type"] == "single"  # Chairman


@pytest.mark.asyncio
async def test_structure_b_stage1_has_responses(mock_openrouter):
    """Verify Stage 1 collects all responses."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    assert "model1" in result.stage1_responses
    assert "model2" in result.stage1_responses
    assert "model3" in result.stage1_responses


@pytest.mark.asyncio
async def test_structure_b_stage3_has_vote_result(mock_openrouter):
    """Verify Stage 3 data includes vote result."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    assert "vote_result" in result.stage3_data
    assert "chairman_tiebreaker" in result.stage3_data


def test_build_stage1_prompt():
    """Test that the shared prompt builder includes FINAL ANSWER instruction."""
    from backend.governance.utils import build_stage1_prompt

    prompt = build_stage1_prompt("What is 2+2?")

    assert "What is 2+2?" in prompt
    assert "FINAL ANSWER:" in prompt


def test_stage2_extract_answers():
    """Test answer extraction from responses."""
    structure = MajorityVoteStructure(
        council_models=["m1", "m2"],
        chairman_model="chair",
    )
    responses = {
        "m1": "I think it's 4. FINAL ANSWER: 4",
        "m2": "The answer is 5. FINAL ANSWER: 5",
    }

    extracted = structure._stage2_extract_answers(responses)

    assert extracted["m1"] == "4"
    assert extracted["m2"] == "5"


def test_stage2_extract_handles_missing_answer():
    """Test extraction handles responses without FINAL ANSWER."""
    structure = MajorityVoteStructure(
        council_models=["m1", "m2"],
        chairman_model="chair",
    )
    responses = {
        "m1": "I think it's 4. FINAL ANSWER: 4",
        "m2": "The answer is probably 5.",  # No FINAL ANSWER
    }

    extracted = structure._stage2_extract_answers(responses)

    assert extracted["m1"] == "4"
    assert extracted["m2"] is None


@pytest.mark.asyncio
async def test_structure_b_handles_no_valid_answers(monkeypatch):
    """Test handling when no responses have valid FINAL ANSWER."""
    call_log = []

    async def mock_query_model(model, messages, temperature=0.0, timeout=None):
        call_log.append({"type": "single", "model": model})
        return {"content": "Chairman fallback. FINAL ANSWER: fallback"}

    async def mock_query_models_parallel(models, messages, temperature=0.0, timeout=None):
        call_log.append({"type": "parallel", "models": models})
        # None of the responses have FINAL ANSWER
        return {
            model: {"content": f"{model} rambles without structure"}
            for model in models
        }

    # Patch at the actual implementation locations (use importlib to avoid namespace collision)
    base_module = importlib.import_module("backend.governance.base")

    # Both query_model and query_models_parallel are now in base class
    monkeypatch.setattr(base_module, "query_model", mock_query_model)
    monkeypatch.setattr(base_module, "query_models_parallel", mock_query_models_parallel)

    structure = MajorityVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    # Should fall back to chairman's answer
    assert result.final_answer == "fallback"


def test_structure_b_imports_from_package():
    """Verify MajorityVoteStructure can be imported from governance package."""
    from backend.governance import MajorityVoteStructure

    assert MajorityVoteStructure is not None


@pytest.mark.asyncio
async def test_structure_b_vote_metadata(mock_openrouter):
    """Verify Stage 3 data includes comprehensive vote metadata."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    stage3 = result.stage3_data

    # Check all vote metadata fields are present
    assert "raw_answers" in stage3
    assert "normalized_answers" in stage3
    assert "vote_counts" in stage3
    assert "is_tie" in stage3
    assert "winning_answer" in stage3
    assert "tiebreaker_used" in stage3

    # Verify raw_answers has all models
    assert len(stage3["raw_answers"]) == 3

    # Verify normalized_answers has all models
    assert len(stage3["normalized_answers"]) == 3

    # Verify vote_counts is a dict with counts
    assert isinstance(stage3["vote_counts"], dict)
    assert len(stage3["vote_counts"]) > 0

    # Verify is_tie is boolean
    assert isinstance(stage3["is_tie"], bool)

    # Verify tiebreaker_used is boolean
    assert isinstance(stage3["tiebreaker_used"], bool)


@pytest.mark.asyncio
async def test_structure_b_vote_metadata_tie(mock_openrouter_tie):
    """Verify vote metadata correctly indicates tie and tiebreaker usage."""
    structure = MajorityVoteStructure(
        council_models=["model1", "model2"],  # Even number for tie
        chairman_model="chairman",
    )
    result = await structure.run("A or B?")

    stage3 = result.stage3_data

    # Should be a tie with 2 models giving different answers
    assert stage3["is_tie"] is True

    # Tiebreaker should have been used
    assert stage3["tiebreaker_used"] is True

    # Vote counts should show tied votes
    assert len(stage3["vote_counts"]) == 2
