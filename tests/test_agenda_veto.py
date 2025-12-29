"""Tests for Agenda Setter + Veto Structure."""

import importlib

import pytest

from backend.governance import AgendaSetterVetoStructure
from backend.governance.utils import extract_vote_accept_veto


class TestExtractVoteAcceptVeto:
    """Tests for the extract_vote_accept_veto utility function."""

    def test_extract_accept_structured(self):
        """Test extracting structured ACCEPT vote."""
        text = "I think this is good. FINAL VOTE: ACCEPT"
        assert extract_vote_accept_veto(text) == "ACCEPT"

    def test_extract_veto_structured(self):
        """Test extracting structured VETO vote."""
        text = "I disagree with this proposal. FINAL VOTE: VETO"
        assert extract_vote_accept_veto(text) == "VETO"

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        text = "final vote: accept"
        assert extract_vote_accept_veto(text) == "ACCEPT"

    def test_fallback_veto(self):
        """Test fallback detection of VETO."""
        text = "I strongly veto this proposal."
        assert extract_vote_accept_veto(text) == "VETO"

    def test_fallback_accept(self):
        """Test fallback detection of ACCEPT."""
        text = "I accept the chairman's proposal."
        assert extract_vote_accept_veto(text) == "ACCEPT"

    def test_veto_takes_precedence(self):
        """Test that VETO is found if both words present."""
        text = "I was going to accept but I veto instead."
        assert extract_vote_accept_veto(text) == "VETO"

    def test_none_for_empty(self):
        """Test None returned for empty string."""
        assert extract_vote_accept_veto("") is None

    def test_none_for_no_vote(self):
        """Test None returned when no vote word found."""
        text = "I'm not sure about this proposal."
        assert extract_vote_accept_veto(text) is None


@pytest.fixture
def mock_openrouter_proposal_passes(monkeypatch):
    """Mock where the proposal passes (few vetoes)."""
    call_log = []

    async def mock_query_model(model, messages, temperature=0.0, timeout=None):
        call_log.append({"model": model, "messages": messages})
        content = messages[-1]["content"]

        if "chairman" in model.lower() or "propose" in content.lower():
            # Chairman proposal
            return {"content": "The answer is clearly 4. FINAL ANSWER: 4"}
        elif "vote" in content.lower():
            # Voting stage - mostly accept
            return {"content": "I accept the proposal. FINAL VOTE: ACCEPT"}
        else:
            # Stage 1
            return {"content": f"{model} thinks the answer is 4. FINAL ANSWER: 4"}

    async def mock_query_models_parallel(models, messages, temperature=0.0, timeout=None):
        return {model: {"content": f"{model} FINAL ANSWER: 4"} for model in models}

    openrouter_module = importlib.import_module("backend.openrouter")
    monkeypatch.setattr(openrouter_module, "query_model", mock_query_model)

    av_module = importlib.import_module("backend.governance.agenda_veto")
    monkeypatch.setattr(av_module, "query_model", mock_query_model)

    base_module = importlib.import_module("backend.governance.base")
    monkeypatch.setattr(base_module, "query_model", mock_query_model)
    monkeypatch.setattr(base_module, "query_models_parallel", mock_query_models_parallel)

    return call_log


@pytest.fixture
def mock_openrouter_proposal_fails(monkeypatch):
    """Mock where the proposal fails (majority vetoes)."""
    call_log = []

    async def mock_query_model(model, messages, temperature=0.0, timeout=None):
        call_log.append({"model": model, "messages": messages})
        content = messages[-1]["content"]

        if "as chairman" in content.lower():
            # Chairman proposal (stage 2) - has "As chairman"
            return {"content": "I propose 5. FINAL ANSWER: 5"}
        elif "final vote" in content.lower() or "you must now vote" in content.lower():
            # Voting stage - majority veto
            return {"content": "I disagree with 5. FINAL VOTE: VETO"}
        else:
            # Stage 1 - council says 4
            return {"content": f"{model} thinks the answer is 4. FINAL ANSWER: 4"}

    async def mock_query_models_parallel(models, messages, temperature=0.0, timeout=None):
        return {model: {"content": f"{model} FINAL ANSWER: 4"} for model in models}

    openrouter_module = importlib.import_module("backend.openrouter")
    monkeypatch.setattr(openrouter_module, "query_model", mock_query_model)

    av_module = importlib.import_module("backend.governance.agenda_veto")
    monkeypatch.setattr(av_module, "query_model", mock_query_model)

    base_module = importlib.import_module("backend.governance.base")
    monkeypatch.setattr(base_module, "query_model", mock_query_model)
    monkeypatch.setattr(base_module, "query_models_parallel", mock_query_models_parallel)

    return call_log


@pytest.mark.asyncio
async def test_agenda_veto_runs(mock_openrouter_proposal_passes):
    """Verify agenda setter + veto structure executes successfully."""
    structure = AgendaSetterVetoStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.final_answer is not None
    assert len(result.stage1_responses) == 3


@pytest.mark.asyncio
async def test_agenda_veto_name():
    """Verify structure has correct name."""
    structure = AgendaSetterVetoStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    assert structure.name == "Agenda Setter + Veto"


@pytest.mark.asyncio
async def test_proposal_passes_uses_chair_answer(mock_openrouter_proposal_passes):
    """Test that passed proposals use the chairman's answer."""
    structure = AgendaSetterVetoStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.stage3_data["proposal_passed"] is True
    assert result.final_answer == "4"  # Chairman's answer


@pytest.mark.asyncio
async def test_proposal_fails_uses_fallback(mock_openrouter_proposal_fails):
    """Test that failed proposals use fallback rule."""
    structure = AgendaSetterVetoStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.stage3_data["proposal_passed"] is False
    assert result.stage3_data["fallback_used"] is True
    # Fallback to stage1 majority vote (all said 4)
    assert result.final_answer == "4"


@pytest.mark.asyncio
async def test_stage2_data_contains_proposal(mock_openrouter_proposal_passes):
    """Verify stage2_data contains chairman proposal."""
    structure = AgendaSetterVetoStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert "chair_proposal" in result.stage2_data
    assert "chair_response" in result.stage2_data
    assert result.stage2_data["chair_proposal"] is not None


@pytest.mark.asyncio
async def test_stage3_data_contains_votes(mock_openrouter_proposal_passes):
    """Verify stage3_data contains voting information."""
    structure = AgendaSetterVetoStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    stage3 = result.stage3_data
    assert "votes" in stage3
    assert "veto_count" in stage3
    assert "veto_threshold" in stage3
    assert "proposal_passed" in stage3
    assert "fallback_used" in stage3


@pytest.mark.asyncio
async def test_veto_threshold_default():
    """Test default veto threshold is majority."""
    structure = AgendaSetterVetoStructure(
        council_models=["m1", "m2", "m3", "m4", "m5"],
        chairman_model="chair",
    )
    # ceil(5/2) = 3
    assert structure.veto_threshold == 3


@pytest.mark.asyncio
async def test_custom_veto_threshold():
    """Test custom veto threshold."""
    structure = AgendaSetterVetoStructure(
        council_models=["m1", "m2", "m3"],
        chairman_model="chair",
        veto_threshold=1,  # Even one veto rejects
    )
    assert structure.veto_threshold == 1


def test_agenda_veto_imports_from_package():
    """Verify AgendaSetterVetoStructure can be imported from governance package."""
    from backend.governance import AgendaSetterVetoStructure

    assert AgendaSetterVetoStructure is not None


def test_agenda_veto_in_structure_registry():
    """Verify structure is registered in STRUCTURES dict."""
    from backend.governance import STRUCTURES, get_structure

    assert "agenda_veto" in STRUCTURES
    assert get_structure("agenda_veto") == AgendaSetterVetoStructure
