"""Tests for Structure A: Independent → Rank → Synthesize."""

import pytest

from backend.governance.independent_rank_synthesize import IndependentRankSynthesize


@pytest.fixture
def mock_openrouter(monkeypatch):
    """Mock openrouter API calls for testing."""
    call_log = []

    async def mock_query_model(model, messages):
        call_log.append({"type": "single", "model": model, "messages": messages})
        content = messages[-1]["content"]

        # Detect which stage based on prompt content
        if "FINAL RANKING:" in content:
            # Stage 2: ranking request
            return {"content": "After review, FINAL RANKING: A, B"}
        elif "synthesize" in content.lower() or "chairman" in content.lower():
            # Stage 3: synthesis request
            return {"content": "Based on the council's input, the answer is 4. FINAL ANSWER: 4"}
        else:
            # Stage 1: initial response
            return {"content": f"Response from {model}: The answer is 4. FINAL ANSWER: 4"}

    async def mock_query_models_parallel(models, messages):
        call_log.append({"type": "parallel", "models": models, "messages": messages})
        content = messages[-1]["content"]

        results = {}
        for model in models:
            if "FINAL RANKING:" in content:
                # Stage 2: ranking
                # Return different rankings from different models
                if "model1" in model:
                    results[model] = {"content": "I think FINAL RANKING: A, B"}
                else:
                    results[model] = {"content": "My ranking is FINAL RANKING: B, A"}
            else:
                # Stage 1: initial responses
                results[model] = {
                    "content": f"Response from {model}: I believe the answer is 4. FINAL ANSWER: 4"
                }
        return results

    from backend.governance import independent_rank_synthesize

    monkeypatch.setattr(
        independent_rank_synthesize, "query_model", mock_query_model
    )
    monkeypatch.setattr(
        independent_rank_synthesize, "query_models_parallel", mock_query_models_parallel
    )

    return call_log


@pytest.mark.asyncio
async def test_structure_a_runs(mock_openrouter):
    """Verify Structure A executes all three stages."""
    structure = IndependentRankSynthesize(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.final_answer is not None
    assert len(result.stage1_responses) == 2


@pytest.mark.asyncio
async def test_structure_a_name():
    """Verify Structure A has correct name."""
    structure = IndependentRankSynthesize(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    assert structure.name == "Independent → Rank → Synthesize"


@pytest.mark.asyncio
async def test_structure_a_stage1_responses(mock_openrouter):
    """Verify Stage 1 collects responses from all council models."""
    structure = IndependentRankSynthesize(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert "model1" in result.stage1_responses
    assert "model2" in result.stage1_responses
    assert "model1" in result.stage1_responses["model1"]


@pytest.mark.asyncio
async def test_structure_a_stage2_rankings(mock_openrouter):
    """Verify Stage 2 collects rankings from all models."""
    structure = IndependentRankSynthesize(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert "rankings" in result.stage2_data
    assert "label_to_model" in result.stage2_data
    assert len(result.stage2_data["rankings"]) == 2


@pytest.mark.asyncio
async def test_structure_a_stage3_synthesis(mock_openrouter):
    """Verify Stage 3 produces synthesis."""
    structure = IndependentRankSynthesize(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert "synthesis" in result.stage3_data
    assert result.stage3_data["synthesis"] is not None


@pytest.mark.asyncio
async def test_structure_a_extracts_final_answer(mock_openrouter):
    """Verify final answer is extracted from synthesis."""
    structure = IndependentRankSynthesize(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.final_answer == "4"


@pytest.mark.asyncio
async def test_structure_a_api_call_sequence(mock_openrouter):
    """Verify correct sequence of API calls."""
    structure = IndependentRankSynthesize(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    await structure.run("What is 2+2?")

    # Should have: Stage 1 (parallel), Stage 2 (parallel), Stage 3 (single)
    assert len(mock_openrouter) == 3
    assert mock_openrouter[0]["type"] == "parallel"  # Stage 1
    assert mock_openrouter[1]["type"] == "parallel"  # Stage 2
    assert mock_openrouter[2]["type"] == "single"  # Stage 3 chairman


def test_parse_ranking_valid():
    """Test parsing valid ranking string."""
    structure = IndependentRankSynthesize(
        council_models=["m1", "m2", "m3"],
        chairman_model="chair",
    )
    text = "After consideration, FINAL RANKING: B, A, C"
    result = structure._parse_ranking(text, ["A", "B", "C"])
    assert result == ["B", "A", "C"]


def test_parse_ranking_partial():
    """Test parsing ranking with some invalid labels."""
    structure = IndependentRankSynthesize(
        council_models=["m1", "m2"],
        chairman_model="chair",
    )
    text = "FINAL RANKING: A, X, B"
    result = structure._parse_ranking(text, ["A", "B"])
    assert result == ["A", "B"]


def test_parse_ranking_fallback():
    """Test fallback when no ranking found."""
    structure = IndependentRankSynthesize(
        council_models=["m1", "m2"],
        chairman_model="chair",
    )
    text = "I cannot decide on a ranking."
    result = structure._parse_ranking(text, ["A", "B"])
    assert result == ["A", "B"]  # Falls back to original order


def test_calculate_aggregate_rankings():
    """Test aggregate ranking calculation."""
    structure = IndependentRankSynthesize(
        council_models=["m1", "m2"],
        chairman_model="chair",
    )
    rankings = {
        "model1": ["A", "B", "C"],
        "model2": ["B", "A", "C"],
    }
    label_to_model = {"A": "m1", "B": "m2", "C": "m3"}

    result = structure._calculate_aggregate_rankings(rankings, label_to_model)

    # A: avg rank (1+2)/2 = 1.5
    # B: avg rank (2+1)/2 = 1.5
    # C: avg rank (3+3)/2 = 3.0
    assert len(result) == 3
    # A and B should be tied at 1.5, C should be 3.0
    assert result[2][0] == "C"
    assert result[2][1] == 3.0


def test_structure_a_imports_from_package():
    """Verify IndependentRankSynthesize can be imported from governance package."""
    from backend.governance import IndependentRankSynthesize

    assert IndependentRankSynthesize is not None
