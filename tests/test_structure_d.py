"""Tests for Structure D: Independent → Deliberate → Synthesize."""

import pytest

from backend.governance.structure_d import DeliberateSynthesizeStructure


@pytest.fixture
def mock_openrouter(monkeypatch):
    """Mock openrouter API calls for testing."""
    call_log = []

    async def mock_query_model(model, messages):
        call_log.append({"type": "single", "model": model, "messages": messages})
        content = messages[-1]["content"]

        # Detect stage based on prompt content
        if "chairman" in content.lower() and "synthesize" in content.lower():
            # Stage 3: Chairman synthesis
            return {
                "content": "After reviewing all positions, the answer is 4. FINAL ANSWER: 4"
            }
        elif "previously answered" in content.lower():
            # Stage 2: Deliberation
            return {
                "content": f"{model} reconsidered: I still think 4. FINAL ANSWER: 4"
            }
        else:
            # Stage 1 or other
            return {"content": f"{model} initial: The answer is 4. FINAL ANSWER: 4"}

    async def mock_query_models_parallel(models, messages):
        call_log.append({"type": "parallel", "models": models, "messages": messages})
        results = {}
        for i, model in enumerate(models):
            answer = 4 + i  # Different initial answers
            results[model] = {
                "content": f"{model}: I think {answer}. FINAL ANSWER: {answer}"
            }
        return results

    import backend.governance.structure_d as struct_d_module

    monkeypatch.setattr(struct_d_module, "query_model", mock_query_model)
    monkeypatch.setattr(struct_d_module, "query_models_parallel", mock_query_models_parallel)

    return call_log


@pytest.mark.asyncio
async def test_structure_d_runs(mock_openrouter):
    """Verify Structure D executes successfully."""
    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.final_answer is not None
    assert len(result.stage1_responses) == 2


@pytest.mark.asyncio
async def test_structure_d_name():
    """Verify Structure D has correct name."""
    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    assert structure.name == "Independent → Deliberate → Synthesize"


@pytest.mark.asyncio
async def test_structure_d_has_deliberation_stage(mock_openrouter):
    """Verify Stage 2 deliberation responses are captured."""
    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert "deliberation_responses" in result.stage2_data
    assert len(result.stage2_data["deliberation_responses"]) == 2


@pytest.mark.asyncio
async def test_structure_d_has_synthesis(mock_openrouter):
    """Verify Stage 3 produces synthesis."""
    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert "synthesis" in result.stage3_data
    assert result.stage3_data["synthesis"] is not None
    assert len(result.stage3_data["synthesis"]) > 0


@pytest.mark.asyncio
async def test_structure_d_extracts_final_answer(mock_openrouter):
    """Verify final answer is extracted from synthesis."""
    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.final_answer == "4"


@pytest.mark.asyncio
async def test_structure_d_api_call_sequence(mock_openrouter):
    """Verify correct sequence of API calls."""
    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    await structure.run("What is 2+2?")

    # Should have:
    # 1. Stage 1: parallel query
    # 2. Stage 2: individual deliberation calls
    # 3. Stage 3: chairman synthesis
    assert mock_openrouter[0]["type"] == "parallel"  # Stage 1

    # Remaining calls should be single (deliberation + synthesis)
    single_calls = [c for c in mock_openrouter if c["type"] == "single"]
    assert len(single_calls) >= 3  # 2 deliberation + 1 synthesis


@pytest.mark.asyncio
async def test_structure_d_stage1_responses(mock_openrouter):
    """Verify Stage 1 collects all responses."""
    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    assert "model1" in result.stage1_responses
    assert "model2" in result.stage1_responses
    assert "model3" in result.stage1_responses


def test_build_stage1_prompt():
    """Test that the shared prompt builder includes FINAL ANSWER instruction."""
    from backend.governance.utils import build_stage1_prompt

    prompt = build_stage1_prompt("What is 2+2?")

    assert "What is 2+2?" in prompt
    assert "FINAL ANSWER:" in prompt


def test_build_deliberation_prompt():
    """Test deliberation prompt includes required elements."""
    structure = DeliberateSynthesizeStructure(
        council_models=["m1", "m2"],
        chairman_model="chair",
    )

    all_responses = {
        "m1": "I think 4. FINAL ANSWER: 4",
        "m2": "I think 5. FINAL ANSWER: 5",
    }

    prompt = structure._build_deliberation_prompt(
        query="What is 2+2?",
        model="m1",
        own_response="I think 4. FINAL ANSWER: 4",
        all_responses=all_responses,
    )

    assert "What is 2+2?" in prompt
    assert "previously answered" in prompt.lower()
    assert "I think 4" in prompt
    assert "I think 5" in prompt
    assert "FINAL ANSWER:" in prompt


def test_build_synthesis_prompt():
    """Test synthesis prompt includes both stages."""
    structure = DeliberateSynthesizeStructure(
        council_models=["m1", "m2"],
        chairman_model="chair",
    )

    stage1 = {
        "m1": "Initial response 1",
        "m2": "Initial response 2",
    }
    stage2 = {
        "m1": "Deliberated response 1",
        "m2": "Deliberated response 2",
    }

    prompt = structure._build_synthesis_prompt(
        query="Test question",
        stage1_responses=stage1,
        stage2_responses=stage2,
    )

    assert "Test question" in prompt
    assert "chairman" in prompt.lower()
    assert "Initial response 1" in prompt
    assert "Initial response 2" in prompt
    assert "Deliberated response 1" in prompt
    assert "Deliberated response 2" in prompt
    assert "FINAL ANSWER:" in prompt


def test_synthesis_prompt_mentions_deliberation_insights():
    """Test synthesis prompt guides chairman on what to consider."""
    structure = DeliberateSynthesizeStructure(
        council_models=["m1"],
        chairman_model="chair",
    )

    prompt = structure._build_synthesis_prompt(
        query="Test",
        stage1_responses={"m1": "resp1"},
        stage2_responses={"m1": "resp2"},
    )

    # Should mention key considerations
    assert "agreement" in prompt.lower() or "deliberat" in prompt.lower()


@pytest.mark.asyncio
async def test_structure_d_synthesis_uses_both_stages(monkeypatch):
    """Verify synthesis prompt includes both initial and deliberated responses."""
    captured_prompts = []

    async def mock_query_model(model, messages):
        captured_prompts.append(messages[-1]["content"])
        return {"content": "Synthesis result. FINAL ANSWER: 42"}

    async def mock_query_models_parallel(models, messages):
        return {m: {"content": f"{m} response"} for m in models}

    import backend.governance.structure_d as struct_d_module

    monkeypatch.setattr(struct_d_module, "query_model", mock_query_model)
    monkeypatch.setattr(struct_d_module, "query_models_parallel", mock_query_models_parallel)

    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    await structure.run("Test query")

    # Last prompt should be synthesis
    synthesis_prompt = captured_prompts[-1]
    assert "Initial" in synthesis_prompt or "initial" in synthesis_prompt
    assert "Deliberation" in synthesis_prompt or "deliberat" in synthesis_prompt.lower()


@pytest.mark.asyncio
async def test_structure_d_handles_synthesis_without_final_answer(monkeypatch):
    """Test fallback when synthesis doesn't have FINAL ANSWER."""
    async def mock_query_model(model, messages):
        content = messages[-1]["content"]
        if "chairman" in content.lower():
            # Synthesis without FINAL ANSWER pattern
            return {"content": "The best answer based on deliberation is forty-two."}
        return {"content": "Model response. FINAL ANSWER: 4"}

    async def mock_query_models_parallel(models, messages):
        return {m: {"content": f"{m}: FINAL ANSWER: 4"} for m in models}

    import backend.governance.structure_d as struct_d_module

    monkeypatch.setattr(struct_d_module, "query_model", mock_query_model)
    monkeypatch.setattr(struct_d_module, "query_models_parallel", mock_query_models_parallel)

    structure = DeliberateSynthesizeStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    # Should fall back to last sentence
    assert result.final_answer is not None
    assert "forty-two" in result.final_answer


def test_structure_d_imports_from_package():
    """Verify DeliberateSynthesizeStructure can be imported from governance package."""
    from backend.governance import DeliberateSynthesizeStructure

    assert DeliberateSynthesizeStructure is not None
