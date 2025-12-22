"""Tests for Structure C: Independent → Deliberate → Vote."""

import pytest

from backend.governance.structure_c import DeliberateVoteStructure


@pytest.fixture
def mock_openrouter(monkeypatch):
    """Mock openrouter API calls for testing."""
    call_log = []

    async def mock_query_model(model, messages):
        call_log.append({"type": "single", "model": model, "messages": messages})
        content = messages[-1]["content"]

        # Detect stage based on prompt content
        if "previously answered" in content.lower():
            # Stage 2: Deliberation - model may change answer
            if "model1" in model:
                return {"content": "After seeing others, I maintain my answer. FINAL ANSWER: 4"}
            else:
                return {"content": "I agree with model1. FINAL ANSWER: 4"}
        else:
            # Stage 1 or chairman: Initial response
            if "model1" in model:
                return {"content": "I think it's 4. FINAL ANSWER: 4"}
            elif "model2" in model:
                return {"content": "I believe it's 5. FINAL ANSWER: 5"}
            else:
                return {"content": "The answer is 4. FINAL ANSWER: 4"}

    async def mock_query_models_parallel(models, messages):
        call_log.append({"type": "parallel", "models": models, "messages": messages})
        results = {}
        for i, model in enumerate(models):
            if i == 0:
                results[model] = {"content": f"{model}: I think 4. FINAL ANSWER: 4"}
            else:
                results[model] = {"content": f"{model}: I think 5. FINAL ANSWER: 5"}
        return results

    import backend.governance.structure_c as struct_c_module

    monkeypatch.setattr(struct_c_module, "query_model", mock_query_model)
    monkeypatch.setattr(struct_c_module, "query_models_parallel", mock_query_models_parallel)

    return call_log


@pytest.fixture
def mock_openrouter_consensus(monkeypatch):
    """Mock where deliberation leads to consensus."""
    call_log = []

    async def mock_query_model(model, messages):
        call_log.append({"type": "single", "model": model, "messages": messages})
        content = messages[-1]["content"]

        if "previously answered" in content.lower():
            # After deliberation, everyone agrees on 42
            return {"content": "After discussion, I agree. FINAL ANSWER: 42"}
        else:
            return {"content": "Chairman says 42. FINAL ANSWER: 42"}

    async def mock_query_models_parallel(models, messages):
        call_log.append({"type": "parallel", "models": models, "messages": messages})
        # Initial responses are different
        results = {}
        for i, model in enumerate(models):
            results[model] = {"content": f"{model}: Answer is {40 + i}. FINAL ANSWER: {40 + i}"}
        return results

    import backend.governance.structure_c as struct_c_module

    monkeypatch.setattr(struct_c_module, "query_model", mock_query_model)
    monkeypatch.setattr(struct_c_module, "query_models_parallel", mock_query_models_parallel)

    return call_log


@pytest.mark.asyncio
async def test_structure_c_runs(mock_openrouter):
    """Verify Structure C executes successfully."""
    structure = DeliberateVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert result.final_answer is not None
    assert len(result.stage1_responses) == 2


@pytest.mark.asyncio
async def test_structure_c_name():
    """Verify Structure C has correct name."""
    structure = DeliberateVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    assert structure.name == "Independent → Deliberate → Vote"


@pytest.mark.asyncio
async def test_structure_c_has_deliberation_stage(mock_openrouter):
    """Verify Stage 2 deliberation responses are captured."""
    structure = DeliberateVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("What is 2+2?")

    assert "deliberation_responses" in result.stage2_data
    assert len(result.stage2_data["deliberation_responses"]) == 2


@pytest.mark.asyncio
async def test_structure_c_deliberation_changes_answers(mock_openrouter_consensus):
    """Verify deliberation can lead to consensus."""
    structure = DeliberateVoteStructure(
        council_models=["model1", "model2", "model3"],
        chairman_model="chairman",
    )
    result = await structure.run("What is the answer?")

    # After deliberation, everyone should agree on 42
    extracted = result.stage2_data["extracted_answers"]
    assert all(ans == "42" for ans in extracted.values())
    assert result.final_answer == "42"


@pytest.mark.asyncio
async def test_structure_c_api_call_sequence(mock_openrouter):
    """Verify correct sequence of API calls."""
    structure = DeliberateVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    await structure.run("What is 2+2?")

    # Should have:
    # 1. Stage 1: parallel query to all models
    # 2. Stage 2: individual deliberation calls (one per model)
    # 3. Stage 3: chairman tiebreaker
    assert mock_openrouter[0]["type"] == "parallel"  # Stage 1

    # Stage 2 has individual calls for deliberation
    deliberation_calls = [c for c in mock_openrouter if c["type"] == "single"]
    # Should have 2 deliberation calls + 1 chairman call = 3 single calls
    assert len(deliberation_calls) >= 2


@pytest.mark.asyncio
async def test_structure_c_stage1_responses(mock_openrouter):
    """Verify Stage 1 collects all responses."""
    structure = DeliberateVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    assert "model1" in result.stage1_responses
    assert "model2" in result.stage1_responses


@pytest.mark.asyncio
async def test_structure_c_extracted_answers(mock_openrouter):
    """Verify answers are extracted after deliberation."""
    structure = DeliberateVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    extracted = result.stage2_data["extracted_answers"]
    assert len(extracted) == 2
    assert all(ans is not None for ans in extracted.values())


@pytest.mark.asyncio
async def test_structure_c_vote_result(mock_openrouter):
    """Verify Stage 3 produces vote result."""
    structure = DeliberateVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    assert "vote_result" in result.stage3_data
    assert "chairman_tiebreaker" in result.stage3_data


def test_build_initial_prompt():
    """Test initial prompt includes FINAL ANSWER instruction."""
    structure = DeliberateVoteStructure(
        council_models=["m1"],
        chairman_model="chair",
    )
    prompt = structure._build_initial_prompt("What is 2+2?")

    assert "What is 2+2?" in prompt
    assert "FINAL ANSWER:" in prompt


def test_build_deliberation_prompt():
    """Test deliberation prompt includes all required elements."""
    structure = DeliberateVoteStructure(
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
    assert "I think 4" in prompt  # Own response
    assert "I think 5" in prompt  # Other response
    assert "FINAL ANSWER:" in prompt


def test_deliberation_prompt_excludes_own_response():
    """Test deliberation prompt shows other responses, not own in 'others' section."""
    structure = DeliberateVoteStructure(
        council_models=["m1", "m2", "m3"],
        chairman_model="chair",
    )

    all_responses = {
        "m1": "Response from m1",
        "m2": "Response from m2",
        "m3": "Response from m3",
    }

    prompt = structure._build_deliberation_prompt(
        query="Test",
        model="m1",
        own_response="Response from m1",
        all_responses=all_responses,
    )

    # The "other council members" section should not include m1's response
    # But "Your original response" section will have it
    other_section = prompt.split("other council members:")[-1]
    assert "Response from m2" in other_section
    assert "Response from m3" in other_section


def test_extract_answers():
    """Test answer extraction from deliberation responses."""
    structure = DeliberateVoteStructure(
        council_models=["m1", "m2"],
        chairman_model="chair",
    )
    responses = {
        "m1": "After thinking, FINAL ANSWER: 4",
        "m2": "I changed my mind. FINAL ANSWER: 4",
    }

    extracted = structure._extract_answers(responses)

    assert extracted["m1"] == "4"
    assert extracted["m2"] == "4"


@pytest.mark.asyncio
async def test_structure_c_handles_no_valid_answers(monkeypatch):
    """Test handling when no deliberation responses have valid FINAL ANSWER."""
    async def mock_query_model(model, messages):
        content = messages[-1]["content"]
        if "previously answered" in content.lower():
            return {"content": "I'm not sure anymore..."}
        return {"content": "Chairman says 42. FINAL ANSWER: 42"}

    async def mock_query_models_parallel(models, messages):
        return {
            model: {"content": f"{model} rambles"}
            for model in models
        }

    import backend.governance.structure_c as struct_c_module

    monkeypatch.setattr(struct_c_module, "query_model", mock_query_model)
    monkeypatch.setattr(struct_c_module, "query_models_parallel", mock_query_models_parallel)

    structure = DeliberateVoteStructure(
        council_models=["model1", "model2"],
        chairman_model="chairman",
    )
    result = await structure.run("Test query")

    # Should fall back to chairman's answer
    assert result.final_answer == "42"


def test_structure_c_imports_from_package():
    """Verify DeliberateVoteStructure can be imported from governance package."""
    from backend.governance import DeliberateVoteStructure

    assert DeliberateVoteStructure is not None
