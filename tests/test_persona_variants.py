"""Tests for persona variant council experiment."""

import pytest

from backend.persona_variants import (
    BASE_MODEL,
    PERSONA_MODELS,
    PERSONA_TRANSFORMERS,
    is_persona_model,
    get_actual_model,
    transform_prompt_for_persona,
    transform_mathematician,
    transform_scientist,
    transform_engineer,
    transform_teacher,
    query_model_with_personas,
    query_models_parallel_with_personas,
)


class TestPersonaDefinitions:
    """Test persona definitions and constants."""

    def test_persona_models_count(self):
        """Verify we have exactly 4 personas."""
        assert len(PERSONA_MODELS) == 4

    def test_all_personas_have_transformers(self):
        """Verify each persona model has a transformer function."""
        for model in PERSONA_MODELS:
            assert model in PERSONA_TRANSFORMERS

    def test_base_model_defined(self):
        """Verify base model is google/gemma-2-9b-it."""
        assert BASE_MODEL == "google/gemma-2-9b-it"

    def test_persona_models_have_prefix(self):
        """Verify persona models have consistent naming."""
        for model in PERSONA_MODELS:
            assert model.startswith("persona/")


class TestPersonaTransformations:
    """Test individual persona transformation functions."""

    def test_mathematician_adds_persona(self):
        """Verify mathematician persona is applied."""
        original = "What is 2+2?"
        transformed = transform_mathematician(original)

        assert "rigorous mathematician" in transformed.lower()
        assert "proof" in transformed.lower()
        assert original in transformed

    def test_scientist_adds_persona(self):
        """Verify scientist persona is applied."""
        original = "What is 2+2?"
        transformed = transform_scientist(original)

        assert "skeptical scientist" in transformed.lower()
        assert "question" in transformed.lower()
        assert original in transformed

    def test_engineer_adds_persona(self):
        """Verify engineer persona is applied."""
        original = "What is 2+2?"
        transformed = transform_engineer(original)

        assert "practical engineer" in transformed.lower()
        assert "sanity" in transformed.lower()
        assert original in transformed

    def test_teacher_adds_persona(self):
        """Verify teacher persona is applied."""
        original = "What is 2+2?"
        transformed = transform_teacher(original)

        assert "enthusiastic teacher" in transformed.lower()
        assert "explain" in transformed.lower()
        assert original in transformed

    def test_transformations_preserve_original(self):
        """All transformations should preserve original prompt."""
        original = "Complex question with special chars: $%^&"

        for transformer in PERSONA_TRANSFORMERS.values():
            transformed = transformer(original)
            assert original in transformed

    def test_personas_are_distinct(self):
        """Each persona should produce different output."""
        original = "Test question"
        results = [t(original) for t in PERSONA_TRANSFORMERS.values()]

        # All should be different from each other
        assert len(set(results)) == 4


class TestPersonaDetection:
    """Test pseudo-model detection and routing."""

    def test_is_persona_model_true(self):
        """Verify persona models are detected."""
        for model in PERSONA_MODELS:
            assert is_persona_model(model) is True

    def test_is_persona_model_false_for_regular(self):
        """Verify regular models are not detected as personas."""
        assert is_persona_model("google/gemma-2-9b-it") is False
        assert is_persona_model("openai/gpt-4") is False
        assert is_persona_model("prompt-variant/step-by-step") is False

    def test_get_actual_model_for_persona(self):
        """Persona models should route to BASE_MODEL."""
        for model in PERSONA_MODELS:
            assert get_actual_model(model) == BASE_MODEL

    def test_get_actual_model_passthrough(self):
        """Regular models should pass through unchanged."""
        assert get_actual_model("openai/gpt-4") == "openai/gpt-4"
        assert get_actual_model("google/gemma-2-9b-it") == "google/gemma-2-9b-it"


class TestTransformPromptForPersona:
    """Test the transform_prompt_for_persona function."""

    def test_transforms_persona_models(self):
        """Persona models should have prompts transformed."""
        prompt = "Original prompt"

        for model in PERSONA_MODELS:
            transformed = transform_prompt_for_persona(model, prompt)
            assert transformed != prompt
            assert prompt in transformed

    def test_passthrough_regular_models(self):
        """Regular models should not have prompts transformed."""
        prompt = "Original prompt"
        result = transform_prompt_for_persona("openai/gpt-4", prompt)
        assert result == prompt


class TestQueryModelWithPersonas:
    """Test query_model_with_personas function."""

    @pytest.fixture
    def mock_query_model(self, monkeypatch):
        """Mock the original query_model function."""
        call_log = []

        async def mock_fn(model, messages):
            call_log.append({"model": model, "messages": messages})
            return {"content": "Test response. FINAL ANSWER: 42"}

        import backend.persona_variants as pv
        monkeypatch.setattr(pv, "original_query_model", mock_fn)

        return call_log

    @pytest.mark.asyncio
    async def test_persona_model_transforms_prompt(self, mock_query_model):
        """Persona models should have their prompts transformed."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        await query_model_with_personas(
            "persona/mathematician",
            messages
        )

        # Should call with BASE_MODEL
        assert mock_query_model[0]["model"] == BASE_MODEL

        # Should have transformed prompt with persona
        transformed_content = mock_query_model[0]["messages"][0]["content"]
        assert "rigorous mathematician" in transformed_content.lower()
        assert "What is 2+2?" in transformed_content

    @pytest.mark.asyncio
    async def test_regular_model_passthrough(self, mock_query_model):
        """Regular models should pass through unchanged."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        await query_model_with_personas("openai/gpt-4", messages)

        assert mock_query_model[0]["model"] == "openai/gpt-4"
        assert mock_query_model[0]["messages"][0]["content"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_preserves_system_messages(self, mock_query_model):
        """System messages should not be transformed."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        await query_model_with_personas(
            "persona/scientist",
            messages
        )

        transformed_messages = mock_query_model[0]["messages"]

        # System message unchanged
        assert transformed_messages[0]["content"] == "You are helpful."
        # User message transformed with persona
        assert "skeptical scientist" in transformed_messages[1]["content"].lower()


class TestQueryModelsParallelWithPersonas:
    """Test query_models_parallel_with_personas function."""

    @pytest.fixture
    def mock_query_model(self, monkeypatch):
        """Mock the original query_model function."""
        call_log = []

        async def mock_fn(model, messages):
            call_log.append({"model": model, "messages": messages})
            return {"content": f"Response from {model}. FINAL ANSWER: 42"}

        import backend.persona_variants as pv
        monkeypatch.setattr(pv, "original_query_model", mock_fn)

        return call_log

    @pytest.mark.asyncio
    async def test_queries_all_persona_models(self, mock_query_model):
        """Should query all persona models in parallel."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        results = await query_models_parallel_with_personas(
            PERSONA_MODELS,
            messages
        )

        assert len(results) == 4
        for model in PERSONA_MODELS:
            assert model in results
            assert "content" in results[model]

    @pytest.mark.asyncio
    async def test_all_route_to_base_model(self, mock_query_model):
        """All persona models should route to the same base model."""
        messages = [{"role": "user", "content": "Test"}]

        await query_models_parallel_with_personas(PERSONA_MODELS, messages)

        # All calls should be to BASE_MODEL
        for call in mock_query_model:
            assert call["model"] == BASE_MODEL

    @pytest.mark.asyncio
    async def test_different_personas_applied(self, mock_query_model):
        """Each persona model should get its specific persona."""
        messages = [{"role": "user", "content": "Test question"}]

        await query_models_parallel_with_personas(PERSONA_MODELS, messages)

        # Collect all transformed prompts
        prompts = [call["messages"][0]["content"] for call in mock_query_model]

        # Each should be different (different personas applied)
        assert len(set(prompts)) == 4

        # Verify specific personas
        prompt_texts = " ".join(prompts).lower()
        assert "mathematician" in prompt_texts
        assert "scientist" in prompt_texts
        assert "engineer" in prompt_texts
        assert "teacher" in prompt_texts

    @pytest.mark.asyncio
    async def test_handles_exceptions(self, monkeypatch):
        """Should handle exceptions gracefully."""
        async def failing_query(model, messages):
            raise Exception("API Error")

        import backend.persona_variants as pv
        monkeypatch.setattr(pv, "original_query_model", failing_query)

        messages = [{"role": "user", "content": "Test"}]

        results = await query_models_parallel_with_personas(
            PERSONA_MODELS,
            messages
        )

        # All should have error keys
        assert all("error" in r for r in results.values())


class TestIntegrationWithGovernanceStructures:
    """Integration tests with governance structures."""

    @pytest.mark.asyncio
    async def test_structure_b_with_personas(self, monkeypatch):
        """Test MajorityVoteStructure works with persona models."""
        call_log = []

        async def mock_query(model, messages):
            call_log.append({"model": model})
            return {"content": "Response. FINAL ANSWER: 42"}

        async def mock_parallel(models, messages):
            results = {}
            for model in models:
                results[model] = await mock_query(model, messages)
            return results

        # Patch at the persona_variants level
        import backend.persona_variants as pv
        monkeypatch.setattr(pv, "original_query_model", mock_query)

        # Patch the structure's query functions
        import backend.governance.structure_b as struct_b
        from backend.persona_variants import (
            query_model_with_personas,
            query_models_parallel_with_personas
        )
        monkeypatch.setattr(struct_b, "query_model", query_model_with_personas)
        monkeypatch.setattr(struct_b, "query_models_parallel", query_models_parallel_with_personas)

        from backend.governance.structure_b import MajorityVoteStructure

        structure = MajorityVoteStructure(
            council_models=PERSONA_MODELS,
            chairman_model=PERSONA_MODELS[0],
        )

        result = await structure.run("What is 2+2?")

        assert result.final_answer is not None
        assert len(result.stage1_responses) == 4


def test_persona_models_importable_from_module():
    """Verify persona variants can be imported."""
    from backend.persona_variants import (
        PERSONA_MODELS,
        BASE_MODEL,
        query_model_with_personas,
        query_models_parallel_with_personas,
    )

    assert PERSONA_MODELS is not None
    assert BASE_MODEL is not None
    assert query_model_with_personas is not None
    assert query_models_parallel_with_personas is not None
