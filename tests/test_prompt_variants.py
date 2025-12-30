"""Tests for prompt variant council experiment."""

import pytest

from backend.prompt_variants import (
    BASE_MODEL,
    VARIANT_MODELS,
    VARIANT_TRANSFORMERS,
    is_prompt_variant_model,
    get_actual_model,
    transform_prompt_for_variant,
    transform_step_by_step,
    transform_identify_then_solve,
    transform_skeptical_verifier,
    transform_example_based,
    query_model_with_variants,
    query_models_parallel_with_variants,
)


class TestPromptVariantDefinitions:
    """Test prompt variant definitions and constants."""

    def test_variant_models_count(self):
        """Verify we have exactly 4 prompt variants."""
        assert len(VARIANT_MODELS) == 4

    def test_all_variants_have_transformers(self):
        """Verify each variant model has a transformer function."""
        for model in VARIANT_MODELS:
            assert model in VARIANT_TRANSFORMERS

    def test_base_model_defined(self):
        """Verify base model is google/gemma-2-9b-it."""
        assert BASE_MODEL == "google/gemma-2-9b-it"

    def test_variant_models_have_prefix(self):
        """Verify variant models have consistent naming."""
        for model in VARIANT_MODELS:
            assert model.startswith("prompt-variant/")


class TestPromptTransformations:
    """Test individual prompt transformation functions."""

    def test_step_by_step_adds_prefix(self):
        """Verify step-by-step adds appropriate prefix."""
        original = "What is 2+2?"
        transformed = transform_step_by_step(original)

        assert "step by step" in transformed.lower()
        assert original in transformed

    def test_identify_then_solve_adds_prefix(self):
        """Verify identify-then-solve adds appropriate prefix."""
        original = "What is 2+2?"
        transformed = transform_identify_then_solve(original)

        assert "identify" in transformed.lower()
        assert "key information" in transformed.lower()
        assert original in transformed

    def test_skeptical_verifier_adds_prefix(self):
        """Verify skeptical verifier adds appropriate prefix."""
        original = "What is 2+2?"
        transformed = transform_skeptical_verifier(original)

        assert "mistake" in transformed.lower()
        assert "verify" in transformed.lower()
        assert original in transformed

    def test_example_based_adds_prefix(self):
        """Verify example-based adds appropriate prefix."""
        original = "What is 2+2?"
        transformed = transform_example_based(original)

        assert "similar" in transformed.lower()
        assert original in transformed

    def test_transformations_preserve_original(self):
        """All transformations should preserve original prompt."""
        original = "Complex question with special chars: $%^&"

        for transformer in VARIANT_TRANSFORMERS.values():
            transformed = transformer(original)
            assert original in transformed


class TestVariantDetection:
    """Test pseudo-model detection and routing."""

    def test_is_prompt_variant_model_true(self):
        """Verify variant models are detected."""
        for model in VARIANT_MODELS:
            assert is_prompt_variant_model(model) is True

    def test_is_prompt_variant_model_false_for_regular(self):
        """Verify regular models are not detected as variants."""
        assert is_prompt_variant_model("google/gemma-2-9b-it") is False
        assert is_prompt_variant_model("openai/gpt-4") is False
        assert is_prompt_variant_model("anthropic/claude-3") is False

    def test_get_actual_model_for_variant(self):
        """Variant models should route to BASE_MODEL."""
        for model in VARIANT_MODELS:
            assert get_actual_model(model) == BASE_MODEL

    def test_get_actual_model_passthrough(self):
        """Regular models should pass through unchanged."""
        assert get_actual_model("openai/gpt-4") == "openai/gpt-4"
        assert get_actual_model("google/gemma-2-9b-it") == "google/gemma-2-9b-it"


class TestTransformPromptForVariant:
    """Test the transform_prompt_for_variant function."""

    def test_transforms_variant_models(self):
        """Variant models should have prompts transformed."""
        prompt = "Original prompt"

        for model in VARIANT_MODELS:
            transformed = transform_prompt_for_variant(model, prompt)
            assert transformed != prompt
            assert prompt in transformed

    def test_passthrough_regular_models(self):
        """Regular models should not have prompts transformed."""
        prompt = "Original prompt"
        result = transform_prompt_for_variant("openai/gpt-4", prompt)
        assert result == prompt


class TestQueryModelWithVariants:
    """Test query_model_with_variants function."""

    @pytest.fixture
    def mock_query_model(self, monkeypatch):
        """Mock the original query_model function."""
        call_log = []

        async def mock_fn(model, messages, temperature=0.0, timeout=None):
            call_log.append({"model": model, "messages": messages})
            return {"content": "Test response. FINAL ANSWER: 42"}

        # Patch at the actual implementation location
        import experiments.assets.prompt_variants as pv
        monkeypatch.setattr(pv, "original_query_model", mock_fn)

        return call_log

    @pytest.mark.asyncio
    async def test_variant_model_transforms_prompt(self, mock_query_model):
        """Variant models should have their prompts transformed."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        await query_model_with_variants(
            "prompt-variant/step-by-step",
            messages
        )

        # Should call with BASE_MODEL
        assert mock_query_model[0]["model"] == BASE_MODEL

        # Should have transformed prompt
        transformed_content = mock_query_model[0]["messages"][0]["content"]
        assert "step by step" in transformed_content.lower()
        assert "What is 2+2?" in transformed_content

    @pytest.mark.asyncio
    async def test_regular_model_passthrough(self, mock_query_model):
        """Regular models should pass through unchanged."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        await query_model_with_variants("openai/gpt-4", messages)

        assert mock_query_model[0]["model"] == "openai/gpt-4"
        assert mock_query_model[0]["messages"][0]["content"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_preserves_system_messages(self, mock_query_model):
        """System messages should not be transformed."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        await query_model_with_variants(
            "prompt-variant/step-by-step",
            messages
        )

        transformed_messages = mock_query_model[0]["messages"]

        # System message unchanged
        assert transformed_messages[0]["content"] == "You are helpful."
        # User message transformed
        assert "step by step" in transformed_messages[1]["content"].lower()


class TestQueryModelsParallelWithVariants:
    """Test query_models_parallel_with_variants function."""

    @pytest.fixture
    def mock_query_model(self, monkeypatch):
        """Mock the original query_model function."""
        call_log = []

        async def mock_fn(model, messages, temperature=0.0, timeout=None):
            call_log.append({"model": model, "messages": messages})
            return {"content": f"Response from {model}. FINAL ANSWER: 42"}

        # Patch at the actual implementation location
        import experiments.assets.prompt_variants as pv
        monkeypatch.setattr(pv, "original_query_model", mock_fn)

        return call_log

    @pytest.mark.asyncio
    async def test_queries_all_variant_models(self, mock_query_model):
        """Should query all variant models in parallel."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        results = await query_models_parallel_with_variants(
            VARIANT_MODELS,
            messages
        )

        assert len(results) == 4
        for model in VARIANT_MODELS:
            assert model in results
            assert "content" in results[model]

    @pytest.mark.asyncio
    async def test_all_route_to_base_model(self, mock_query_model):
        """All variant models should route to the same base model."""
        messages = [{"role": "user", "content": "Test"}]

        await query_models_parallel_with_variants(VARIANT_MODELS, messages)

        # All calls should be to BASE_MODEL
        for call in mock_query_model:
            assert call["model"] == BASE_MODEL

    @pytest.mark.asyncio
    async def test_different_transformations_applied(self, mock_query_model):
        """Each variant model should get its specific transformation."""
        messages = [{"role": "user", "content": "Test question"}]

        await query_models_parallel_with_variants(VARIANT_MODELS, messages)

        # Collect all transformed prompts
        prompts = [call["messages"][0]["content"] for call in mock_query_model]

        # Each should be different (different transformations applied)
        assert len(set(prompts)) == 4

        # Verify specific transformations
        prompt_texts = " ".join(prompts).lower()
        assert "step by step" in prompt_texts
        assert "identify" in prompt_texts
        assert "mistake" in prompt_texts
        assert "similar" in prompt_texts

    @pytest.mark.asyncio
    async def test_handles_exceptions(self, monkeypatch):
        """Should handle exceptions gracefully."""
        async def failing_query(model, messages, temperature=0.0, timeout=None):
            raise Exception("API Error")

        # Patch at the actual implementation location
        import experiments.assets.prompt_variants as pv
        monkeypatch.setattr(pv, "original_query_model", failing_query)

        messages = [{"role": "user", "content": "Test"}]

        results = await query_models_parallel_with_variants(
            VARIANT_MODELS,
            messages
        )

        # All should have error keys
        assert all("error" in r for r in results.values())


class TestIntegrationWithGovernanceStructures:
    """Integration tests with governance structures."""

    @pytest.mark.asyncio
    async def test_structure_b_with_variants(self, monkeypatch):
        """Test MajorityVoteStructure works with variant models."""
        import importlib

        call_log = []

        async def mock_query(model, messages, temperature=0.0, timeout=None):
            call_log.append({"model": model})
            return {"content": "Response. FINAL ANSWER: 42"}

        async def mock_parallel(models, messages, temperature=0.0, timeout=None):
            results = {}
            for model in models:
                results[model] = await mock_query(model, messages)
            return results

        # Patch at the actual implementation location
        import experiments.assets.prompt_variants as pv
        monkeypatch.setattr(pv, "original_query_model", mock_query)

        # Patch the structure's query functions (use importlib to avoid namespace collision)
        base_module = importlib.import_module("backend.governance.base")
        from backend.prompt_variants import (
            query_model_with_variants,
            query_models_parallel_with_variants
        )
        # Both query_model and query_models_parallel are now in base class
        monkeypatch.setattr(base_module, "query_model", query_model_with_variants)
        monkeypatch.setattr(base_module, "query_models_parallel", query_models_parallel_with_variants)

        from backend.governance import MajorityVoteStructure

        structure = MajorityVoteStructure(
            council_models=VARIANT_MODELS,
            chairman_model=VARIANT_MODELS[0],
        )

        result = await structure.run("What is 2+2?")

        assert result.final_answer is not None
        assert len(result.stage1_responses) == 4


def test_variant_models_importable_from_module():
    """Verify prompt variants can be imported."""
    from backend.prompt_variants import (
        VARIANT_MODELS,
        BASE_MODEL,
        query_model_with_variants,
        query_models_parallel_with_variants,
    )

    assert VARIANT_MODELS is not None
    assert BASE_MODEL is not None
    assert query_model_with_variants is not None
    assert query_models_parallel_with_variants is not None
