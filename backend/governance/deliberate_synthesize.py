"""Structure D: Independent → Deliberate → Synthesize."""

from typing import Dict, Tuple

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.utils import extract_final_answer_with_fallback
from backend.openrouter import query_model


class DeliberateSynthesizeStructure(GovernanceStructure):
    """
    Governance Structure D: Independent → Deliberate → Synthesize.

    Stage 1: Each LLM answers independently
    Stage 2: All LLMs see all responses, asked to reconsider (deliberation)
    Stage 3: Chairman synthesizes final answer based on deliberated responses
    """

    @property
    def name(self) -> str:
        return "Independent → Deliberate → Synthesize"

    async def run(self, query: str) -> CouncilResult:
        """Execute the deliberation and synthesis governance process."""
        # Stage 1: Collect independent responses
        stage1_responses = await self._stage1_collect_responses(query)

        # Stage 2: Deliberation - each model sees all responses
        stage2_responses = await self._stage2_deliberate(query, stage1_responses)

        # Stage 3: Chairman synthesizes from deliberated responses
        synthesis, final_answer = await self._stage3_synthesize(
            query, stage1_responses, stage2_responses
        )

        return CouncilResult(
            final_answer=final_answer,
            stage1_responses=stage1_responses,
            stage2_data={
                "deliberation_responses": stage2_responses,
            },
            stage3_data={
                "synthesis": synthesis,
            },
        )

    def _build_deliberation_prompt(
        self,
        query: str,
        model: str,
        own_response: str,
        all_responses: Dict[str, str],
    ) -> str:
        """Build deliberation prompt for a model to reconsider."""
        # Format other responses (excluding own)
        other_responses_text = "\n\n".join(
            f"Council Member {i+1}:\n{response}"
            for i, (m, response) in enumerate(all_responses.items())
            if m != model
        )

        return f"""You previously answered the following question:

{query}

Your original response was:
{own_response}

Here are the responses from other council members:

{other_responses_text}

Consider their reasoning. You may revise your answer or maintain your original position.
Provide your (possibly updated) response, ending with:
FINAL ANSWER: [your answer]"""

    def _build_synthesis_prompt(
        self,
        query: str,
        stage1_responses: Dict[str, str],
        stage2_responses: Dict[str, str],
    ) -> str:
        """Build synthesis prompt for chairman."""
        # Format initial responses
        initial_text = "\n\n".join(
            f"Council Member {i+1} (Initial):\n{response}"
            for i, response in enumerate(stage1_responses.values())
        )

        # Format deliberated responses
        deliberated_text = "\n\n".join(
            f"Council Member {i+1} (After Deliberation):\n{response}"
            for i, response in enumerate(stage2_responses.values())
        )

        return f"""You are the chairman of a council of AI models. The council was asked:

{query}

The council members first provided independent responses:

{initial_text}

After seeing each other's responses, they deliberated and provided updated answers:

{deliberated_text}

Based on the council's initial responses and their deliberated positions, synthesize the best final answer. Pay special attention to:
- Points of agreement after deliberation
- Strong arguments that convinced others to change positions
- Areas where the council remains divided

End your response with:
FINAL ANSWER: [your synthesized answer]"""

    async def _stage2_deliberate(
        self, query: str, stage1_responses: Dict[str, str]
    ) -> Dict[str, str]:
        """Stage 2: Each model deliberates after seeing all responses."""
        import asyncio

        async def query_single(model: str, prompt: str):
            messages = [{"role": "user", "content": prompt}]
            result = await query_model(model, messages)
            return model, result

        # Build deliberation prompts for each model
        tasks = []
        for model in self.council_models:
            prompt = self._build_deliberation_prompt(
                query=query,
                model=model,
                own_response=stage1_responses[model],
                all_responses=stage1_responses,
            )
            tasks.append(query_single(model, prompt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            model: (
                result.get("content", result.get("error", ""))
                if not isinstance(result, Exception)
                else str(result)
            )
            for model, result in results
        }

    async def _stage3_synthesize(
        self,
        query: str,
        stage1_responses: Dict[str, str],
        stage2_responses: Dict[str, str],
    ) -> Tuple[str, str]:
        """
        Stage 3: Chairman synthesizes final answer.

        Returns:
            Tuple of (full synthesis response, extracted final answer)
        """
        prompt = self._build_synthesis_prompt(query, stage1_responses, stage2_responses)
        messages = [{"role": "user", "content": prompt}]
        result = await query_model(self.chairman_model, messages)

        synthesis = result.get("content", "")
        final_answer = extract_final_answer_with_fallback(synthesis)

        return synthesis, final_answer
