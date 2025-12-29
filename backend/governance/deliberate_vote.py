"""Structure C: Independent → Deliberate → Vote."""

from typing import Dict, Optional

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.utils import compute_vote_metadata, extract_final_answer
from backend.openrouter import query_model


class DeliberateVoteStructure(GovernanceStructure):
    """
    Governance Structure C: Independent → Deliberate → Vote.

    Stage 1: Each LLM answers independently (with FINAL ANSWER instruction)
    Stage 2: All LLMs see all responses, asked to reconsider (deliberation)
    Stage 3: Extract updated answers, majority vote
    """

    @property
    def name(self) -> str:
        return "Independent → Deliberate → Vote"

    async def run(self, query: str) -> CouncilResult:
        """Execute the deliberation and voting governance process."""
        # Stage 1: Collect independent responses
        stage1_responses = await self._stage1_collect_responses(query)

        # Stage 2: Deliberation - each model sees all responses
        stage2_responses = await self._stage2_deliberate(query, stage1_responses)

        # Stage 3: Extract answers and vote
        extracted_answers = self._extract_answers(stage2_responses)
        final_answer, chairman_answer, vote_metadata = await self._stage3_vote(
            query, extracted_answers
        )

        return CouncilResult(
            final_answer=final_answer,
            stage1_responses=stage1_responses,
            stage2_data={
                "deliberation_responses": stage2_responses,
                "extracted_answers": extracted_answers,
            },
            stage3_data={
                "chairman_tiebreaker": chairman_answer,
                "vote_result": final_answer,
                **vote_metadata,
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

    async def _stage2_deliberate(
        self, query: str, stage1_responses: Dict[str, str]
    ) -> Dict[str, str]:
        """Stage 2: Each model deliberates after seeing all responses."""
        # Build deliberation prompts for each model
        deliberation_tasks = {}
        for model in self.council_models:
            prompt = self._build_deliberation_prompt(
                query=query,
                model=model,
                own_response=stage1_responses[model],
                all_responses=stage1_responses,
            )
            deliberation_tasks[model] = prompt

        # Query all models with their deliberation prompts
        results = await self._query_models_with_different_prompts(deliberation_tasks)

        return {
            model: result.get("content", result.get("error", ""))
            for model, result in results.items()
        }

    async def _query_models_with_different_prompts(
        self, model_prompts: Dict[str, str]
    ) -> Dict[str, Dict]:
        """Query multiple models with different prompts in parallel."""
        import asyncio

        async def query_single(model: str, prompt: str):
            messages = [{"role": "user", "content": prompt}]
            result = await query_model(model, messages)
            return model, result

        tasks = [
            query_single(model, prompt) for model, prompt in model_prompts.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            model: result if not isinstance(result, Exception) else {"error": str(result)}
            for model, result in results
        }

    def _extract_answers(
        self, responses: Dict[str, str]
    ) -> Dict[str, Optional[str]]:
        """Extract final answers from responses."""
        return {
            model: extract_final_answer(response)
            for model, response in responses.items()
        }

    async def _stage3_vote(
        self, query: str, extracted_answers: Dict[str, Optional[str]]
    ) -> tuple[str, Optional[str], Dict]:
        """
        Stage 3: Perform majority vote with chairman as tiebreaker.

        Returns:
            Tuple of (winning answer, chairman's answer for tiebreaker, vote_metadata)
        """
        # Filter out None answers
        valid_answers = [
            ans for ans in extracted_answers.values() if ans is not None
        ]

        if not valid_answers:
            # No valid answers extracted, ask chairman directly
            chairman_answer = await self._get_chairman_answer(query)
            empty_metadata = {
                "raw_answers": {m: a for m, a in extracted_answers.items()},
                "normalized_answers": {m: None for m in extracted_answers.keys()},
                "vote_counts": {},
                "is_tie": False,
                "winning_answer": chairman_answer,
                "tiebreaker_used": False,
            }
            return chairman_answer, chairman_answer, empty_metadata

        # Get chairman's answer for tiebreaker
        chairman_answer = await self._get_chairman_answer(query)

        # Perform majority vote with metadata
        final_answer, vote_metadata = compute_vote_metadata(
            extracted_answers, tiebreaker=chairman_answer
        )

        return final_answer, chairman_answer, vote_metadata
