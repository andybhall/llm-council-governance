"""Structure B: Independent → Majority Vote."""

from typing import Dict, List, Optional

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.utils import build_stage1_prompt, extract_final_answer, majority_vote
from backend.openrouter import query_model, query_models_parallel


class MajorityVoteStructure(GovernanceStructure):
    """
    Governance Structure B: Independent → Majority Vote.

    Stage 1: Each LLM answers independently (with FINAL ANSWER instruction)
    Stage 2: Extract final answer from each response
    Stage 3: Majority vote determines winner (chairman as tiebreaker)
    """

    @property
    def name(self) -> str:
        return "Independent → Majority Vote"

    async def run(self, query: str) -> CouncilResult:
        """Execute the voting governance process."""
        # Stage 1: Collect independent responses with FINAL ANSWER instruction
        stage1_responses = await self._stage1_collect_responses(query)

        # Stage 2: Extract answers from each response
        extracted_answers = self._stage2_extract_answers(stage1_responses)

        # Stage 3: Majority vote
        final_answer, chairman_answer = await self._stage3_majority_vote(
            query, extracted_answers
        )

        return CouncilResult(
            final_answer=final_answer,
            stage1_responses=stage1_responses,
            stage2_data={
                "extracted_answers": extracted_answers,
            },
            stage3_data={
                "chairman_tiebreaker": chairman_answer,
                "vote_result": final_answer,
            },
        )

    async def _stage1_collect_responses(self, query: str) -> Dict[str, str]:
        """Stage 1: Query all council models with FINAL ANSWER instruction."""
        prompt = build_stage1_prompt(query)
        messages = [{"role": "user", "content": prompt}]
        results = await query_models_parallel(self.council_models, messages)

        return {
            model: result.get("content", result.get("error", ""))
            for model, result in results.items()
        }

    def _stage2_extract_answers(
        self, stage1_responses: Dict[str, str]
    ) -> Dict[str, Optional[str]]:
        """Stage 2: Extract final answer from each response."""
        return {
            model: extract_final_answer(response)
            for model, response in stage1_responses.items()
        }

    async def _stage3_majority_vote(
        self, query: str, extracted_answers: Dict[str, Optional[str]]
    ) -> tuple[str, Optional[str]]:
        """
        Stage 3: Perform majority vote with chairman as tiebreaker.

        Returns:
            Tuple of (winning answer, chairman's answer for tiebreaker)
        """
        # Filter out None answers
        valid_answers = [
            ans for ans in extracted_answers.values() if ans is not None
        ]

        if not valid_answers:
            # No valid answers extracted, ask chairman directly
            chairman_answer = await self._get_chairman_answer(query)
            return chairman_answer, chairman_answer

        # Get chairman's answer for tiebreaker
        chairman_answer = await self._get_chairman_answer(query)

        # Perform majority vote
        final_answer = majority_vote(valid_answers, tiebreaker=chairman_answer)

        return final_answer, chairman_answer

    async def _get_chairman_answer(self, query: str) -> str:
        """Get chairman's answer for tiebreaker."""
        prompt = build_stage1_prompt(query)
        messages = [{"role": "user", "content": prompt}]
        result = await query_model(self.chairman_model, messages)

        content = result.get("content", "")
        answer = extract_final_answer(content)

        return answer if answer else content.strip()[-100:]
