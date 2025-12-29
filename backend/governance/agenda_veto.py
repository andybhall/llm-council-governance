"""Agenda Setter + Veto: Political economy governance structure."""

import math
from typing import Dict, List, Optional, Any

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.utils import (
    build_stage1_prompt,
    compute_vote_metadata,
    extract_final_answer,
    extract_vote_accept_veto,
)
from backend.openrouter import query_model


class AgendaSetterVetoStructure(GovernanceStructure):
    """
    Agenda Setter + Veto: A political economy governance mechanism.

    This structure models a legislative process with agenda control:
    1. Stage 1 (N calls): Council members answer independently
    2. Stage 2 (1 call): Chairman (agenda setter) proposes an answer
    3. Stage 3 (N calls): Council members vote ACCEPT or VETO on the proposal

    If vetoes >= threshold, the proposal fails and we fall back to majority vote
    on stage 1 answers. Otherwise, the chairman's proposal becomes final.

    Total API calls: 2N + 1 (compute-matched to deliberation structures)
    """

    def __init__(
        self,
        council_models: List[str],
        chairman_model: str,
        veto_threshold: Optional[int] = None,
        fallback_rule: str = "stage1_majority",
    ):
        """
        Initialize agenda setter + veto structure.

        Args:
            council_models: List of model identifiers for council members
            chairman_model: Model identifier for the chairman (agenda setter)
            veto_threshold: Number of vetoes needed to reject proposal.
                           Default: ceil(N/2) = majority
            fallback_rule: Rule to apply if proposal is vetoed.
                          "stage1_majority" (default) uses majority vote on stage 1 answers.
        """
        super().__init__(council_models, chairman_model)
        self.veto_threshold = veto_threshold or math.ceil(len(council_models) / 2)
        self.fallback_rule = fallback_rule

    @property
    def name(self) -> str:
        return "Agenda Setter + Veto"

    async def run(self, query: str) -> CouncilResult:
        """Execute the agenda setter + veto governance process."""
        # Stage 1: Collect independent responses (N calls)
        stage1_responses = await self._stage1_collect_responses(query)
        stage1_extracted = {
            model: extract_final_answer(resp)
            for model, resp in stage1_responses.items()
        }

        # Stage 2: Chairman proposes answer (1 call)
        chair_proposal, chair_response = await self._stage2_propose(
            query, stage1_responses
        )

        # Stage 3: Council votes on proposal (N calls)
        votes, vote_responses = await self._stage3_vote_on_proposal(
            query, chair_proposal, stage1_responses
        )

        # Determine outcome
        veto_count = sum(1 for v in votes.values() if v == "VETO")
        proposal_passed = veto_count < self.veto_threshold

        if proposal_passed:
            final_answer = chair_proposal
            fallback_used = False
            fallback_result = None
        else:
            # Apply fallback rule
            final_answer, fallback_result = self._apply_fallback(stage1_extracted)
            fallback_used = True

        return CouncilResult(
            final_answer=final_answer,
            stage1_responses=stage1_responses,
            stage2_data={
                "chair_proposal": chair_proposal,
                "chair_response": chair_response,
                "extracted_answers": stage1_extracted,
            },
            stage3_data={
                "votes": votes,
                "vote_responses": vote_responses,
                "veto_count": veto_count,
                "veto_threshold": self.veto_threshold,
                "proposal_passed": proposal_passed,
                "fallback_used": fallback_used,
                "fallback_rule": self.fallback_rule,
                "fallback_result": fallback_result,
            },
        )

    async def _stage2_propose(
        self, query: str, stage1_responses: Dict[str, str]
    ) -> tuple[str, str]:
        """
        Stage 2: Chairman proposes an answer after seeing all stage 1 responses.

        Returns:
            Tuple of (extracted proposal, full response text)
        """
        # Format stage 1 responses for the chairman
        responses_text = "\n\n".join(
            f"Council Member {i+1}:\n{response}"
            for i, response in enumerate(stage1_responses.values())
        )

        prompt = f"""You are the chairman of a council that has been asked:

{query}

The council members have provided their initial answers:

{responses_text}

As chairman, you must now propose a single final answer for the council to vote on.
Consider all the council members' reasoning and propose the best answer.

End your response with:
FINAL ANSWER: [your proposed answer]"""

        messages = [{"role": "user", "content": prompt}]
        result = await query_model(self.chairman_model, messages)
        response_text = result.get("content", "")
        proposal = extract_final_answer(response_text) or ""

        return proposal, response_text

    async def _stage3_vote_on_proposal(
        self, query: str, proposal: str, stage1_responses: Dict[str, str]
    ) -> tuple[Dict[str, Optional[str]], Dict[str, str]]:
        """
        Stage 3: Council members vote ACCEPT or VETO on the chairman's proposal.

        Returns:
            Tuple of (votes dict, full responses dict)
        """
        import asyncio

        # Format anonymized stage 1 answers for context
        answers_summary = "\n".join(
            f"- Member {i+1}: {extract_final_answer(resp) or '[no clear answer]'}"
            for i, resp in enumerate(stage1_responses.values())
        )

        async def vote_single(model: str, own_response: str) -> tuple[str, str, str]:
            own_answer = extract_final_answer(own_response) or "[no clear answer]"

            prompt = f"""The council was asked:

{query}

Your original answer was: {own_answer}

The chairman has proposed this final answer: {proposal}

For reference, here are the council members' original answers:
{answers_summary}

You must now vote on the chairman's proposal.
If you accept it, respond with: FINAL VOTE: ACCEPT
If you reject it, respond with: FINAL VOTE: VETO

Provide brief reasoning, then state your vote."""

            messages = [{"role": "user", "content": prompt}]
            result = await query_model(model, messages)
            response_text = result.get("content", "")
            vote = extract_vote_accept_veto(response_text)

            return model, vote, response_text

        # Query all council members in parallel
        tasks = [
            vote_single(model, stage1_responses[model])
            for model in self.council_models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes = {}
        vote_responses = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            model, vote, response_text = result
            votes[model] = vote
            vote_responses[model] = response_text

        return votes, vote_responses

    def _apply_fallback(
        self, stage1_extracted: Dict[str, Optional[str]]
    ) -> tuple[str, Dict[str, Any]]:
        """
        Apply fallback rule when proposal is vetoed.

        Returns:
            Tuple of (final answer, fallback metadata)
        """
        if self.fallback_rule == "stage1_majority":
            final_answer, metadata = compute_vote_metadata(stage1_extracted)
            return final_answer, metadata
        else:
            # Unknown fallback rule - default to stage1 majority
            final_answer, metadata = compute_vote_metadata(stage1_extracted)
            return final_answer, metadata
