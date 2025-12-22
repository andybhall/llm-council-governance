"""Structure A: Independent → Rank → Synthesize (Karpathy Baseline)."""

import re
from typing import Dict, List, Tuple

from backend.governance.base import CouncilResult, GovernanceStructure
from backend.governance.utils import build_stage1_prompt, extract_final_answer_with_fallback
from backend.openrouter import query_model, query_models_parallel


class IndependentRankSynthesize(GovernanceStructure):
    """
    Governance Structure A: Independent → Rank → Synthesize.

    Stage 1: Each LLM answers independently
    Stage 2: Each LLM ranks all answers (anonymized as Response A, B, C, ...)
    Stage 3: Chairman synthesizes final answer based on rankings
    """

    @property
    def name(self) -> str:
        return "Independent → Rank → Synthesize"

    async def run(self, query: str) -> CouncilResult:
        """Execute the three-stage governance process."""
        # Stage 1: Collect independent responses
        stage1_responses = await self._stage1_collect_responses(query)

        # Stage 2: Collect rankings (anonymized)
        stage2_data, label_to_model = await self._stage2_collect_rankings(
            query, stage1_responses
        )

        # Stage 3: Synthesize final answer
        stage3_response, final_answer = await self._stage3_synthesize_final(
            query, stage1_responses, stage2_data, label_to_model
        )

        return CouncilResult(
            final_answer=final_answer,
            stage1_responses=stage1_responses,
            stage2_data={
                "rankings": stage2_data,
                "label_to_model": label_to_model,
            },
            stage3_data={"synthesis": stage3_response},
        )

    async def _stage1_collect_responses(self, query: str) -> Dict[str, str]:
        """Stage 1: Query all council models in parallel with standardized prompt."""
        prompt = build_stage1_prompt(query)
        messages = [{"role": "user", "content": prompt}]
        results = await query_models_parallel(self.council_models, messages)

        return {
            model: result.get("content", result.get("error", ""))
            for model, result in results.items()
        }

    async def _stage2_collect_rankings(
        self, query: str, stage1_responses: Dict[str, str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Stage 2: Have each model rank all responses (anonymized).

        Returns:
            Tuple of (rankings dict, label_to_model mapping)
        """
        # Create anonymized labels
        models = list(stage1_responses.keys())
        labels = [chr(ord("A") + i) for i in range(len(models))]
        label_to_model = dict(zip(labels, models))
        model_to_label = dict(zip(models, labels))

        # Build anonymized response text
        anonymized_responses = "\n\n".join(
            f"Response {label}:\n{stage1_responses[model]}"
            for label, model in label_to_model.items()
        )

        ranking_prompt = f"""You are evaluating responses to the following question:

{query}

Here are the responses from different sources:

{anonymized_responses}

Please rank these responses from best to worst. Consider accuracy, completeness, and clarity.

Provide your ranking in this exact format:
FINAL RANKING: [best], [second], [third], ...

For example: FINAL RANKING: B, A, C, D"""

        messages = [{"role": "user", "content": ranking_prompt}]
        results = await query_models_parallel(self.council_models, messages)

        rankings = {}
        for model, result in results.items():
            content = result.get("content", "")
            parsed_ranking = self._parse_ranking(content, labels)
            rankings[model] = parsed_ranking

        return rankings, label_to_model

    def _parse_ranking(self, text: str, valid_labels: List[str]) -> List[str]:
        """Extract ranking from response text."""
        pattern = r"FINAL RANKING:\s*([A-Z](?:\s*,\s*[A-Z])*)"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            ranking_str = match.group(1)
            ranking = [label.strip().upper() for label in ranking_str.split(",")]
            # Filter to only valid labels
            ranking = [label for label in ranking if label in valid_labels]
            return ranking

        # Fallback: return labels in original order
        return valid_labels.copy()

    def _calculate_aggregate_rankings(
        self, rankings: Dict[str, List[str]], label_to_model: Dict[str, str]
    ) -> List[Tuple[str, float]]:
        """Calculate average rank for each response."""
        labels = list(label_to_model.keys())
        rank_sums: Dict[str, float] = {label: 0.0 for label in labels}
        rank_counts: Dict[str, int] = {label: 0 for label in labels}

        for model, ranking in rankings.items():
            for position, label in enumerate(ranking):
                if label in rank_sums:
                    rank_sums[label] += position + 1  # 1-indexed ranking
                    rank_counts[label] += 1

        # Calculate averages
        avg_ranks = []
        for label in labels:
            if rank_counts[label] > 0:
                avg = rank_sums[label] / rank_counts[label]
            else:
                avg = float("inf")
            avg_ranks.append((label, avg))

        # Sort by average rank (lower is better)
        avg_ranks.sort(key=lambda x: x[1])
        return avg_ranks

    async def _stage3_synthesize_final(
        self,
        query: str,
        stage1_responses: Dict[str, str],
        rankings: Dict[str, List[str]],
        label_to_model: Dict[str, str],
    ) -> Tuple[str, str]:
        """
        Stage 3: Chairman synthesizes final answer based on rankings.

        Returns:
            Tuple of (full synthesis response, extracted final answer)
        """
        # Calculate aggregate rankings
        aggregate_rankings = self._calculate_aggregate_rankings(rankings, label_to_model)

        # Build ranked responses text
        ranked_responses = []
        for rank, (label, avg_rank) in enumerate(aggregate_rankings, 1):
            model = label_to_model[label]
            response = stage1_responses[model]
            ranked_responses.append(
                f"Rank {rank} (Response {label}, avg rank {avg_rank:.1f}):\n{response}"
            )
        ranked_text = "\n\n".join(ranked_responses)

        synthesis_prompt = f"""You are the chairman of a council of AI models. The council was asked:

{query}

The council members provided responses, which were then ranked by all members. Here are the responses in order of aggregate ranking (best first):

{ranked_text}

Based on the council's responses and rankings, synthesize the best final answer. Consider the highest-ranked responses most heavily, but incorporate valuable insights from lower-ranked responses if appropriate.

End your response with:
FINAL ANSWER: [your synthesized answer]"""

        messages = [{"role": "user", "content": synthesis_prompt}]
        result = await query_model(self.chairman_model, messages)

        synthesis = result.get("content", "")
        final_answer = extract_final_answer_with_fallback(synthesis)

        return synthesis, final_answer
