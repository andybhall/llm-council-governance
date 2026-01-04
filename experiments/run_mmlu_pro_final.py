"""Final MMLU-Pro experiment runner for paper.

Key changes from original pilot:
1. Self-Consistency uses Llama 3.1 8B (best individual model) for fair comparison
2. Increased timeout (600s vs 300s) to reduce timeouts
3. Reduced concurrency (10 vs 15) to avoid rate limiting
4. 150 questions for better statistical power
"""

import asyncio
from datetime import datetime

from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS, WEIGHTS_FILE
from backend.evaluation.mmlu_pro import MMLUProBenchmark
from backend.governance import (
    AgendaSetterVetoStructure,
    DeliberateSynthesizeStructure,
    DeliberateVoteStructure,
    IndependentRankSynthesize,
    MajorityVoteStructure,
    SelfConsistencyVoteStructure,
    WeightedMajorityVote,
)
from experiments.run_pilot import run_experiment


async def run_mmlu_pro_final(
    n_questions: int = 150,
    output_dir: str = "experiments/results_mmlu_pro_final",
):
    """
    Run final MMLU-Pro experiment with optimized settings.

    Args:
        n_questions: Number of questions (default 150 for better power)
        output_dir: Output directory for results
    """
    print("=== Final MMLU-Pro Experiment ===")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Best individual model for baseline comparison
    BEST_INDIVIDUAL_MODEL = "meta-llama/llama-3.1-8b-instruct"

    # Initialize governance structures
    structures = [
        IndependentRankSynthesize(COUNCIL_MODELS, CHAIRMAN_MODEL),
        MajorityVoteStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        DeliberateVoteStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        DeliberateSynthesizeStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        WeightedMajorityVote(COUNCIL_MODELS, CHAIRMAN_MODEL, weights_file=WEIGHTS_FILE),
        # Self-consistency: use SAME model as baseline for fair comparison
        SelfConsistencyVoteStructure(
            base_model=BEST_INDIVIDUAL_MODEL,  # Llama 3.1 8B (best individual)
            n_samples=9,  # Match compute budget
            temperature=0.7,
        ),
        AgendaSetterVetoStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
    ]

    print(f"Structures ({len(structures)}):")
    for s in structures:
        print(f"  - {s.name}")
    print()

    # MMLU-Pro Math benchmark
    benchmark = MMLUProBenchmark(category="math")

    print(f"Benchmark: {benchmark.name}")
    print(f"Questions: {n_questions}")
    print(f"Replications: 1")
    print(f"Total trials: {n_questions * len(structures)}")
    print()

    # Run with optimized settings
    # Concurrent execution with hard asyncio timeouts to prevent hangs
    results_df = await run_experiment(
        structures=structures,
        benchmarks=[benchmark],
        n_questions=n_questions,
        n_replications=1,
        output_dir=output_dir,
        max_concurrent=3,
    )

    print(f"\nExperiment complete: {len(results_df)} results")
    return results_df


if __name__ == "__main__":
    asyncio.run(run_mmlu_pro_final())
