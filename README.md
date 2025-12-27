# LLM Council Governance Study

An experimental framework comparing different governance structures for LLM councils. Extends [Karpathy's llm-council](https://github.com/karpathy/llm-council) concept with multiple governance implementations and a rigorous evaluation harness.

## Motivation

Karpathy proposed and built an "llm council" to advise a user. This raises an interesting governance question: what is the optimal procedure for this council to adopt? To what extent should the council deliberate and learn from one another, vs. to what extent should the council vote based on each llm's private information? The goal of this research is to start testing different governance structures and see which arrives at superior decisions.

## Key Findings

We ran 2,880 trials across three experiments to explore what factors appear to influence LLM council effectiveness. These findings are specific to our experimental setup: 7-9B parameter models, GSM8K and TruthfulQA benchmarks, and the particular prompts and structures tested.

### Overview

| Configuration | Accuracy | vs Single Model |
|---------------|----------|-----------------|
| Single model (Gemma 2 9B) | 84.5% | — |
| Same model + prompt diversity | 84.1% | -0.4% |
| Same model + persona diversity | 83.0% | -1.5% |
| Multi-model council (best) | 85.4% | +0.9% |

In this experiment, the multi-model council showed a modest improvement over the single best model, though the difference was not statistically significant. Prompt and persona diversity did not appear to help. These results may not generalize to other models, benchmarks, or prompt designs.

### Four Observations

#### 1. Model diversity appeared more effective than prompt diversity

We tested whether a single model (Gemma 2 9B) with 4 different prompts could match a true multi-model council. In our experiments:

- **Prompt variants**: 84.1% — no apparent improvement over baseline
- **Persona variants**: 83.0% — no apparent improvement over baseline
- **Different models**: 85.4% — modest improvement, but not statistically significant

This suggests that, at least for these particular prompts and benchmarks, whatever value councils provide may come from models trained on different data rather than from prompting the same model differently. Other prompt designs might yield different results.

#### 2. Deliberation appeared to help small models

For the 7-9B parameter models tested, letting council members see each other's answers before voting appeared to improve accuracy:

| Structure | Accuracy | Description |
|-----------|----------|-------------|
| Deliberate → Vote | 85.4% | See others' answers, then vote |
| Deliberate → Synthesize | 84.5% | See others, chairman synthesizes |
| Weighted Majority Vote | 84.2% | Vote weighted by model accuracy |
| Majority Vote | 84.1% | Vote without seeing others |
| Rank → Synthesize | 82.9% | Rank answers, chairman synthesizes |

*Note: These differences were not statistically significant (p=0.97), so we cannot rule out chance variation.*

In our data, deliberation appeared to help because:
- Agreement increased from 87.4% → 92.9% after deliberation
- Net effect: +56 more answers fixed than broken
- Weaker models may have benefited from seeing stronger models' reasoning

This pattern might differ for frontier models, where deliberation could potentially introduce groupthink.

#### 3. Voting appeared more reliable than synthesis

In our experiments, structures ending with voting outperformed those where a "chairman" synthesized the final answer:

| Final Stage | Best Accuracy |
|-------------|---------------|
| Majority vote | 87.8% |
| Chairman synthesis | 84.7% |

Synthesis may introduce a single point of failure, though this could depend on which model serves as chairman and how the synthesis prompt is designed.

#### 4. Weighted voting did not appear to help

We tested whether weighting votes by each model's historical accuracy would improve results. Each model's vote was weighted by its individual accuracy rate:

| Model | Weight (Accuracy) |
|-------|-------------------|
| Gemma 2 9B | 0.845 |
| Qwen 2.5 7B | 0.817 |
| Llama 3.1 8B | 0.748 |
| Mistral 7B | 0.671 |

**Results:**
| Voting Method | Accuracy | Difference |
|---------------|----------|------------|
| Simple majority vote | 84.1% | — |
| Weighted majority vote | 84.2% | +0.1% |

In our experiments, weighting by accuracy did not appear to improve over simple majority voting. The difference (+0.1%) was not statistically significant (McNemar's test, p=1.0). This suggests that for this council configuration, each model's vote may contribute roughly equal marginal value despite differing individual accuracy rates.

However, this null result could be specific to:
- The particular weight values used (derived from the same benchmark)
- The small spread in model accuracies (67-85%)
- The council size (4 models)

Different weighting schemes or more diverse accuracy spreads might yield different results.

### Summary

| Question | Observation (in this experiment) |
|----------|----------------------------------|
| Do councils beat single models? | Slightly (+0.9%), but not statistically significant |
| Does deliberation help? | Appeared to, for these small models |
| Does prompt diversity help? | Did not appear to, with these prompts |
| Does persona diversity help? | Did not appear to, with these personas |
| Does weighted voting help? | Did not appear to, with these weights |
| Is voting or synthesis better? | Voting appeared slightly better |
| Best structure tested? | Deliberate → Vote (85.4%) |

---

## Experimental Details

### The Five Governance Structures

| Structure | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| **A: Rank→Synthesize** | 4 models answer independently | Each model ranks all answers | Chairman synthesizes based on rankings |
| **B: Majority Vote** | 4 models answer independently | — | Take majority vote (equal weights) |
| **C: Deliberate→Vote** | 4 models answer independently | Each model sees others' answers, can revise | Take majority vote |
| **D: Deliberate→Synthesize** | 4 models answer independently | Each model sees others' answers, can revise | Chairman synthesizes |
| **E: Weighted Vote** | 4 models answer independently | — | Take weighted majority vote (by accuracy) |

### Models Tested

| Model | Overall | GSM8K | TruthfulQA |
|-------|---------|-------|------------|
| Gemma 2 9B | 84.4% | 85.5% | 83.3% |
| Qwen 2.5 7B | 79.5% | 88.6% | 70.5% |
| Llama 3.1 8B | 75.3% | 74.5% | 76.2% |
| Mistral 7B | 68.9% | 62.3% | 75.5% |

### Council Performance by Benchmark

| Structure | GSM8K | TruthfulQA | Overall |
|-----------|-------|------------|---------|
| Deliberate → Vote | 89.2% | 81.5% | 85.4% |
| Deliberate → Synthesize | 92.5% | 76.5% | 84.5% |
| Weighted Majority Vote | 87.5% | 80.8% | 84.2% |
| Majority Vote | 86.6% | 81.7% | 84.1% |
| Rank → Synthesize | 88.3% | 77.5% | 82.9% |

*Based on 1,197 valid trials across 5 structures. Differences between structures were not statistically significant (p=0.97).*

### Deliberation Behavior

Models changed their answers after seeing others' responses:

| Metric | Value |
|--------|-------|
| Answers fixed (wrong → correct) | +146 |
| Answers broken (correct → wrong) | -90 |
| Net change | +56 |

| Model | Change Rate | Fix Rate | Break Rate |
|-------|-------------|----------|------------|
| Llama 3.1 8B | 19.3% | 48.3% | 5.8% |
| Gemma 2 9B | 16.1% | 48.6% | 9.8% |
| Qwen 2.5 7B | 11.7% | 40.7% | 4.1% |
| Mistral 7B | 9.2% | 44.2% | 4.2% |

---

## Prompt & Persona Experiments

We tested whether artificial diversity could substitute for model diversity by using the same model (Gemma 2 9B) with different instructions.

### Prompt Variants Tested

Four reasoning-style prompts were prepended to each question:

**1. Step-by-Step**
> Think through this step by step. Break down the problem into smaller parts and solve each one carefully before moving to the next.

**2. Identify-then-Solve**
> First, identify the key information and constraints in this problem. List them explicitly. Then, use that information to systematically solve the problem.

**3. Skeptical Verifier**
> Consider common mistakes people make with problems like this. As you solve it, verify each step of your reasoning and check for errors before proceeding.

**4. Example-Based**
> Think of similar problems you've seen before. What patterns or approaches worked for those? Apply those insights to solve this problem.

**Results by Structure:**
| Structure | Accuracy | vs Baseline |
|-----------|----------|-------------|
| Rank → Synthesize | 86.2% | +1.8% |
| Deliberate → Vote | 86.1% | +1.6% |
| Majority Vote | 85.4% | +0.9% |
| Deliberate → Synthesize | 78.6% | -5.9% |
| **Overall** | **84.1%** | **-0.4%** |

### Persona Variants Tested

Four character personas were prepended to each question:

**1. The Rigorous Mathematician**
> You are a rigorous mathematician who demands formal precision. You find sloppy reasoning physically painful. Every claim must be justified, every step must follow logically from the previous one. You don't trust intuition—you trust proof. If something seems obvious, that's exactly when you should verify it carefully.

**2. The Skeptical Scientist**
> You are a skeptical scientist who questions everything. Your first instinct is to ask "Is this actually true?" You've seen too many confident wrong answers to trust anything at face value. You look for hidden assumptions, check edge cases, and actively try to find flaws in reasoning—including your own.

**3. The Practical Engineer**
> You are a practical engineer who cares about real-world correctness. Before accepting any answer, you ask: "Does this make sense?" You use estimation and sanity checks. If a math problem gives you a negative number of apples or a person's age of 500 years, something went wrong. You trust your intuition about what's reasonable.

**4. The Enthusiastic Teacher**
> You are an enthusiastic teacher who loves making things crystal clear. You believe any problem can be solved if you break it down carefully enough. You explain your reasoning as if teaching a student, making each step explicit. You use concrete examples when helpful and always double-check your work before giving a final answer.

**Results by Structure:**
| Structure | Accuracy | vs Baseline |
|-----------|----------|-------------|
| Majority Vote | 85.7% | +1.2% |
| Deliberate → Vote | 85.2% | +0.7% |
| Rank → Synthesize | 84.0% | -0.5% |
| Deliberate → Synthesize | 77.0% | -7.5% |
| **Overall** | **83.0%** | **-1.5%** |

### Interpretation

Neither prompt nor persona diversity appeared to improve over the single-model baseline in our experiments. However, this does not rule out the possibility that other prompt designs, more distinctive personas, or different base models might yield different results. The prompts and personas tested here represent only a small sample of possible approaches.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/andybhall/llm-council-governance.git
cd llm-council-governance

# Install dependencies
pip install -e .

# Configure API key
cp .env.example .env
# Edit .env and add your OpenRouter API key from https://openrouter.ai/keys

# Verify setup
python scripts/check_setup.py
```

## Usage

### Run experiments

```bash
# Main pilot study (multi-model council)
python -m experiments.run_pilot
python -m experiments.analyze_pilot

# Prompt diversity experiment
python -m experiments.run_prompt_experiment
python -m experiments.analyze_prompt_experiment

# Persona diversity experiment
python -m experiments.run_persona_experiment
python -m experiments.analyze_persona_experiment

# Voting strategy comparison (weighted vs simple majority)
python -m experiments.run_voting_comparison
python -m experiments.run_voting_comparison analyze
```

### Configuration

Edit `.env` to switch between cheap and frontier models:

```bash
# Cheap models for testing (~$3-5)
USE_CHEAP_MODELS=true

# Frontier models for production (~$150-350)
USE_CHEAP_MODELS=false
```

**Cheap models** (default):
- meta-llama/llama-3.1-8b-instruct
- mistralai/mistral-7b-instruct
- google/gemma-2-9b-it
- qwen/qwen-2.5-7b-instruct

## Project Structure

```
├── backend/
│   ├── config.py              # Model and API configuration
│   ├── openrouter.py          # OpenRouter API client
│   ├── prompt_variants.py     # Prompt diversity experiment
│   ├── persona_variants.py    # Persona diversity experiment
│   ├── governance/
│   │   ├── base.py            # GovernanceStructure ABC
│   │   ├── utils.py           # Answer extraction, voting
│   │   ├── voting.py          # Voting strategy implementations
│   │   ├── independent_rank_synthesize.py  # Structure A
│   │   ├── structure_b.py     # Majority Vote
│   │   ├── structure_c.py     # Deliberate → Vote
│   │   ├── structure_d.py     # Deliberate → Synthesize
│   │   └── structure_e.py     # Weighted Majority Vote
│   └── evaluation/
│       ├── base.py            # Benchmark ABC
│       ├── gsm8k.py           # Math reasoning benchmark
│       └── truthfulqa.py      # Factual accuracy benchmark
├── experiments/
│   ├── run_pilot.py           # Main experiment runner
│   ├── run_prompt_experiment.py
│   ├── run_persona_experiment.py
│   ├── run_voting_comparison.py  # Voting strategy comparison
│   ├── analyze_*.py           # Analysis scripts
│   └── results/               # Output data and charts
├── tests/                     # Test suite (405 tests)
└── scripts/
    └── check_setup.py         # Setup verification
```

## Benchmarks

- **GSM8K**: Grade school math word problems
- **TruthfulQA**: Factual questions testing resistance to common misconceptions

## Running Tests

```bash
pytest tests/ -v
```

## Limitations

- Results are specific to the 7-9B parameter models tested
- Only two benchmarks were used (GSM8K, TruthfulQA)
- Prompt and persona designs represent a small sample of possible approaches
- Statistical significance was not achieved for structure comparisons
- Results may not generalize to frontier models or other domains

## License

MIT

## Acknowledgments

- Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council)
- Uses [OpenRouter](https://openrouter.ai/) for multi-model API access
- Benchmarks from [GSM8K](https://github.com/openai/grade-school-math) and [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- Related work: [Du et al. (2023)](https://arxiv.org/abs/2305.14325) on multi-agent debate
