# LLM Council Governance Study

An experimental framework comparing different governance structures for LLM councils. Extends [Karpathy's llm-council](https://github.com/karpathy/llm-council) concept with multiple governance implementations and a rigorous evaluation harness.

## Motivation

Karpathy proposed and built an "llm council" to advise a user. This raises an interesting governance question: what is the optimal procedure for this council to adopt? To what extent should the council deliberate and learn from one another, vs. to what extent should the council vote based on each llm's private information? The goal of this research is to start testing different governance structures and see which arrives at superior decisions.

## Key Findings

We ran 1,680 trials in the main pilot study plus additional experiments to explore what factors influence LLM council effectiveness. These findings are specific to our experimental setup: 7-9B parameter models, GSM8K and TruthfulQA benchmarks, and the particular prompts and structures tested.

### Overview

| Configuration | Accuracy | vs Single Model |
|---------------|----------|-----------------|
| Single model (Gemma 2 9B) | 85.2% | — |
| Same model + prompt diversity | 84.1% | -1.1% |
| Same model + persona diversity | 83.0% | -2.2% |
| Self-consistency baseline | 85.8% | +0.6% |
| Multi-model council (range) | 85.8–90.8% | +0.6% to +5.6% |

The top-performing approach was **Deliberate → Synthesize** (90.8%), where council members see each other's reasoning before a chairman synthesizes the final answer. Self-consistency voting (sampling Gemma 2 9B multiple times) performed no better than the base model, suggesting the value comes from model diversity rather than aggregation alone. Prompt and persona diversity did not appear to help.

### Five Observations

#### 1. Model diversity matters more than sampling the same model

Self-consistency voting (sampling Gemma 2 9B eleven times with temperature 0.7) achieved only 85.8% accuracy—essentially identical to the base model's 85.2%. Multi-model councils performed better:

| Approach | Accuracy | vs Majority Vote |
|----------|----------|------------------|
| Deliberate → Synthesize | 90.8% | +3.8% |
| Deliberate → Vote | 87.8% | +0.8% |
| Multi-model councils | 85.8–87.8% | baseline |
| Self-Consistency Vote | 85.8% | -1.2% |

This suggests that the value of councils comes from model diversity (different training data, architectures) rather than from aggregating multiple samples of the same model. When a model is already at its accuracy ceiling, sampling it multiple times doesn't help.

#### 2. Model diversity appeared more effective than prompt diversity

We tested whether a single model (Gemma 2 9B) with 4 different prompts could match a true multi-model council. In our experiments:

- **Prompt variants**: 84.1% — no apparent improvement over baseline
- **Persona variants**: 83.0% — no apparent improvement over baseline
- **Different models**: 85.8–90.8% — improvement over baseline

This suggests that, at least for these particular prompts and benchmarks, whatever value councils provide may come from models trained on different data rather than from prompting the same model differently.

#### 3. Deliberation appeared to help small models

For the 7-9B parameter models tested, letting council members see each other's answers before voting appeared to improve accuracy:

| Structure | Accuracy | 95% CI |
|-----------|----------|--------|
| Deliberate → Synthesize | 90.8% | [86.4%, 93.8%] |
| Deliberate → Vote | 87.8% | [83.0%, 91.3%] |
| Rank → Synthesize | 87.4% | [82.6%, 91.1%] |
| Majority Vote | 87.1% | [82.2%, 90.7%] |
| Agenda Setter + Veto | 86.3% | [81.3%, 90.1%] |
| Self-Consistency Vote | 85.8% | [80.9%, 89.7%] |
| Weighted Majority Vote | 85.8% | [80.9%, 89.7%] |

*Note: Differences between structures were not statistically significant, though Deliberate → Synthesize showed the strongest performance.*

In our data, deliberation appeared to help because:
- Agreement increased from 90.2% → 94.2% after deliberation
- Net effect: +50 more answers fixed than broken
- Weaker models benefited from seeing stronger models' reasoning

This pattern might differ for frontier models, where deliberation could potentially introduce groupthink.

#### 4. No clear winner between voting and synthesis

In our experiments, synthesis-based and voting-based structures performed similarly:

| Final Stage | Best Accuracy | 95% CI |
|-------------|---------------|--------|
| Chairman synthesis | 90.8% | [86.4%, 93.8%] |
| Voting | 87.8% | [83.0%, 91.3%] |

The 3% difference is within sampling error. A well-prompted chairman can effectively synthesize council opinions, but we cannot conclude it's better than voting based on this data.

#### 5. Weighted voting did not improve accuracy

We tested whether weighting votes by each model's historical accuracy would improve results. Each model's vote was weighted by its individual accuracy rate:

| Model | Weight (Accuracy) |
|-------|-------------------|
| Gemma 2 9B | 0.853 |
| Qwen 2.5 7B | 0.839 |
| Llama 3.1 8B | 0.826 |
| Mistral 7B | 0.711 |

**Comparison:**

| Voting Method | Accuracy |
|---------------|----------|
| Simple majority vote | 87.1% |
| Weighted majority vote | 85.8% |

Weighted voting showed no improvement. However, **this comparison is not very informative** due to high model agreement:

| Vote Pattern | Frequency | Weights Matter? |
|--------------|-----------|-----------------|
| 3-1 or 4-0 (clear majority) | 96% | No |
| 2-2 tie | 4% | Potentially |

With 96% of trials having a clear majority, weights rarely get a chance to influence the outcome. Additionally, the top 3 models have similar weights (0.826–0.853), so even in 2-2 ties, the weighted outcome usually matches what the stronger models already voted for.

To properly test weighted voting, you would need either more model disagreement or more diverse accuracy rates.

### Summary

| Question | Observation (in this experiment) |
|----------|----------------------------------|
| Best approach tested? | Deliberate → Synthesize (90.8%) |
| Do multi-model councils beat single models? | Yes, +5.6% for best structure |
| Does self-consistency help? | No, same as base model accuracy |
| Does deliberation help? | Appears to, for these small models |
| Does prompt diversity help? | No, with these prompts |
| Does persona diversity help? | No, with these personas |
| Is voting or synthesis better? | No significant difference |
| Does agenda-setting help? | No, performed similar to baseline |

*Notes: (1) Differences between structures were not statistically significant. (2) Self-consistency voting with Gemma 2 9B showed no improvement over the base model. (3) Weighted voting was also tested but proved uninformative due to high model agreement.*

---

## Experimental Details

### The Seven Governance Structures

| Structure | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| **Rank→Synthesize** | 4 models answer independently | Each model ranks all answers | Chairman synthesizes based on rankings |
| **Majority Vote** | 4 models answer independently | — | Take majority vote (equal weights) |
| **Deliberate→Vote** | 4 models answer independently | Each model sees others' answers, can revise | Take majority vote |
| **Deliberate→Synthesize** | 4 models answer independently | Each model sees others' answers, can revise | Chairman synthesizes |
| **Weighted Vote** | 4 models answer independently | — | Take weighted majority vote (by accuracy) |
| **Self-Consistency Vote** | Single model sampled 11× with temp=0.7 | — | Take majority vote across samples |
| **Agenda Setter + Veto** | 4 models answer independently | Chairman proposes answer | Council votes ACCEPT/VETO; fallback to majority if vetoed |

### Models Tested

| Model | Overall | GSM8K | TruthfulQA |
|-------|---------|-------|------------|
| Gemma 2 9B | 85.2% | 87.0% | 83.4% |
| Qwen 2.5 7B | 83.8% | 90.8% | 76.7% |
| Llama 3.1 8B | 83.0% | 82.1% | 83.8% |
| Mistral 7B | 71.4% | 62.8% | 79.8% |

*Self-Consistency Vote uses Gemma 2 9B as the base model (same as best council model).*

### Council Performance by Benchmark

| Structure | GSM8K | TruthfulQA | Overall |
|-----------|-------|------------|---------|
| Deliberate → Synthesize | 92.4% | 89.2% | 90.8% |
| Self-Consistency Vote | 89.2% | 82.5% | 85.8% |
| Deliberate → Vote | 91.5% | 84.0% | 87.8% |
| Rank → Synthesize | 88.2% | 86.7% | 87.4% |
| Majority Vote | 88.3% | 85.8% | 87.1% |
| Agenda Setter + Veto | 84.3% | 88.2% | 86.3% |
| Weighted Majority Vote | 85.8% | 85.8% | 85.8% |

*Based on 1,668 valid trials across 7 structures (1,680 total, 12 errors).*

### Deliberation Behavior

Models changed their answers after seeing others' responses:

| Metric | Value |
|--------|-------|
| Total answer changes | 217 |
| Answers fixed (wrong → correct) | +122 |
| Answers broken (correct → wrong) | -72 |
| Net change | +50 |

| Model | Change Rate | Fix Rate | Break Rate |
|-------|-------------|----------|------------|
| Llama 3.1 8B | 14.9% | 46.9% | 5.6% |
| Gemma 2 9B | 12.1% | 47.9% | 4.0% |
| Qwen 2.5 7B | 10.9% | 39.5% | 5.0% |
| Mistral 7B | 10.1% | 47.6% | 4.3% |

### Influence Patterns

During deliberation, some models were more influential than others:

| Most Influential (convinced others) | Most Influenced (followed others) |
|-------------------------------------|-----------------------------------|
| Gemma 2 9B: 119.7× | Mistral 7B: 162.0× |
| Llama 3.1 8B: 109.7× | Gemma 2 9B: 84.0× |
| Qwen 2.5 7B: 107.5× | Qwen 2.5 7B: 77.0× |

Mistral 7B (the weakest model) was most likely to change its answer to match others, while Gemma 2 9B (the strongest) was most likely to convince others.

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
│   ├── governance/
│   │   ├── base.py            # GovernanceStructure ABC
│   │   ├── utils.py           # Answer extraction, voting
│   │   ├── voting.py          # Voting strategy implementations
│   │   ├── independent_rank_synthesize.py  # Rank → Synthesize
│   │   ├── majority_vote.py   # Majority Vote
│   │   ├── deliberate_vote.py # Deliberate → Vote
│   │   ├── deliberate_synthesize.py  # Deliberate → Synthesize
│   │   ├── weighted_vote.py   # Weighted Majority Vote
│   │   ├── self_consistency_vote.py  # Self-Consistency Vote
│   │   └── agenda_veto.py     # Agenda Setter + Veto
│   └── evaluation/
│       ├── base.py            # Benchmark ABC
│       ├── gsm8k.py           # Math reasoning benchmark
│       └── truthfulqa.py      # Factual accuracy benchmark
├── experiments/
│   ├── run_pilot.py           # Main experiment runner
│   ├── analyze_pilot.py       # Results analysis
│   ├── analyze_deliberation_dynamics.py  # Mind-change analysis
│   ├── stats.py               # Paired bootstrap statistics
│   └── results/               # Output data and charts
├── tests/                     # Test suite (485 tests)
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
- Statistical significance was not achieved for most structure comparisons
- Results may not generalize to frontier models or other domains

## License

MIT

## Acknowledgments

- Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council)
- Uses [OpenRouter](https://openrouter.ai/) for multi-model API access
- Benchmarks from [GSM8K](https://github.com/openai/grade-school-math) and [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- Related work: [Du et al. (2023)](https://arxiv.org/abs/2305.14325) on multi-agent debate
