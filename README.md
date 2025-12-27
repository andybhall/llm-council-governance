# LLM Council Governance Study

An experimental framework comparing different governance structures for LLM councils. Extends [Karpathy's llm-council](https://github.com/karpathy/llm-council) concept with multiple governance implementations and a rigorous evaluation harness.

## Motivation

Karpathy proposed and built an "llm council" to advise a user. This raises an interesting governance question: what is the optimal procedure for this council to adopt? To what extent should the council deliberate and learn from one another, vs. to what extent should the council vote based on each llm's private information? The goal of this research is to start testing different governance structures and see which arrives at superior decisions.

## Key Findings

We ran 2,880 trials across three experiments to answer: **What makes LLM councils effective?**

### The Big Picture

| Configuration | Accuracy | vs Single Model |
|---------------|----------|-----------------|
| Single model (Gemma 2 9B) | 84.5% | — |
| Same model + prompt diversity | 84.1% | -0.4% |
| Same model + persona diversity | 83.0% | -1.5% |
| **Multi-model council** | **87.8%** | **+3.3%** |

**Bottom line:** Council benefits come from genuine model diversity (different architectures and training), not from prompt engineering or persona simulation.

### Three Core Insights

#### 1. Real diversity matters, artificial diversity doesn't

We tested whether a single model (Gemma 2 9B) with 4 different prompts could match a true multi-model council:

- **Prompt variants** (step-by-step, skeptical verifier, etc.): 84.1% — no improvement
- **Persona variants** (mathematician, scientist, engineer, teacher): 83.0% — no improvement
- **Different models** (Gemma, Qwen, Llama, Mistral): 87.8% — significant improvement

The value of councils comes from models that were trained differently, not from prompting the same model to "think differently."

#### 2. Deliberation helps small models learn from peers

For 7-9B parameter models, letting council members see each other's answers before voting improves accuracy:

| Structure | Accuracy | Description |
|-----------|----------|-------------|
| **Deliberate → Vote** | **87.8%** | See others' answers, then vote |
| Majority Vote | 85.0% | Vote without seeing others |
| Deliberate → Synthesize | 84.7% | See others, chairman synthesizes |
| Rank → Synthesize | 82.1% | Rank answers, chairman synthesizes |

Why it works:
- Weaker models learn from stronger models' reasoning
- Agreement increases from 85.9% → 93.1% after deliberation
- Net effect: +76 more answers fixed than broken

This contrasts with frontier models, where deliberation can introduce groupthink that hurts performance.

#### 3. Voting beats synthesis

Structures that end with voting outperform those where a "chairman" synthesizes the final answer:

| Final Stage | Best Accuracy |
|-------------|---------------|
| Majority vote | 87.8% |
| Chairman synthesis | 84.7% |

Synthesis introduces a single point of failure. Voting aggregates the wisdom of the council more reliably.

### Summary Table

| Question | Answer |
|----------|--------|
| Do councils beat single models? | Yes, +3.3% accuracy |
| Does deliberation help? | Yes, for small models |
| Does prompt diversity help? | No |
| Does persona diversity help? | No |
| Is voting or synthesis better? | Voting |
| Best structure overall? | Deliberate → Vote (87.8%) |

---

## Detailed Results

### The Four Governance Structures

| Structure | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| **A: Rank→Synthesize** | 4 models answer independently | Each model ranks all answers | Chairman synthesizes based on rankings |
| **B: Majority Vote** | 4 models answer independently | — | Take majority vote |
| **C: Deliberate→Vote** | 4 models answer independently | Each model sees others' answers, can revise | Take majority vote |
| **D: Deliberate→Synthesize** | 4 models answer independently | Each model sees others' answers, can revise | Chairman synthesizes |

### Individual Model Performance

| Model | Overall | GSM8K | TruthfulQA |
|-------|---------|-------|------------|
| Gemma 2 9B | 84.4% | 85.5% | 83.3% |
| Qwen 2.5 7B | 79.5% | 88.6% | 70.5% |
| Llama 3.1 8B | 75.3% | 74.5% | 76.2% |
| Mistral 7B | 68.9% | 62.3% | 75.5% |

### Council Performance by Benchmark

| Structure | GSM8K | TruthfulQA | Overall |
|-----------|-------|------------|---------|
| **Deliberate → Vote** | **89.8%** | **85.7%** | **87.8%** |
| Majority Vote | 87.5% | 82.5% | 85.0% |
| Deliberate → Synthesize | 88.0% | 81.5% | 84.7% |
| Rank → Synthesize | 86.7% | 77.5% | 82.1% |

*Based on 960 trials. No statistically significant differences between structures (p=0.39).*

### Deliberation Behavior

Models change their answers after seeing others' responses:

| Metric | Value |
|--------|-------|
| Answers fixed (wrong → correct) | +159 |
| Answers broken (correct → wrong) | -83 |
| **Net improvement** | **+76** |

| Model | Change Rate | Fix Rate | Break Rate |
|-------|-------------|----------|------------|
| Llama 3.1 8B | 16.8% | 38.5% | 6.2% |
| Qwen 2.5 7B | 16.6% | 52.0% | 4.9% |
| Gemma 2 9B | 14.0% | 48.5% | 7.4% |
| Mistral 7B | 12.2% | 53.7% | 4.2% |

### Prompt & Persona Experiments

Tested whether artificial diversity could substitute for model diversity:

**Prompt Variants** (4 reasoning styles with Gemma 2 9B):
| Structure | Accuracy | vs Baseline |
|-----------|----------|-------------|
| Rank → Synthesize | 86.2% | +1.8% |
| Deliberate → Vote | 86.1% | +1.6% |
| Majority Vote | 85.4% | +0.9% |
| Deliberate → Synthesize | 78.6% | -5.9% |
| **Overall** | **84.1%** | **-0.4%** |

**Persona Variants** (4 character personas with Gemma 2 9B):
| Structure | Accuracy | vs Baseline |
|-----------|----------|-------------|
| Majority Vote | 85.7% | +1.2% |
| Deliberate → Vote | 85.2% | +0.7% |
| Rank → Synthesize | 84.0% | -0.5% |
| Deliberate → Synthesize | 77.0% | -7.5% |
| **Overall** | **83.0%** | **-1.5%** |

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
│   │   ├── independent_rank_synthesize.py  # Structure A
│   │   ├── structure_b.py     # Majority Vote
│   │   ├── structure_c.py     # Deliberate → Vote
│   │   └── structure_d.py     # Deliberate → Synthesize
│   └── evaluation/
│       ├── base.py            # Benchmark ABC
│       ├── gsm8k.py           # Math reasoning benchmark
│       └── truthfulqa.py      # Factual accuracy benchmark
├── experiments/
│   ├── run_pilot.py           # Main experiment runner
│   ├── run_prompt_experiment.py
│   ├── run_persona_experiment.py
│   ├── analyze_*.py           # Analysis scripts
│   └── results/               # Output data and charts
├── tests/                     # Test suite (405 tests)
└── scripts/
    └── check_setup.py         # Setup verification
```

## Benchmarks

- **GSM8K**: Grade school math word problems (89.8% best council accuracy)
- **TruthfulQA**: Factual questions testing resistance to common misconceptions (85.7% best council accuracy)

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT

## Acknowledgments

- Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council)
- Uses [OpenRouter](https://openrouter.ai/) for multi-model API access
- Benchmarks from [GSM8K](https://github.com/openai/grade-school-math) and [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- Supports findings from [Du et al. (2023)](https://arxiv.org/abs/2305.14325) on multi-agent debate
