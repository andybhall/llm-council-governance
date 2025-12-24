# LLM Council Governance Study

An experimental framework comparing different governance structures for LLM councils. Extends [Karpathy's llm-council](https://github.com/karpathy/llm-council) concept with multiple governance implementations and a rigorous evaluation harness.

## Motivation

Karpathy proposed and built an "llm council" to advise a user. This raises an interesting governance question: what is the optimal procedure for this council to adopt? To what extent should the council deliberate and learn from one another, vs. to what extent should the council vote based on each llm's private information? The goal of this research is to start testing different governance structures and see which arrives at superior decisions.

## Key Findings

### 1. Councils outperform all individual models

| | Best Individual (Gemma) | Best Council |
|---|-------------------------|---------|
| Accuracy | 84.4% | 87.8% |
| GSM8K | 85.5% | 89.8% |
| TruthfulQA | 83.3% | 85.7% |

Council governance provides consistent improvement over relying on any single model, supporting findings from [Du et al. (2023)](https://arxiv.org/abs/2305.14325).

### 2. Deliberate → Vote performs best

**Deliberation followed by voting outperforms simple majority voting** for small LLMs (7-9B parameters).

| Structure | Accuracy | 95% CI | Time/Trial |
|-----------|----------|--------|------------|
| **Deliberate → Vote** | **87.8%** | [83.0%, 91.3%] | 22.9s |
| Majority Vote | 85.0% | [79.9%, 89.0%] | 18.6s |
| Deliberate → Synthesize | 84.7% | [79.6%, 88.8%] | 22.6s |
| Rank → Synthesize | 82.1% | [76.7%, 86.4%] | 20.6s |

*Based on 960 trials across GSM8K and TruthfulQA benchmarks. No statistically significant differences (p=0.39).*

### Why does deliberation help?

For smaller models, deliberation enables **learning from peers**:
- Agreement increases from 85.9% → 93.1% after deliberation
- Net positive: deliberation fixes +76 more answers than it breaks
- Weaker models benefit from seeing stronger models' reasoning

This contrasts with frontier models where deliberation can hurt performance by introducing groupthink.

## The Four Governance Structures

| Structure | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| **A: Rank→Synthesize** | 4 models answer independently | Each model ranks all answers | Chairman synthesizes based on rankings |
| **B: Majority Vote** | 4 models answer independently | — | Take majority vote |
| **C: Deliberate→Vote** | 4 models answer independently | Each model sees others' answers, can revise | Take majority vote |
| **D: Deliberate→Synthesize** | 4 models answer independently | Each model sees others' answers, can revise | Chairman synthesizes |

## Results

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

### Council vs Best Individual Model

Paired t-tests comparing each structure's council outcome with the best individual model's (Gemma) outcome on the same trial:

| Structure | Council | Gemma | Difference | Significant? |
|-----------|---------|-------|------------|--------------|
| **Deliberate → Vote** | **87.8%** | 86.1% | +1.7% | No (p=0.45) |
| Majority Vote | 85.0% | 83.3% | +1.7% | No (p=0.43) |
| Deliberate → Synthesize | 84.7% | 85.2% | -0.4% | No (p=0.87) |
| Rank → Synthesize | 82.1% | 82.9% | -0.8% | No (p=0.73) |

![Council vs Individual](experiments/results/council_vs_individual.png)

### Deliberation Analysis

Models change their answers after seeing others' responses:
- **+159** answers fixed (wrong → correct)
- **-83** answers broken (correct → wrong)
- **Net: +76** (deliberation helps overall)

#### Model-Level Deliberation Behavior

| Model | Change Rate | Fix Rate (when wrong) | Break Rate (when correct) |
|-------|-------------|----------------------|---------------------------|
| Llama 3.1 8B | 16.8% | 38.5% | 6.2% |
| Qwen 2.5 7B | 16.6% | 52.0% | 4.9% |
| Gemma 2 9B | 14.0% | 48.5% | 7.4% |
| Mistral 7B | 12.2% | 53.7% | 4.2% |

#### Groupthink Effect

| Metric | Before Deliberation | After Deliberation | Change |
|--------|--------------------|--------------------|--------|
| Agreement rate | 85.9% | 93.1% | +7.1% |

Deliberation increases agreement. For small models, this correlation is beneficial as weaker models learn from stronger ones.

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-council-governance.git
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

### Run the pilot study

```bash
# Run with cheap models (~$3-5, 2-3 hours)
python -m experiments.run_pilot

# Analyze results
python -m experiments.analyze_pilot
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

**Frontier models** (December 2025):
- openai/gpt-5.2
- google/gemini-3-pro-preview
- anthropic/claude-opus-4.5
- x-ai/grok-4

## Project Structure

```
├── backend/
│   ├── config.py              # Model and API configuration
│   ├── openrouter.py          # OpenRouter API client
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
│   ├── run_pilot.py           # Experiment runner
│   ├── analyze_pilot.py       # Analysis and visualization
│   └── results/               # Output data and charts
├── tests/                     # Test suite (219 tests)
└── scripts/
    └── check_setup.py         # Setup verification
```

## Benchmarks

- **GSM8K**: Grade school math word problems (88% council accuracy)
- **TruthfulQA**: Factual questions testing resistance to common misconceptions (86% council accuracy)

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
