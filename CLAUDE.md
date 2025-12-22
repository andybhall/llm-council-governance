# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose
Experimental framework comparing LLM council governance structures. Extends Karpathy's llm-council with multiple governance implementations and evaluation harness.

## Quick Start

### 1. Install Dependencies
```bash
pip install -e .           # Install package
pip install -e ".[dev]"    # Include dev dependencies (pytest)
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key from https://openrouter.ai/keys
```

### 3. Verify Setup
```bash
python scripts/check_setup.py
```

### 4. Run Pilot Study
```bash
python -m experiments.run_pilot
```

### 5. Analyze Results
```bash
python -m experiments.analyze_pilot
```

## Commands
- `python scripts/check_setup.py` — Verify setup and API connectivity
- `pytest tests/ -v` — Run all tests (219 tests)
- `pytest tests/test_X.py -v` — Run specific test file
- `pytest tests/ --cov=backend` — Run with coverage
- `python -m experiments.run_pilot` — Run pilot experiment
- `python -m experiments.analyze_pilot` — Analyze results

## Code Style
- Use async/await for all LLM calls
- Type hints on all function signatures
- Docstrings for public functions
- Extract common logic to `governance/utils.py`

## Testing Rules — CRITICAL
- Write test BEFORE or IMMEDIATELY AFTER each function
- Run tests after every change
- Never claim code works without a passing test
- Mock `openrouter.query_model` in tests to avoid real API calls

## Workflow
1. Complete one module at a time
2. Run tests for that module
3. Commit with descriptive message
4. Report status and STOP
5. Wait for confirmation before next module

## Key Abstractions
- `GovernanceStructure` (base.py): ABC for all governance structures
- `Benchmark` (evaluation/base.py): ABC for all benchmarks
- `CouncilResult`: Dataclass for governance outputs
- `extract_final_answer()`: Parse "FINAL ANSWER: X" from responses

## Four Governance Structures
- **A**: Independent → Rank → Synthesize (baseline)
- **B**: Independent → Majority Vote
- **C**: Independent → Deliberate → Vote
- **D**: Independent → Deliberate → Synthesize

## Answer Extraction
All prompts must include: "End with FINAL ANSWER: [your answer]"
Parse with: `r"FINAL ANSWER:\s*(.+?)(?:\n|$)"`

## Architecture

### Target Directory Structure
```
backend/
├── config.py                 # Model configs, API keys
├── openrouter.py             # OpenRouter API client (query_model, query_models_parallel)
├── council.py                # Core 3-stage logic from original repo
├── governance/
│   ├── base.py               # GovernanceStructure ABC, CouncilResult dataclass
│   ├── utils.py              # extract_final_answer(), majority_vote()
│   ├── independent_rank_synthesize.py  # Structure A
│   ├── majority_vote.py      # Structure B
│   ├── deliberate_vote.py    # Structure C
│   └── deliberate_synthesize.py  # Structure D
└── evaluation/
    ├── base.py               # Benchmark ABC, Question/EvalResult dataclasses
    ├── gsm8k.py              # GSM8K loader (HuggingFace datasets)
    └── truthfulqa.py         # TruthfulQA loader
experiments/
├── run_pilot.py              # Main experiment runner
├── analyze_pilot.py          # Results analysis
└── results/                  # Output directory
```

### Benchmarks
- **GSM8K** (40 questions): Grade school math with numerical ground truth
- **TruthfulQA** (40 questions): Multiple choice, tests resistance to misconceptions
- **Custom** (20 questions): Open-ended, requires human evaluation

## Implementation Status
All 12 modules complete with 219 passing tests. See INSTRUCTIONS.md for implementation details.
