# INSTRUCTIONS.md — LLM Council Governance Pilot Study

## CRITICAL WORKFLOW REQUIREMENTS

**YOU MUST FOLLOW THESE RULES. THEY ARE NON-NEGOTIABLE.**

1. **NEVER claim something works without running a test to prove it.** After writing any code, immediately write and run a test. If you cannot test it, say so explicitly.

2. **Work modularly.** Complete one module at a time. After each module:
   - Run all tests for that module
   - Report what you built, what tests passed, and any issues
   - STOP and wait for my confirmation before proceeding to the next module

3. **Iterate and fix your own errors.** When tests fail:
   - Read the error message carefully
   - Diagnose the problem
   - Fix it yourself
   - Re-run the test
   - Repeat until it passes
   - Do NOT ask me to report errors back to you

4. **Use version control.** Commit after each successful module with a descriptive message.

5. **When in doubt, ask clarifying questions BEFORE coding**, not after.

---

## Project Overview

We are building an experimental framework to compare different LLM council governance structures. This extends Karpathy's llm-council repo (https://github.com/karpathy/llm-council) with:

1. Multiple governance structure implementations
2. An evaluation harness for running experiments
3. Benchmark loaders (GSM8K, TruthfulQA)
4. Results logging and analysis

**Goal**: Run a pilot study comparing 4 governance structures on 100 questions to determine which approaches warrant further investigation.

---

## Existing Codebase Architecture

The llm-council repo has this structure:

```
llm-council/
├── backend/
│   ├── __init__.py
│   ├── config.py          # Model configs, API keys
│   ├── openrouter.py      # OpenRouter API client
│   ├── council.py         # Core 3-stage logic
│   ├── storage.py         # JSON file storage
│   └── main.py            # FastAPI endpoints
├── frontend/              # React UI (we won't modify this)
├── CLAUDE.md              # Project documentation
├── pyproject.toml         # Dependencies (uses uv)
└── start.sh               # Launch script
```

### Key Functions in Existing Code

**backend/config.py**:
```python
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview", 
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
```

**backend/openrouter.py**:
- `query_model(model, messages)`: Single async model query
- `query_models_parallel(models, messages)`: Parallel queries using asyncio.gather()

**backend/council.py**:
- `stage1_collect_responses(query)`: Parallel queries to all council models
- `stage2_collect_rankings(query, stage1_results)`: Anonymizes responses ("Response A", "Response B", etc.), each LLM ranks them
- `stage3_synthesize_final(query, stage1_results, stage2_results)`: Chairman synthesizes
- `parse_ranking_from_text(text)`: Extracts "FINAL RANKING:" section
- `calculate_aggregate_rankings(stage2_results, label_to_model)`: Computes average rank

**Anonymization**: In Stage 2, responses are labeled "Response A", "Response B", etc. to prevent models from playing favorites.

---

## What We Are Building

### Four Governance Structures

**Structure A: Independent → Rank → Synthesize (Karpathy Baseline)**
- Stage 1: Each LLM answers independently
- Stage 2: Each LLM ranks all answers (anonymized)
- Stage 3: Chairman synthesizes final answer
- *This is what the existing repo does*

**Structure B: Independent → Majority Vote**
- Stage 1: Each LLM answers independently
- Stage 2: Extract final answer from each response
- Stage 3: Majority vote determines winner (no synthesis)

**Structure C: Independent → Deliberate → Vote**
- Stage 1: Each LLM answers independently
- Stage 2: All LLMs see all responses, asked to reconsider
- Stage 3: Extract updated answers, majority vote

**Structure D: Independent → Deliberate → Synthesize**
- Stage 1: Each LLM answers independently
- Stage 2: All LLMs see all responses, asked to reconsider
- Stage 3: Chairman synthesizes (like A, but after deliberation)

### Benchmarks

1. **GSM8K** (40 questions): Grade school math, clear ground truth
2. **TruthfulQA** (40 questions): Tests resistance to common misconceptions
3. **Custom** (20 questions): Open-ended, human evaluation

### Pilot Study Parameters

- 100 total questions
- 4 governance structures
- 3 replications per condition
- ~1,200 total council runs

---

## Implementation Plan — MODULE BY MODULE

### Module 1: Project Setup and Existing Code Understanding

**Tasks**:
1. Clone the repo (or set up a new project with the same structure)
2. Verify you can import and use the existing `openrouter.py` functions
3. Create a simple test that mocks an API call and verifies the flow

**Deliverables**:
- Working project structure
- Test file: `tests/test_setup.py`
- Passing test proving imports work

**Test to write**:
```python
def test_config_loads():
    """Verify config.py loads without errors."""
    from backend.config import COUNCIL_MODELS, CHAIRMAN_MODEL
    assert len(COUNCIL_MODELS) == 4
    assert CHAIRMAN_MODEL is not None
```

**STOP after this module and report status.**

---

### Module 2: Governance Structure Abstraction

**Tasks**:
1. Create `backend/governance/base.py` with abstract base class
2. Create `backend/governance/__init__.py`

**File: backend/governance/base.py**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class CouncilResult:
    """Result from running a governance structure."""
    final_answer: str
    stage1_responses: Dict[str, str]  # model -> response
    stage2_data: Optional[Any] = None  # Structure-specific
    stage3_data: Optional[Any] = None  # Structure-specific
    metadata: Optional[Dict[str, Any]] = None  # Timings, token counts, etc.

class GovernanceStructure(ABC):
    """Abstract base class for governance structures."""
    
    def __init__(self, council_models: List[str], chairman_model: str):
        self.council_models = council_models
        self.chairman_model = chairman_model
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this structure."""
        pass
    
    @abstractmethod
    async def run(self, query: str) -> CouncilResult:
        """Execute the governance process and return result."""
        pass
```

**Test to write** (`tests/test_governance_base.py`):
```python
def test_council_result_dataclass():
    """Verify CouncilResult can be instantiated."""
    from backend.governance.base import CouncilResult
    result = CouncilResult(
        final_answer="42",
        stage1_responses={"model1": "response1"}
    )
    assert result.final_answer == "42"
```

**STOP after this module and report status.**

---

### Module 3: Structure A — Independent → Rank → Synthesize

**Tasks**:
1. Create `backend/governance/independent_rank_synthesize.py`
2. Wrap existing council.py logic into the new abstraction
3. Test with a mock API

**Key implementation notes**:
- Reuse `stage1_collect_responses`, `stage2_collect_rankings`, `stage3_synthesize_final`
- Extract final answer from chairman's synthesis

**Test to write**:
```python
@pytest.mark.asyncio
async def test_structure_a_runs(mock_openrouter):
    """Verify Structure A executes all three stages."""
    from backend.governance.independent_rank_synthesize import IndependentRankSynthesize
    
    structure = IndependentRankSynthesize(
        council_models=["model1", "model2"],
        chairman_model="chairman"
    )
    result = await structure.run("What is 2+2?")
    
    assert result.final_answer is not None
    assert len(result.stage1_responses) == 2
```

You'll need to create a `mock_openrouter` fixture that patches API calls.

**STOP after this module and report status.**

---

### Module 4: Answer Extraction Utility

**Tasks**:
1. Create `backend/governance/utils.py`
2. Implement answer extraction for voting structures

**Key challenge**: We need to extract the "final answer" from free-form LLM responses.

**Approach**:
- Add to prompts: "End your response with FINAL ANSWER: [your answer]"
- Parse with regex: `r"FINAL ANSWER:\s*(.+?)(?:\n|$)"`
- Fallback: Use last sentence or ask LLM to extract

**File: backend/governance/utils.py**
```python
import re
from typing import Optional

def extract_final_answer(response: str) -> Optional[str]:
    """Extract final answer from a response that includes 'FINAL ANSWER: X'."""
    pattern = r"FINAL ANSWER:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def majority_vote(answers: List[str], tiebreaker: Optional[str] = None) -> str:
    """Return the most common answer. Use tiebreaker if provided."""
    from collections import Counter
    counts = Counter(answers)
    max_count = max(counts.values())
    winners = [ans for ans, count in counts.items() if count == max_count]
    
    if len(winners) == 1:
        return winners[0]
    elif tiebreaker and tiebreaker in winners:
        return tiebreaker
    else:
        return winners[0]  # Arbitrary if no tiebreaker
```

**Tests to write**:
```python
def test_extract_final_answer():
    response = "Let me think... The answer is 4. FINAL ANSWER: 4"
    assert extract_final_answer(response) == "4"

def test_extract_final_answer_not_found():
    response = "The answer is 4."
    assert extract_final_answer(response) is None

def test_majority_vote_clear_winner():
    assert majority_vote(["A", "A", "B", "A"]) == "A"

def test_majority_vote_tie_with_tiebreaker():
    assert majority_vote(["A", "A", "B", "B"], tiebreaker="B") == "B"
```

**STOP after this module and report status.**

---

### Module 5: Structure B — Independent → Majority Vote

**Tasks**:
1. Create `backend/governance/majority_vote.py`
2. Modify prompts to request structured output

**Key implementation**:
- Stage 1: Query all models with modified prompt including "FINAL ANSWER:" instruction
- Stage 2: Extract answers using `extract_final_answer`
- Stage 3: Run `majority_vote`, chairman as tiebreaker

**Modified prompt template**:
```
{original_query}

After your reasoning, state your final answer in this exact format:
FINAL ANSWER: [your answer]
```

**STOP after this module and report status.**

---

### Module 6: Structure C — Independent → Deliberate → Vote

**Tasks**:
1. Create `backend/governance/deliberate_vote.py`
2. Implement deliberation prompt

**Deliberation prompt**:
```
You previously answered the following question:

{original_query}

Your original response was:
{your_response}

Here are the responses from other council members:

{other_responses}

Consider their reasoning. You may revise your answer or maintain your original position.
Provide your (possibly updated) response, ending with:
FINAL ANSWER: [your answer]
```

**STOP after this module and report status.**

---

### Module 7: Structure D — Independent → Deliberate → Synthesize

**Tasks**:
1. Create `backend/governance/deliberate_synthesize.py`
2. Combine deliberation from C with synthesis from A

**STOP after this module and report status.**

---

### Module 8: Benchmark Loader — GSM8K

**Tasks**:
1. Create `backend/evaluation/base.py` with Benchmark ABC
2. Create `backend/evaluation/gsm8k.py`

**File: backend/evaluation/base.py**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Question:
    id: str
    text: str
    ground_truth: Optional[str] = None
    metadata: Optional[dict] = None

@dataclass  
class EvalResult:
    question_id: str
    is_correct: Optional[bool]
    predicted: str
    expected: Optional[str]
    
class Benchmark(ABC):
    @abstractmethod
    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        """Load questions from the benchmark."""
        pass
    
    @abstractmethod
    def evaluate(self, question: Question, response: str) -> EvalResult:
        """Evaluate a response against ground truth."""
        pass
```

**GSM8K specifics**:
- Load from HuggingFace: `datasets.load_dataset("gsm8k", "main")`
- Ground truth is numerical
- Need to extract number from response and compare

**STOP after this module and report status.**

---

### Module 9: Benchmark Loader — TruthfulQA

**Tasks**:
1. Create `backend/evaluation/truthfulqa.py`
2. Use multiple-choice subset for easy evaluation

**TruthfulQA specifics**:
- Load from HuggingFace: `datasets.load_dataset("truthful_qa", "multiple_choice")`
- Format as multiple choice (A, B, C, D)
- Extract letter answer and compare

**STOP after this module and report status.**

---

### Module 10: Experiment Runner

**Tasks**:
1. Create `experiments/run_pilot.py`
2. Implement main experiment loop with logging

**Structure**:
```python
async def run_experiment(
    structures: List[GovernanceStructure],
    benchmarks: List[Benchmark],
    n_replications: int = 3,
    output_dir: str = "experiments/results"
) -> pd.DataFrame:
    """Run the full pilot experiment."""
    results = []
    
    for benchmark in benchmarks:
        questions = benchmark.load_questions()
        for question in questions:
            for structure in structures:
                for rep in range(n_replications):
                    # Run council
                    council_result = await structure.run(question.text)
                    
                    # Evaluate
                    eval_result = benchmark.evaluate(
                        question, 
                        council_result.final_answer
                    )
                    
                    # Log
                    results.append({
                        "benchmark": benchmark.name,
                        "question_id": question.id,
                        "structure": structure.name,
                        "replication": rep,
                        "is_correct": eval_result.is_correct,
                        "predicted": eval_result.predicted,
                        "expected": eval_result.expected,
                        # Add timing, tokens, etc.
                    })
                    
                    # Save incrementally
                    save_results(results, output_dir)
    
    return pd.DataFrame(results)
```

**STOP after this module and report status.**

---

### Module 11: Analysis Script

**Tasks**:
1. Create `experiments/analyze_pilot.py`
2. Compute accuracy by structure × benchmark

**Output**:
- Table of accuracy by structure and question type
- Cost summary
- Statistical tests (if sample size permits)

**STOP after this module and report status.**

---

### Module 12: Integration Test

**Tasks**:
1. Create `tests/test_integration.py`
2. Run a mini-experiment with 2 questions × 2 structures × 1 replication
3. Verify end-to-end flow works

This is the final validation before running the real pilot.

**STOP after this module and report status.**

---

## File Structure When Complete

```
llm-council/
├── backend/
│   ├── config.py                  # (existing)
│   ├── openrouter.py              # (existing)
│   ├── council.py                 # (existing)
│   ├── governance/
│   │   ├── __init__.py
│   │   ├── base.py                # GovernanceStructure ABC
│   │   ├── utils.py               # Answer extraction, voting
│   │   ├── independent_rank_synthesize.py  # Structure A
│   │   ├── majority_vote.py       # Structure B
│   │   ├── deliberate_vote.py     # Structure C
│   │   └── deliberate_synthesize.py  # Structure D
│   └── evaluation/
│       ├── __init__.py
│       ├── base.py                # Benchmark ABC
│       ├── gsm8k.py
│       └── truthfulqa.py
├── experiments/
│   ├── run_pilot.py
│   ├── analyze_pilot.py
│   └── results/                   # Output directory
├── tests/
│   ├── test_setup.py
│   ├── test_governance_base.py
│   ├── test_utils.py
│   ├── test_structures.py
│   ├── test_benchmarks.py
│   └── test_integration.py
├── CLAUDE.md
└── pyproject.toml
```

---

## Dependencies to Add

Add to `pyproject.toml`:
```toml
[project]
dependencies = [
    # Existing deps...
    "datasets",      # HuggingFace datasets
    "pandas",        # Data analysis
    "pytest",        # Testing
    "pytest-asyncio", # Async test support
]
```

---

## Testing Strategy

1. **Unit tests**: Test each function in isolation with mocks
2. **Integration tests**: Test module interactions
3. **End-to-end test**: Run mini-experiment before real pilot

**Mocking approach**:
- Mock `openrouter.query_model` to avoid real API calls in tests
- Use `pytest-asyncio` for async tests
- Create fixtures for common test data

**Example mock fixture**:
```python
@pytest.fixture
def mock_openrouter(monkeypatch):
    async def mock_query(model, messages):
        return {
            "content": f"Mock response from {model}. FINAL ANSWER: 42"
        }
    
    monkeypatch.setattr(
        "backend.openrouter.query_model", 
        mock_query
    )
```

---

## IMPORTANT REMINDERS

1. **Run tests after every change.** Use `pytest tests/ -v`

2. **Commit after each module.** Use `git commit -m "Module N: description"`

3. **If a test fails, fix it yourself.** Read the error, diagnose, fix, re-run.

4. **Report status after each module.** Include:
   - What you built
   - What tests pass
   - Any issues or questions

5. **Do not proceed to next module without confirmation.**

---

## Quick Reference: Key Commands

```bash
# Run tests
pytest tests/ -v

# Run single test file
pytest tests/test_governance_base.py -v

# Run with coverage
pytest tests/ --cov=backend

# Type check (if using mypy)
mypy backend/

# Commit changes
git add -A && git commit -m "Module N: description"
```

---

## Questions to Ask Before Starting

If anything is unclear, ask before coding:
- Which Python version should I target?
- Should I use the existing llm-council repo or start fresh?
- Where should the OpenRouter API key come from in tests?
- Any preferences on logging/output format?

---

**BEGIN WITH MODULE 1. REPORT STATUS WHEN COMPLETE.**
