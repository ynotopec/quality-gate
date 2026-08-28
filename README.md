# Quality Gate

**Auto quality control for AI outputs** — hallucination detection, self-scoring, and automatic rollback when quality falls below threshold.

## Architecture

```
Prompt → LLM → Quality Gate → Score → OK: proceed / FAIL: rollback
                        ↓
                Hallucination detection
                Citation checking
                Self-scoring
```

## Features

- **Self-scoring**: Heuristic quality score (0-10) based on structure, length, language consistency
- **Hallucination detection**: Heuristic signals for fake citations, overconfidence, contradictions
- **Citation checking**: Verifies claims are properly sourced
- **Automatic fallback**: When quality fails, returns "I don't know" response
- **Pipeline mode**: LLM → check → regenerate → check → fallback
- **Configurable thresholds**: Adjustable via dict or YAML

## Installation

```bash
pip install -e .
```

## Usage

### Basic: Evaluate a text

```python
from quality_gate import QualityGate

gate = QualityGate()
report = gate.evaluate(
    text="The moon is a satellite [1].",
    prompt="What is the moon?"
)

print(report.level)  # OK | WARNING | FAIL
print(report.self_score)  # 0-10
print(report.hallucination_rate)  # 0.0-1.0
```

### Pipeline: LLM call with automatic quality check

```python
from quality_gate import QualityGatePipeline
from quality_gate.backend import create_backend

backend = create_backend(config={
    "openai_api_key": "sk-...",
    "primary_model": "gpt-4o",
})

pipeline = QualityGatePipeline(backend)
result = pipeline.call(
    prompt="Explain quantum computing",
    system="You are a helpful assistant.",
    max_retries=2,
)

# result = {
#     "response": str,      # Final response text
#     "quality": QualityReport,  # Quality assessment
#     "retry_count": int,   # How many regeneration attempts
#     "fallback_used": bool # Whether fallback was triggered
# }
```

### Customize thresholds

```python
gate = QualityGate(config={
    "self_score_threshold": 7.0,       # Stricter self-score
    "hallucination_rate_threshold": 0.10,  # 10% max hallucination
    "min_citations": 3,               # Require 3+ citations
    "max_retries": 3,                  # Up to 3 regeneration attempts
})
```

### Load from YAML

```yaml
# quality_gate.yaml
self_score_threshold: 7.0
hallucination_rate_threshold: 0.10
min_citations: 3
max_retries: 3
```

```python
from quality_gate import QualityGate
from quality_gate.config import QualityConfig

config = QualityConfig.from_yaml("quality_gate.yaml")
gate = QualityGate(config.__dict__)
```

## Quality Levels

| Level | Meaning | Action |
|-------|---------|--------|
| `OK` | All checks pass | Proceed |
| `WARNING` | One issue detected | Consider regenerating |
| `FAIL` | Multiple issues | Automatic fallback |

## Hallucination Signals

The system detects these heuristic signals:

- **Fake citations**: Patterns like `Ref 1: Fake Source — Fake Data`
- **Uncertain markers**: "probably", "it seems", "according to certain sources"
- **Overconfident facts**: "it's proven", "it's certain", "the truth is that"
- **Made-up names**: Title + name patterns without context
- **Timestamp anomalies**: Dates in wrong formats
- **Unclear sources**: "diverse reports", "multiple sources"
- **Internal contradictions**: "X is Y" followed by "X is not Y"

## Testing

```bash
pytest tests/
```

## Project Structure

```
quality-gate/
├── pyproject.toml
├── README.md
├── src/quality_gate/
│   ├── __init__.py
│   ├── config.py         # QualityConfig with YAML loading
│   ├── backend.py        # LLMBackend + OpenAI factory
│   └── quality_gate.py   # Core: QualityGate, QualityGatePipeline
└── tests/
    ├── test_quality_gate.py  # Main quality gate tests
    └── test_backend.py       # Backend tests
```
