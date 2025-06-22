# DynaGraph-LLM: Dynamic Ontological Memory Framework

A neuro-symbolic framework for mitigating contextual amnesia in Large Language Models through dynamic knowledge graphs.

## Features

- Temporal-semantic knowledge graph construction
- Multi-scale δ-depth retrieval
- Dual-phase memory consolidation
- Cognitive depth optimization
- Comprehensive evaluation suite

## Architecture

![System Architecture](assets/architecture.png)

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```python
from dynagraph import DynaGraphSystem

system = DynaGraphSystem()
response = system.process_input("What activities would you recommend in Kyoto?")
print(response)
```

## Configuration

Modify config.py to adjust:
Semantic-temporal balance (α)

Temporal decay rate (γ)

Cognitive depth (δ)

Beam width and traversal parameters

## Evaluation

# Run long-range dependency evaluation

python evaluation/long_range_eval.py

# Measure conversational coherence

python evaluation/coherence_eval.py

# Test contradiction robustness

python evaluation/robustness_eval.py

## Documentation

See project wiki for detailed API documentation.
