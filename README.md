# DynaGraph-LLM: Dynamic Ontological Memory Framework

A neuro-symbolic framework for mitigating contextual amnesia in Large Language Models through dynamic knowledge graphs.

## Features

- Temporal-semantic knowledge graph construction
- Multi-scale δ-depth retrieval
- Dual-phase memory consolidation
- Cognitive depth optimization
- **Advanced Regex JSON extraction for robust triplet generation**
- **Native Google AI Studio (Gemini 2.5 Flash) integration**
- **Interactive Visualizer using Matplotlib/NetworkX**
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
from main import DynaGraphSystem

system = DynaGraphSystem()
# Interactive mode via CLI supports standard commands:
# - 'voir' or 'visualize': Displays the topology of the current memory graph
# - 'consolider': Triggers explicit spatial prune/merge of the graph
```

Or just run the interactive terminal:
```bash
python main.py
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
