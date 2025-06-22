"""
Evaluation Suite
================

Comprehensive evaluation modules for assessing system performance:

- long_range_eval: Measures long-range dependency resolution
- coherence_eval: Assesses conversational consistency
- robustness_eval: Tests contradiction handling
- cognitive_load: Quantifies computational efficiency
"""

from .long_range_eval import LongRangeEvaluator
from .coherence_eval import CoherenceEvaluator

# Stub imports for future expansion
def robustness_eval():
    from . import robustness_eval
    return robustness_eval

def cognitive_load():
    from . import cognitive_load
    return cognitive_load

__all__ = [
    'LongRangeEvaluator',
    'CoherenceEvaluator',
    'robustness_eval',
    'cognitive_load'
]