"""
Utility Modules
===============

Helper functions and tools used throughout the DynaGraph-LLM system:

- text_processing: Text cleaning and NLP utilities
- embedding_utils: Semantic embedding operations
- time_utils: Turn management and time formatting
- graph_utils: NetworkX graph operations (imported dynamically)
"""

from .text_processing import clean_text, split_into_sentences, tokenize_with_offsets
from .embedding_utils import get_embedding, cosine_similarity, semantic_search
from .time_utils import TurnCounter, format_duration

# Import graph_utils only when needed to avoid circular imports
def graph_utils():
    from . import graph_utils
    return graph_utils

__all__ = [
    'clean_text',
    'split_into_sentences',
    'tokenize_with_offsets',
    'get_embedding',
    'cosine_similarity',
    'semantic_search',
    'TurnCounter',
    'format_duration',
    'graph_utils'
]