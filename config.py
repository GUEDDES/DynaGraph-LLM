class DynaGraphConfig:
    # Temporal Knowledge Constructor parameters
    ALPHA = 0.7  # Semantic-temporal balance
    GAMMA = 0.1  # Temporal decay rate
    PRUNE_THRESHOLD = 0.15  # Node centrality threshold for pruning
    REWIRING_INTERVAL = 5  # Turns between graph rewiring
    
    # Multi-Scale Retriever parameters
    BEAM_WIDTH = 3  # Beam search width
    KAPPA = 0.8  # Degree preference in traversal
    MAX_ANCHORS = 5  # Maximum anchor nodes to consider
    DELTA_RANGE = (1, 5)  # Min/max cognitive depth
    
    # Memory Consolidator parameters
    MERGE_SIMILARITY = 0.85  # Node merging threshold
    COMMUNITY_RESOLUTION = 1.0  # Louvain community detection resolution
    
    # LLM Integration
    TRIPLET_MODEL = "gpt-4-1106-preview"
    MAIN_MODEL = "gpt-4-1106-preview"
    
    # Evaluation
    LONG_RANGE_TEST_SIZE = 100
    COHERENCE_WINDOW = 20