import networkx as nx
import numpy as np

# Create the graph from the paper
graph = nx.DiGraph()

# Add Nodes
graph.add_node("target_audience", last_updated=1, created=1, centrality=0.0)
graph.add_node("high-school_students", last_updated=1, created=1, centrality=0.0)
graph.add_node("QC_workshop", last_updated=1, created=1, centrality=0.0)
graph.add_node("ages_14-18", last_updated=1, created=1, centrality=0.0)
graph.add_node("prior_knowledge", last_updated=1, created=1, centrality=0.0)
graph.add_node("vectors", last_updated=1, created=1, centrality=0.0)
graph.add_node("matrices", last_updated=1, created=1, centrality=0.0)

# Add Edges
graph.add_edge("target_audience", "high-school_students", predicate="is-a", weight=0.9, last_updated=1)
graph.add_edge("target_audience", "QC_workshop", predicate="located-in", weight=0.8, last_updated=1)
graph.add_edge("high-school_students", "ages_14-18", predicate="has-property", weight=0.7, last_updated=1)
graph.add_edge("high-school_students", "prior_knowledge", predicate="related-to", weight=0.6, last_updated=1)
graph.add_edge("prior_knowledge", "vectors", predicate="includes", weight=0.6, last_updated=1)
graph.add_edge("prior_knowledge", "matrices", predicate="includes", weight=0.6, last_updated=1)

print("--- GRAPH CREATED ---")
print(f"Nodes: {len(graph.nodes)}")
print(f"Edges: {len(graph.edges)}")

from core.retriever import MultiScaleRetriever
retriever = MultiScaleRetriever(beam_width=5)

# Test dynamic cognitive depth
degrees = [d for n, d in graph.degree()]
b = max(1.1, sum(degrees) / len(degrees))
delta_star = retriever._determine_cognitive_depth(b)
print(f"--- DYNAMIC COGNITIVE DEPTH ---")
print(f"Average Branching Factor (b): {b:.2f}")
print(f"Calculated Delta*: {delta_star}")
print(f"Initial Beam Width: {retriever._initial_beam_width}")
print(f"Dynamic Beam Width (Prop 2 bound): {retriever.beam_width}")

# Let's perform a retrieval
print("\n--- RETRIEVING CONTEXT FOR 'target audience' ---")
# Manually skipping LLM anchor extraction by calling beam search directly
retriever.beam_width = 5 # reset for consistency
paths = retriever._beam_search(graph, "target_audience", delta_star)

context_subgraph = nx.DiGraph()
for path in paths:
    retriever._add_path_to_context(path, graph, context_subgraph)

result = retriever._linearize_context(context_subgraph)
print("Linearized Context:")
print(result)

