import networkx as nx
import numpy as np
from typing import Dict, Any

class TemporalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.turn_counter = 0
    
    def update(self, text: str, constructor) -> None:
        """Update the graph with new information from text"""
        self.turn_counter += 1
        self.graph = constructor.update_graph(
            self.graph, 
            text, 
            self.turn_counter
        )
    
    def consolidate(self, consolidator) -> None:
        """Apply memory consolidation to the graph"""
        self.graph = consolidator.online_consolidation(
            self.graph, 
            self.turn_counter
        )
    
    def offline_consolidation(self, consolidator) -> nx.Graph:
        """Perform offline abstraction and return abstracted graph"""
        return consolidator.offline_consolidation(self.graph)
    
    def get_graph_metrics(self) -> Dict[str, Any]:
        """Return metrics about the current graph state"""
        return {
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "density": nx.density(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "turn": self.turn_counter
        }
    
    def export_rdf(self) -> str:
        """Export graph to RDF-like format"""
        rdf_lines = []
        for u, v, data in self.graph.edges(data=True):
            rdf_lines.append(f"<{u}> <{data.get('predicate', 'related_to')}> <{v}> .")
        return "\n".join(rdf_lines)