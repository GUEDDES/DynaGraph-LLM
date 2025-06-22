import spacy
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from config import DynaGraphConfig as config

class MultiScaleRetriever:
    def __init__(self, beam_width=3, kappa=0.8):
        self.beam_width = beam_width
        self.kappa = kappa
        self.nlp = spacy.load("en_core_web_sm")
    
    def identify_anchor_nodes(self, query: str, graph: nx.Graph) -> List[str]:
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]
        
        anchor_nodes = []
        for entity in entities[:config.MAX_ANCHORS]:
            best_match, best_score = None, 0.0
            for node in graph.nodes:
                score = self._string_similarity(entity, node)
                if score > best_score and score > 0.4:
                    best_score = score
                    best_match = node
            if best_match:
                anchor_nodes.append(best_match)
        return anchor_nodes
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        if s1_lower == s2_lower:
            return 1.0
            
        # Token overlap similarity
        tokens1 = set(s1_lower.split())
        tokens2 = set(s2_lower.split())
        overlap = len(tokens1 & tokens2) / max(len(tokens1), len(tokens2))
        
        # Prefix/suffix similarity
        prefix = 1.0 if s1_lower.startswith(s2_lower[:3]) or s2_lower.startswith(s1_lower[:3]) else 0.0
        suffix = 1.0 if s1_lower.endswith(s2_lower[-3:]) or s2_lower.endswith(s1_lower[-3:]) else 0.0
        
        return max(overlap, prefix, suffix)
    
    def retrieve_context(self, query: str, graph: nx.Graph, delta: int = None) -> str:
        if not graph.nodes:
            return ""
            
        # Determine cognitive depth
        if delta is None:
            delta = self._determine_cognitive_depth(graph)
        
        anchor_nodes = self.identify_anchor_nodes(query, graph)
        if not anchor_nodes:
            return ""
            
        context_subgraph = nx.DiGraph()
        
        for anchor in anchor_nodes:
            paths = self._beam_search(graph, anchor, delta)
            for path in paths:
                self._add_path_to_context(path, graph, context_subgraph)
                
        return self._linearize_context(context_subgraph)
    
    def _determine_cognitive_depth(self, graph: nx.Graph) -> int:
        """Dynamically set delta based on graph complexity"""
        num_nodes = len(graph.nodes)
        complexity_factor = min(1.0, num_nodes / 100)
        return max(config.DELTA_RANGE[0], 
                 min(config.DELTA_RANGE[1], 
                     int(config.DELTA_RANGE[0] + complexity_factor * (config.DELTA_RANGE[1] - config.DELTA_RANGE[0]))))
    
    def _beam_search(self, graph: nx.Graph, start: str, depth: int) -> List[List[str]]:
        beam = [([start], 0.0)]  # (path, cumulative score)
        all_paths = []
        
        for _ in range(depth):
            new_beam = []
            
            for path, score in beam:
                current = path[-1]
                neighbors = list(graph.neighbors(current))
                
                for neighbor in neighbors:
                    if neighbor in path:  # Avoid cycles
                        continue
                        
                    # Get edge weight and neighbor degree
                    edge_weight = graph[current][neighbor].get('weight', 0.1)
                    neighbor_degree = graph.degree(neighbor)
                    
                    # Calculate traversal probability
                    traversal_prob = (edge_weight * (neighbor_degree + 1) ** self.kappa)
                    new_score = score + np.log(traversal_prob)  # Log probability
                    
                    new_path = path + [neighbor]
                    new_beam.append((new_path, new_score))
            
            if not new_beam:
                break
                
            # Select top paths
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]
            all_paths.extend([path for path, _ in beam])
            
        return all_paths
    
    def _add_path_to_context(self, path: List[str], source_graph: nx.Graph, context_graph: nx.DiGraph):
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not context_graph.has_edge(u, v):
                edge_data = source_graph.get_edge_data(u, v)
                context_graph.add_edge(u, v, **edge_data)
    
    def _linearize_context(self, graph: nx.DiGraph) -> str:
        context_lines = []
        for u, v, data in graph.edges(data=True):
            context_lines.append(f"{u} --[{data.get('predicate', 'related to')}]-> {v}")
        return "\n".join(context_lines) if context_lines else "No relevant context found"