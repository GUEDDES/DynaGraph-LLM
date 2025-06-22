import json
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from config import DynaGraphConfig as config

class TemporalKnowledgeConstructor:
    def __init__(self, alpha=0.7, gamma=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.core_concepts = self._load_core_concepts()
        
    def _load_core_concepts(self):
        return [
            "is a", "has property", "located in", "part of", 
            "related to", "similar to", "type of", "causes",
            "used for", "created by", "belongs to", "depends on"
        ]
    
    def extract_triplets(self, text: str) -> list:
        prompt = f"""
        Extract key facts as a JSON list of [Subject, Predicate, Object] triplets.
        Focus on entities, their attributes, and relationships.
        
        Text: "{text}"
        
        Output format: {{"triplets": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}}
        """
        
        response = openai.chat.completions.create(
            model=config.TRIPLET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("triplets", [])
        except json.JSONDecodeError:
            return []
    
    def compute_semantic_relevance(self, predicate: str) -> float:
        if not self.core_concepts:
            return 0.5
        
        pred_embed = self.semantic_model.encode([predicate])
        core_embeds = self.semantic_model.encode(self.core_concepts)
        similarities = cosine_similarity(pred_embed, core_embeds)
        return np.max(similarities)
    
    def update_graph(self, graph: nx.DiGraph, text: str, turn: int) -> nx.DiGraph:
        triplets = self.extract_triplets(text)
        
        for triplet in triplets:
            if len(triplet) != 3:
                continue
                
            s, p, o = triplet
            
            # Create or update nodes
            if s not in graph.nodes:
                graph.add_node(s, last_updated=turn, created=turn, centrality=0.0)
            if o not in graph.nodes:
                graph.add_node(o, last_updated=turn, created=turn, centrality=0.0)
                
            # Get last update time if edge exists
            last_update = turn
            if graph.has_edge(s, o):
                edge_data = graph.get_edge_data(s, o)
                last_update = edge_data.get('last_updated', turn)
            
            # Calculate temporal-semantic weight
            S_p = self.compute_semantic_relevance(p)
            temporal_factor = np.exp(-self.gamma * (turn - last_update))
            weight = self.alpha * S_p + (1 - self.alpha) * temporal_factor
            
            # Add/update edge
            graph.add_edge(s, o, predicate=p, weight=weight, last_updated=turn)
            
            # Update node timestamps
            graph.nodes[s]['last_updated'] = turn
            graph.nodes[o]['last_updated'] = turn
        
        return graph