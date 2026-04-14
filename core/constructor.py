import re
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
        # Formal Predicate Ontology (Def 1)
        return [
            "is-a", "has-property", "located-in", "part-of", 
            "related-to", "similar-to", "type-of", "caused-by",
            "used-for", "created-by", "belongs-to", "depends-on",
            "temporally-precedes", "interacts-with"
        ]
    
    def extract_triplets(self, text: str) -> list:
        prompt = f"""
        Extract key facts as a JSON list of [Subject, Predicate, Object] triplets.
        Focus on entities, their attributes, and relationships.
        
        Text: "{text}"
        
        Output format: {{"triplets": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}}
        """
        
        client = openai.OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
        response = client.chat.completions.create(
            model=config.TRIPLET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            content = response.choices[0].message.content
            # Make sure we reliably extract JSON if model prefixes strings
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group(0).strip())
                return result.get("triplets", [])
            return []
        except Exception:
            return []
    
    def map_predicate_to_ontology(self, predicate: str) -> tuple[str, float]:
        if not self.core_concepts:
            return predicate, 0.5
        
        pred_embed = self.semantic_model.encode([predicate])
        core_embeds = self.semantic_model.encode(self.core_concepts)
        similarities = cosine_similarity(pred_embed, core_embeds)[0]
        max_idx = np.argmax(similarities)
        
        # If similarity is too low, keep original but assign low relevance
        if similarities[max_idx] < 0.4:
            return predicate, similarities[max_idx]
            
        return self.core_concepts[max_idx], similarities[max_idx]
    
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
                
            # Map to formal ontology
            mapped_p, S_p = self.map_predicate_to_ontology(p)
            
            # Conflict Resolution: Temporal Precedence with Archiving
            last_update = turn
            archived_history = []
            if graph.has_edge(s, o):
                old_edge = graph.get_edge_data(s, o)
                last_update = old_edge.get('last_updated', turn)
                archived_history = old_edge.get('archived_history', [])
                
                # Archive the superseded fact if it differs
                if old_edge.get('predicate') != mapped_p and turn > last_update:
                    archived_history.append({
                        'predicate': old_edge.get('predicate'),
                        'weight': old_edge.get('weight'),
                        'turn': last_update
                    })
            
            # Calculate temporal-semantic weight (w_{t_2} > w_{t_1} enforced by decay logic)
            temporal_factor = np.exp(-self.gamma * (turn - last_update)) if turn > last_update else 1.0
            weight = self.alpha * S_p + (1 - self.alpha) * temporal_factor
            
            # Add/update edge
            graph.add_edge(s, o, predicate=mapped_p, weight=weight, last_updated=turn, archived_history=archived_history)
            
            # Update node timestamps
            graph.nodes[s]['last_updated'] = turn
            graph.nodes[o]['last_updated'] = turn
        
        return graph