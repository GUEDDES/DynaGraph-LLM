import json
import numpy as np
from tqdm import tqdm
from core.graph_manager import TemporalKnowledgeGraph
from core.constructor import TemporalKnowledgeConstructor
from core.retriever import MultiScaleRetriever

class CoherenceEvaluator:
    def __init__(self, dataset="conversations.json"):
        self.dataset = self.load_dataset(dataset)
    
    def load_dataset(self, path):
        with open(path) as f:
            return json.load(f)
    
    def calculate_coherence(self, conversation, use_graph=True):
        graph = TemporalKnowledgeGraph()
        constructor = TemporalKnowledgeConstructor()
        retriever = MultiScaleRetriever()
        
        coherence_scores = []
        contradictions = 0
        
        for i, turn in enumerate(conversation):
            user_input = turn["user"]
            
            if use_graph:
                context = retriever.retrieve_context(user_input, graph.graph)
                # In real evaluation, we would generate response and check
                # Here we simulate by checking for contradictions in context
                
                # Check for contradictions
                if self._check_contradiction(user_input, context):
                    contradictions += 1
            else:
                context = ""
            
            # Update graph with this turn
            if i > 0:  # Skip first turn
                prev_turn = conversation[i-1]
                graph.update(
                    f"User: {prev_turn['user']}\nAssistant: {prev_turn['assistant']}", 
                    constructor
                )
            
            # Calculate turn coherence (simplified)
            if context:
                coherence = min(1.0, len(context.split()) / 100)  # Normalized context richness
            else:
                coherence = 0.0
                
            coherence_scores.append(coherence)
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
        contradiction_rate = contradictions / len(conversation) if conversation else 0
        
        return {
            "avg_coherence": avg_coherence,
            "contradiction_rate": contradiction_rate,
            "graph_size": len(graph.graph.nodes)
        }
    
    def _check_contradiction(self, statement: str, context: str) -> bool:
        # Simplified contradiction detection
        negative_words = {"not", "never", "no", "none", "nothing"}
        statement_words = set(statement.lower().split())
        
        # Check for negation conflicts
        if negative_words & statement_words:
            if " but " in context or " however " in context:
                return True
        return False
    
    def run_evaluation(self):
        results = {"with_graph": [], "without_graph": []}
        
        for conv in tqdm(self.dataset, desc="Evaluating coherence"):
            # With graph memory
            with_graph = self.calculate_coherence(conv, use_graph=True)
            results["with_graph"].append(with_graph)
            
            # Without graph memory
            without_graph = self.calculate_coherence(conv, use_graph=False)
            results["without_graph"].append(without_graph)
        
        # Aggregate results
        aggregated = {}
        for key in results:
            coherences = [r["avg_coherence"] for r in results[key]]
            contradictions = [r["contradiction_rate"] for r in results[key]]
            sizes = [r["graph_size"] for r in results[key]]
            
            aggregated[key] = {
                "mean_coherence": np.mean(coherences),
                "mean_contradiction": np.mean(contradictions),
                "avg_graph_size": np.mean(sizes)
            }
        
        return aggregated

if __name__ == "__main__":
    evaluator = CoherenceEvaluator()
    results = evaluator.run_evaluation()
    
    print("\nConversational Coherence Results:")
    print("Condition        | Avg. Coherence | Contradiction Rate | Avg. Graph Size")
    print("-" * 65)
    for condition, data in results.items():
        print(f"{condition:15} | {data['mean_coherence']:.4f}         | {data['mean_contradiction']:.4f}            | {data['avg_graph_size']:.1f}")