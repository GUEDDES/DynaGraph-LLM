import time
import numpy as np
from tqdm import tqdm
from core import DynaGraphSystem

class CognitiveLoadEvaluator:
    def __init__(self, conversation_dataset="long_conversations.json"):
        self.dataset = self.load_dataset(conversation_dataset)
    
    def load_dataset(self, path):
        # Implement actual loading logic
        return [["Message 1", "Message 2", ...] for _ in range(10)]
    
    def measure_efficiency(self, delta_values=(1, 2, 3, 4, 5)):
        results = {}
        for delta in delta_values:
            latencies = []
            memory_usages = []
            
            for conversation in tqdm(self.dataset, desc=f"Testing δ={delta}"):
                system = DynaGraphSystem()
                start_time = time.perf_counter()
                
                for message in conversation:
                    response = system.process_input(message, delta=delta)
                    # Memory measurement would use resource tracking in real implementation
                    memory_usages.append(len(system.knowledge_graph.graph.nodes))
                
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)
            
            results[delta] = {
                "avg_latency": np.mean(latencies),
                "avg_memory_nodes": np.mean(memory_usages),
                "efficiency_ratio": np.mean(memory_usages) / np.mean(latencies)
            }
        
        return results

if __name__ == "__main__":
    evaluator = CognitiveLoadEvaluator()
    results = evaluator.measure_efficiency()
    
    print("Cognitive Load Evaluation Results:")
    print("Delta | Avg Latency (s) | Avg Memory Nodes | Efficiency Ratio")
    for delta, metrics in results.items():
        print(f"{delta:5} | {metrics['avg_latency']:15.4f} | {metrics['avg_memory_nodes']:17.0f} | {metrics['efficiency_ratio']:15.2f}")