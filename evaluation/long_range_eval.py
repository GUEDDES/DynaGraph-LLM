import random
import json
from tqdm import tqdm
from core.graph_manager import TemporalKnowledgeGraph
from core.constructor import TemporalKnowledgeConstructor
from core.retriever import MultiScaleRetriever

class LongRangeEvaluator:
    def __init__(self, test_file="test_cases.json"):
        self.test_cases = self.load_test_cases(test_file)
        self.constructor = TemporalKnowledgeConstructor()
        self.retriever = MultiScaleRetriever()
    
    def load_test_cases(self, path):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return self.generate_test_cases(100)
    
    def generate_test_cases(self, num_cases):
        cases = []
        for _ in range(num_cases):
            # Generate test case with background info and question
            background = [
                f"My friend {random.choice(['Alex', 'Taylor', 'Jordan'])} is a {random.choice(['doctor', 'engineer', 'teacher'])}",
                f"They live in {random.choice(['New York', 'London', 'Tokyo'])}",
                f"Their favorite hobby is {random.choice(['hiking', 'painting', 'cooking'])}"
            ]
            question = "What does my friend do for a living?"
            answer = background[0].split(" is a ")[1].rstrip('.')
            
            cases.append({
                "background": background,
                "question": question,
                "expected": answer
            })
        
        with open("test_cases.json", "w") as f:
            json.dump(cases, f, indent=2)
            
        return cases
    
    def run_evaluation(self, delta_values=(1, 2, 3, 4, 5)):
        results = {delta: {"correct": 0, "total": 0} for delta in delta_values}
        
        for case in tqdm(self.test_cases, desc="Evaluating"):
            graph = TemporalKnowledgeGraph()
            
            # Inject background information
            for info in case["background"]:
                graph.update(info, self.constructor)
            
            # Ask question with different delta values
            for delta in delta_values:
                context = self.retriever.retrieve_context(
                    case["question"], 
                    graph.graph, 
                    delta=delta
                )
                
                # Simple answer extraction (in real system would use LLM)
                answer_found = case["expected"].lower() in context.lower()
                
                if answer_found:
                    results[delta]["correct"] += 1
                results[delta]["total"] += 1
        
        # Calculate accuracy
        for delta in delta_values:
            if results[delta]["total"] > 0:
                results[delta]["accuracy"] = results[delta]["correct"] / results[delta]["total"]
            else:
                results[delta]["accuracy"] = 0.0
        
        return results

if __name__ == "__main__":
    evaluator = LongRangeEvaluator()
    results = evaluator.run_evaluation()
    
    print("\nLong-range Dependency Resolution Results:")
    print("Delta | Accuracy | Correct/Total")
    for delta, data in results.items():
        print(f"{delta:5} | {data['accuracy']:.3f}    | {data['correct']}/{data['total']}")