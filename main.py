import networkx as nx
from core.constructor import TemporalKnowledgeConstructor
from core.retriever import MultiScaleRetriever
from core.consolidator import MemoryConsolidator
from core.graph_manager import TemporalKnowledgeGraph
import openai
from config import DynaGraphConfig as config

class DynaGraphSystem:
    def __init__(self):
        self.constructor = TemporalKnowledgeConstructor(
            alpha=config.ALPHA,
            gamma=config.GAMMA
        )
        self.retriever = MultiScaleRetriever(
            beam_width=config.BEAM_WIDTH,
            kappa=config.KAPPA
        )
        self.consolidator = MemoryConsolidator(
            merge_threshold=config.MERGE_SIMILARITY
        )
        self.knowledge_graph = TemporalKnowledgeGraph()
        self.conversation_history = []
        self.turn_count = 0
        
    def process_input(self, user_input: str) -> str:
        # Retrieve relevant context
        context = self.retriever.retrieve_context(
            user_input, 
            self.knowledge_graph.graph
        )
        
        # Generate response with context
        response = self._generate_response(user_input, context)
        
        # Update conversation history
        self.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'turn': self.turn_count
        })
        
        # Update knowledge graph
        self.knowledge_graph.update(
            user_input, 
            response, 
            self.constructor
        )
        self.turn_count += 1
        
        # Periodic consolidation
        if self.turn_count % config.REWIRING_INTERVAL == 0:
            self.knowledge_graph.consolidate(self.consolidator)
            
        return response
    
    def _generate_response(self, user_input: str, context: str) -> str:
        prompt = f"""
        [LONG-TERM CONTEXT]
        {context}
        
        [CONVERSATION HISTORY]
        {self._recent_history()}
        
        [USER QUERY]
        {user_input}
        
        [ASSISTANT RESPONSE]
        """
        
        response = openai.chat.completions.create(
            model=config.MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    
    def _recent_history(self, window=3) -> str:
        recent = self.conversation_history[-window:]
        return "\n".join(
            f"Turn {item['turn']}: User: {item['user']}\nAssistant: {item['assistant']}" 
            for item in recent
        )

if __name__ == "__main__":
    import sys
    system = DynaGraphSystem()
    
    print("DynaGraph-LLM System initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = system.process_input(user_input)
        print(f"\nAssistant: {response}")
        print(f"\n[System] Turn {system.turn_count} completed | Graph size: {len(system.knowledge_graph.graph.nodes)} nodes")