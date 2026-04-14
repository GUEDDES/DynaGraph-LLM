import networkx as nx
import matplotlib.pyplot as plt
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
        combined_text = f"User: {user_input}\nAssistant: {response}"
        self.knowledge_graph.update(
            combined_text, 
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
        
        client = openai.OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
        response = client.chat.completions.create(
            model=config.MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    
    def _recent_history(self, window=3) -> str:
        recent = self.conversation_history[-window:]
        return "\n".join(
            f"Turn {item['turn']}: User: {item['user']}\nAssistant: {item['assistant']}" 
            for item in recent
        )

    def visualize_graph(self):
        """Displays the current Knowledge Graph using Matplotlib"""
        G = self.knowledge_graph.graph
        if len(G.nodes) == 0:
            print("[System] The graph is currently empty.")
            return

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=0.5)
        
        # Nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, alpha=0.8)
        
        # Edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=1.5, alpha=0.5)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', font_weight='bold')
        
        # Edge Labels
        edge_labels = {(u, v): d.get('predicate', '') for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        
        plt.title(f"DynaGraph-LLM Topology (Nodes: {len(G.nodes)}, Edges: {len(G.edges)})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import sys
    system = DynaGraphSystem()
    
    print("DynaGraph-LLM System initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if user_input.lower() in ['visualize', 'voir graph', 'voir', 'show graph']:
            system.visualize_graph()
            continue
            
        if user_input.lower() in ['consolider', 'consolidate']:
            print("\n[System] Performing memory consolidation...")
            old_size = len(system.knowledge_graph.graph.nodes)
            system.knowledge_graph.consolidate(system.consolidator)
            new_size = len(system.knowledge_graph.graph.nodes)
            print(f"[System] Consolidation finished: Nodes reduced from {old_size} to {new_size}.")
            system.visualize_graph()
            continue
            
        response = system.process_input(user_input)
        print(f"\nAssistant: {response}")
        print(f"\n[System] Turn {system.turn_count} completed | Graph size: {len(system.knowledge_graph.graph.nodes)} nodes")