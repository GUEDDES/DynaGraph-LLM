import sys
import copy
from core.constructor import TemporalKnowledgeConstructor
from core.retriever import MultiScaleRetriever
from core.consolidator import MemoryConsolidator
from core.graph_manager import TemporalKnowledgeGraph
import networkx as nx
from config import DynaGraphConfig as config
import json

import openai

# Mocking the OpenAI responses to simulate a real conversation
class MockChoices:
    def __init__(self, content):
        self.message = type('obj', (object,), {'content': content})()

class MockCompletions:
    def create(self, **kwargs):
        if 'response_format' in kwargs:
            prompt = kwargs['messages'][0]['content'].lower()
            if 'quantum' in prompt:
                data = {"triplets": [["User", "interest", "Quantum physics"], ["Quantum physics", "is a branch of", "Science"]]}
            elif 'paris' in prompt:
                data = {"triplets": [["User", "wants to visit", "Paris"], ["Paris", "is in", "France"]]}
            else:
                data = {"triplets": [["User", "said", "hello"]]}
            return type('obj', (object,), {'choices': [MockChoices(json.dumps(data))]})()
        else:
            return type('obj', (object,), {'choices': [MockChoices('This is the intelligent mocked response from DynaGraph!')]})()

openai.chat = type('obj', (object,), {'completions': MockCompletions()})()

# Simulate the DynaGraphSystem (simplified main.py loop)
class DynaGraphSystem:
    def __init__(self):
        self.constructor = TemporalKnowledgeConstructor(alpha=config.ALPHA, gamma=config.GAMMA)
        self.retriever = MultiScaleRetriever(beam_width=config.BEAM_WIDTH, kappa=config.KAPPA)
        self.consolidator = MemoryConsolidator(merge_threshold=config.MERGE_SIMILARITY)
        self.knowledge_graph = TemporalKnowledgeGraph()
        self.turn_count = 1

    def process_input(self, user_input):
        print(f"\n[Turn {self.turn_count}] User: {user_input}")
        
        context = self.retriever.retrieve_context(user_input, self.knowledge_graph.graph)
        print(f" -> Retrieved Context (Delta* bound applied):\n{context if context else '   (No context found)'}")
        
        # mock response
        response = "Mocked LLM Response based on context."
        
        combined_text = f"User: {user_input}\nAssistant: {response}"
        self.knowledge_graph.update(combined_text, self.constructor)
        print(f" -> Graph Updated! Current Nodes: {len(self.knowledge_graph.graph.nodes)}, Edges: {len(self.knowledge_graph.graph.edges)}")
        
        self.turn_count += 1
        return response

system = DynaGraphSystem()
system.process_input("Can you teach me about Quantum physics?")
system.process_input("Where is a good place to study it, maybe Paris?")
system.process_input("What was the first subject we discussed?")

print("\n--- FINAL GRAPH STATE ---")
for u, v, data in system.knowledge_graph.graph.edges(data=True):
    print(f"{u} --[{data['predicate']}]--> {v} (weight: {data['weight']:.2f}, turn: {data['last_updated']})")
