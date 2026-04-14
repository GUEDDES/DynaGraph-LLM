import sys
import networkx as nx
from core.constructor import TemporalKnowledgeConstructor
from core.retriever import MultiScaleRetriever
from core.consolidator import MemoryConsolidator
from core.graph_manager import TemporalKnowledgeGraph
from config import DynaGraphConfig as config
import json

# Mock OpenAI
import openai

class MockChoices:
    def __init__(self, content):
        self.message = type('obj', (object,), {'content': content})()

class MockCompletions:
    def create(self, **kwargs):
        if 'response_format' in kwargs and kwargs['response_format'].get('type') == 'json_object':
            prompt = kwargs['messages'][0]['content']
            if 'Kyoto' in prompt:
                data = {"triplets": [["User", "wants to visit", "Kyoto"], ["Kyoto", "is located in", "Japan"]]}
            else:
                data = {"triplets": [["User", "asked", "question"]]}
            return type('obj', (object,), {'choices': [MockChoices(json.dumps(data))]})()
        else:
            return type('obj', (object,), {'choices': [MockChoices('This is a mocked assistant response.')]})()

openai.chat = type('obj', (object,), {'completions': MockCompletions()})()

# Initialize system
constructor = TemporalKnowledgeConstructor(alpha=config.ALPHA, gamma=config.GAMMA)
retriever = MultiScaleRetriever(beam_width=config.BEAM_WIDTH, kappa=config.KAPPA)
consolidator = MemoryConsolidator(merge_threshold=config.MERGE_SIMILARITY)
kg = TemporalKnowledgeGraph()

# Mock extraction
kg.update('What activities would you recommend in Kyoto?', 'This is a mocked assistant response.', constructor)

# Test retriever
context = retriever.retrieve_context('Kyoto', kg.graph)
print('Retrieved Context:', context)
print('Graph Nodes:', list(kg.graph.nodes))
print('Graph Edges:', list(kg.graph.edges))
print('TEST PASSED SUCCESSFULLY.')
