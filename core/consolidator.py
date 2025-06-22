import networkx as nx
import community  # python-louvain
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DynaGraphConfig as config

class MemoryConsolidator:
    def __init__(self, merge_threshold=0.8):
        self.merge_threshold = merge_threshold
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def online_consolidation(self, graph: nx.Graph, current_turn: int) -> nx.Graph:
        """Perform online pruning and merging"""
        # Prune low-centrality nodes
        centrality = nx.betweenness_centrality(graph)
        nodes_to_remove = [
            node for node in graph.nodes 
            if centrality.get(node, 0) < config.PRUNE_THRESHOLD
            and current_turn - graph.nodes[node].get('last_updated', current_turn) > 10
        ]
        graph.remove_nodes_from(nodes_to_remove)
        
        # Merge similar nodes
        self._merge_similar_nodes(graph)
        return graph
    
    def _merge_similar_nodes(self, graph: nx.Graph):
        """Merge nodes with high semantic similarity"""
        nodes = list(graph.nodes)
        embeddings = self.embedding_model.encode(nodes)
        
        # Create similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
        
        # Find pairs to merge
        to_merge = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if similarity_matrix[i, j] > self.merge_threshold:
                    to_merge.append((nodes[i], nodes[j]))
        
        # Merge nodes
        for node1, node2 in to_merge:
            if node1 in graph and node2 in graph:
                # Merge node2 into node1
                nx.contracted_nodes(graph, node1, node2, self_loops=False)
    
    def offline_consolidation(self, graph: nx.Graph) -> nx.Graph:
        """Perform community-based graph abstraction"""
        if len(graph.nodes) < 10:
            return graph  # No need for abstraction on small graphs
            
        # Convert to undirected for community detection
        undirected = graph.to_undirected()
        partition = community.best_partition(undirected, resolution=config.COMMUNITY_RESOLUTION)
        
        # Create super nodes
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        
        super_graph = nx.DiGraph()
        super_nodes = {}
        
        # Create super nodes (representative: highest centrality node)
        for comm_id, nodes in communities.items():
            subgraph = graph.subgraph(nodes)
            if not subgraph.nodes:
                continue
                
            centrality = nx.degree_centrality(subgraph)
            representative = max(centrality, key=centrality.get)
            super_nodes[comm_id] = representative
            super_graph.add_node(representative, community=nodes)
        
        # Add edges between communities
        for comm1 in communities:
            for comm2 in communities:
                if comm1 == comm2:
                    continue
                    
                # Calculate aggregate weight
                weights = []
                for node1 in communities[comm1]:
                    for node2 in communities[comm2]:
                        if graph.has_edge(node1, node2):
                            weights.append(graph[node1][node2].get('weight', 0))
                
                if weights:
                    avg_weight = sum(weights) / len(weights)
                    if avg_weight > 0.3:  # Minimum connection threshold
                        super_graph.add_edge(
                            super_nodes[comm1], 
                            super_nodes[comm2],
                            weight=avg_weight,
                            type="inter_community"
                        )
        
        return super_graph