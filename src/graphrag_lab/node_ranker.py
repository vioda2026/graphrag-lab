"""
Node Ranker Module for GraphRAG.

This module implements SPRIG (Structured Personalized Ranking with 
Iterative Graph traversal) - a personalized PageRank variant for 
ranking nodes by query relevance.

Key Algorithm: SPRIG personalized PageRank with query-aware personalization
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import networkx as nx


def sprig_rank(
    graph: nx.Graph,
    query_vector: np.ndarray,
    node_embeddings: Dict[Any, np.ndarray],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> List[Tuple[Any, float]]:
    """
    Rank nodes using SPRIG personalized PageRank.

    Args:
        graph: Input graph structure (networkx.Graph)
        query_vector: Query embedding vector (np.ndarray)
        node_embeddings: Dictionary mapping node_id to embedding vector
        alpha: Damping factor (default: 0.85)
        max_iter: Maximum iterations for PageRank convergence (default: 100)
        tol: Convergence tolerance (default: 1e-6)

    Returns:
        List of (node_id, relevance_score) tuples sorted by score descending

    Algorithm:
        1. Compute personalization vector based on query-node similarity
        2. Run personalized PageRank with query-aware restart probabilities
        3. Return ranked nodes with relevance scores
    """
    if not graph.nodes():
        return []
    
    nodes = list(graph.nodes())
    n = len(nodes)
    
    # Compute personalization vector (query-node similarity)
    personalization = np.zeros(n)
    for i, node in enumerate(nodes):
        if node in node_embeddings:
            # Cosine similarity between query and node embedding
            node_emb = node_embeddings[node]
            query_norm = np.linalg.norm(query_vector)
            node_norm = np.linalg.norm(node_emb)
            
            if query_norm > 0 and node_norm > 0:
                similarity = np.dot(query_vector, node_emb) / (query_norm * node_norm)
                personalization[i] = max(0, similarity)  # Ensure non-negative
    
    # Normalize personalization vector
    p_sum = personalization.sum()
    if p_sum > 0:
        personalization = personalization / p_sum
    else:
        # Uniform personalization if no embeddings available
        personalization = np.ones(n) / n
    
    # Initialize PageRank scores
    scores = np.ones(n) / n
    
    # Build adjacency matrix (row-normalized)
    adj_matrix = np.zeros((n, n))
    for i, node_i in enumerate(nodes):
        neighbors = list(graph.neighbors(node_i))
        if neighbors:
            for node_j in neighbors:
                j = nodes.index(node_j)
                adj_matrix[i, j] = 1.0 / len(neighbors)
    
    # Iterative PageRank computation
    for iteration in range(max_iter):
        new_scores = (1 - alpha) * personalization + alpha * np.dot(scores, adj_matrix)
        
        # Check convergence
        diff = np.abs(new_scores - scores).sum()
        scores = new_scores
        
        if diff < tol:
            break
    
    # Create ranked list
    ranked_nodes = [(nodes[i], float(scores[i])) for i in range(n)]
    ranked_nodes.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_nodes


def rank_with_community_boost(
    graph: nx.Graph,
    query_vector: np.ndarray,
    node_embeddings: Dict[Any, np.ndarray],
    community_map: Dict[Any, int],
    alpha: float = 0.85,
    community_boost: float = 0.1
) -> List[Tuple[Any, float]]:
    """
    Rank nodes with community-based score boosting.

    Args:
        graph: Input graph structure
        query_vector: Query embedding vector
        node_embeddings: Node embedding dictionary
        community_map: Community assignment from community_detector
        alpha: PageRank damping factor
        community_boost: Boost factor for nodes in same community

    Returns:
        List of (node_id, boosted_score) tuples sorted descending
    """
    # Get base SPRIG ranking
    base_ranking = sprig_rank(graph, query_vector, node_embeddings, alpha=alpha)
    
    if not base_ranking or not community_map:
        return base_ranking
    
    # Find top community (community of highest-ranked node with valid community)
    top_community = -1
    for node, score in base_ranking[:10]:  # Check top 10
        if node in community_map and community_map[node] >= 0:
            top_community = community_map[node]
            break
    
    # Apply community boost
    boosted_ranking = []
    for node, score in base_ranking:
        if node in community_map and community_map[node] == top_community:
            score *= (1 + community_boost)
        boosted_ranking.append((node, score))
    
    # Re-sort after boosting
    boosted_ranking.sort(key=lambda x: x[1], reverse=True)
    
    return boosted_ranking


def get_top_k_nodes(
    graph: nx.Graph,
    query_vector: np.ndarray,
    node_embeddings: Dict[Any, np.ndarray],
    k: int = 10
) -> List[Any]:
    """
    Get top-k most relevant nodes for a query.

    Args:
        graph: Input graph structure
        query_vector: Query embedding vector
        node_embeddings: Node embedding dictionary
        k: Number of top nodes to return

    Returns:
        List of top-k node_ids
    """
    ranked = sprig_rank(graph, query_vector, node_embeddings)
    return [node for node, score in ranked[:k]]


# ============================================================================
# Unit Test Scaffold
# ============================================================================

def _run_tests():
    """Run basic unit tests for node ranking."""
    # Test 1: Empty graph
    empty_graph = nx.Graph()
    query = np.array([1.0, 0.0, 0.0])
    result = sprig_rank(empty_graph, query, {})
    assert result == [], "Empty graph should return empty list"
    print("✓ Test 1 passed: Empty graph")
    
    # Test 2: Simple graph with embeddings
    g = nx.complete_graph(4)
    embeddings = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.8, 0.2, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
        3: np.array([0.0, 0.0, 1.0]),
    }
    query = np.array([1.0, 0.0, 0.0])
    result = sprig_rank(g, query, embeddings)
    
    assert len(result) == 4, "Should rank all nodes"
    assert result[0][0] == 0, "Most similar node should be ranked first"
    print("✓ Test 2 passed: Simple ranking")
    
    # Test 3: Top-k extraction
    top_k = get_top_k_nodes(g, query, embeddings, k=2)
    assert len(top_k) == 2, "Should return exactly k nodes"
    assert top_k[0] == 0, "Top node should be most similar"
    print("✓ Test 3 passed: Top-k extraction")
    
    # Test 4: Community boost
    community_map = {0: 0, 1: 0, 2: 1, 3: 1}
    boosted = rank_with_community_boost(g, query, embeddings, community_map)
    assert len(boosted) == 4, "Should rank all nodes with boost"
    print("✓ Test 4 passed: Community boost")
    
    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    _run_tests()
