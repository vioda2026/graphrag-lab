"""
End-to-End Tests for Node Ranker Module (SPRIG).

Tests personalized PageRank-based node ranking in realistic GraphRAG scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx
from graphrag_lab.node_ranker import sprig_rank, rank_with_community_boost, get_top_k_nodes


def test_e2e_query_aware_ranking():
    """Test SPRIG ranking with query-aware personalization."""
    # Create a small knowledge graph
    g = nx.Graph()
    g.add_edges_from([
        ('ML', 'DL'), ('DL', 'NN'), ('NN', 'CNN'),
        ('ML', 'NLP'), ('NLP', 'Transformer'), ('Transformer', 'BERT')
    ])
    
    # Create query embedding (query about "deep learning")
    np.random.seed(42)
    query_vector = np.random.randn(10)
    
    # Create node embeddings where DL-related nodes have higher similarity
    node_embeddings = {
        'ML': np.random.randn(10),
        'DL': query_vector * 0.9 + np.random.randn(10) * 0.1,
        'NN': query_vector * 0.7 + np.random.randn(10) * 0.3,
        'CNN': query_vector * 0.5 + np.random.randn(10) * 0.5,
        'NLP': np.random.randn(10),
        'Transformer': np.random.randn(10),
        'BERT': np.random.randn(10)
    }
    
    # Run SPRIG ranking
    ranked = sprig_rank(g, query_vector, node_embeddings, max_iter=50)
    
    # Verify ranking is returned
    assert len(ranked) == 7, f"Expected 7 nodes, got {len(ranked)}"
    
    # Verify DL-related nodes rank higher
    top_3 = [node for node, _ in ranked[:3]]
    assert 'DL' in top_3, "DL should be in top 3 (high query similarity)"
    
    # Verify scores are normalized (sum to ~1)
    total_score = sum(score for _, score in ranked)
    assert 0.9 < total_score < 1.1, f"Scores should sum to ~1, got {total_score}"
    
    print("✓ Test 1 passed: Query-aware ranking")
    return True


def test_e2e_community_boosted_ranking():
    """Test ranking with community-based score boosting."""
    # Create graph with 2 communities
    g = nx.Graph()
    
    # Community 1: AI/ML
    g.add_edges_from([
        ('ML', 'DL'), ('DL', 'NN'), ('NN', 'CNN'), ('ML', 'NN')
    ])
    
    # Community 2: NLP
    g.add_edges_from([
        ('NLP', 'Transformer'), ('Transformer', 'BERT'), ('BERT', 'GPT'), ('NLP', 'BERT')
    ])
    
    # Weak connection
    g.add_edge('ML', 'NLP')
    
    # Community assignments
    communities = {
        'ML': 0, 'DL': 0, 'NN': 0, 'CNN': 0,
        'NLP': 1, 'Transformer': 1, 'BERT': 1, 'GPT': 1
    }
    
    # Query about ML
    np.random.seed(42)
    query_vector = np.random.randn(10)
    node_embeddings = {node: np.random.randn(10) for node in g.nodes()}
    node_embeddings['ML'] = query_vector * 0.9 + np.random.randn(10) * 0.1
    
    # Run community-boosted ranking
    ranked = rank_with_community_boost(
        g, query_vector, node_embeddings, communities, community_boost=0.1
    )
    
    # Verify ranking is returned
    assert len(ranked) == 8, f"Expected 8 nodes, got {len(ranked)}"
    
    # Verify ML community nodes get some boost
    top_5 = [node for node, _ in ranked[:5]]
    ml_community_nodes = {'ML', 'DL', 'NN', 'CNN'}
    ml_in_top5 = sum(1 for node in top_5 if node in ml_community_nodes)
    
    assert ml_in_top5 >= 2, f"Expected >=2 ML community nodes in top 5, got {ml_in_top5}"
    
    print("✓ Test 2 passed: Community-boosted ranking")
    return True


def test_e2e_top_k_extraction():
    """Test top-k node extraction with various k values."""
    # Create a graph
    g = nx.complete_graph(10)
    
    np.random.seed(42)
    query_vector = np.random.randn(10)
    node_embeddings = {i: np.random.randn(10) for i in range(10)}
    
    # Test various k values using get_top_k_nodes
    for k in [1, 3, 5, 10]:
        top_k = get_top_k_nodes(g, query_vector, node_embeddings, k=k)
        assert len(top_k) == k, f"Expected {k} nodes, got {len(top_k)}"
    
    # Test k > n_nodes
    top_k = get_top_k_nodes(g, query_vector, node_embeddings, k=20)
    assert len(top_k) == 10, f"Expected 10 nodes (all), got {len(top_k)}"
    
    print("✓ Test 3 passed: Top-k extraction")
    return True


def test_e2e_convergence_behavior():
    """Test SPRIG convergence behavior."""
    # Create a graph
    g = nx.erdos_renyi_graph(50, 0.3)
    
    np.random.seed(42)
    query_vector = np.random.randn(20)
    node_embeddings = {i: np.random.randn(20) for i in g.nodes()}
    
    # Run with different iteration limits
    ranked_10 = sprig_rank(g, query_vector, node_embeddings, max_iter=10)
    ranked_50 = sprig_rank(g, query_vector, node_embeddings, max_iter=50)
    ranked_100 = sprig_rank(g, query_vector, node_embeddings, max_iter=100)
    
    # Verify convergence (results should stabilize)
    top_5_10 = [node for node, _ in ranked_10[:5]]
    top_5_50 = [node for node, _ in ranked_50[:5]]
    top_5_100 = [node for node, _ in ranked_100[:5]]
    
    # At least 3/5 should be stable between 50 and 100 iterations
    stable_50_100 = len(set(top_5_50) & set(top_5_100))
    assert stable_50_100 >= 3, f"Expected >=3 stable nodes between 50/100 iters, got {stable_50_100}"
    
    print("✓ Test 4 passed: Convergence behavior")
    return True


def test_e2e_empty_and_edge_cases():
    """Test edge cases in SPRIG ranking."""
    np.random.seed(42)
    query_vector = np.random.randn(10)
    
    # Empty graph
    g_empty = nx.Graph()
    ranked_empty = sprig_rank(g_empty, query_vector, {})
    assert len(ranked_empty) == 0, "Empty graph should return empty ranking"
    
    # Single node
    g_single = nx.Graph()
    g_single.add_node('single')
    ranked_single = sprig_rank(g_single, query_vector, {'single': np.random.randn(10)})
    assert len(ranked_single) == 1, "Single node graph should return 1 node"
    
    # No embeddings (should use uniform personalization)
    g = nx.path_graph(5)
    ranked_no_emb = sprig_rank(g, query_vector, {})
    assert len(ranked_no_emb) == 5, "Should rank all nodes even without embeddings"
    
    print("✓ Test 5 passed: Empty and edge cases")
    return True


def run_all_tests():
    """Run all e2e tests."""
    print("=" * 60)
    print("Running Node Ranker (SPRIG) E2E Tests")
    print("=" * 60)
    
    tests = [
        test_e2e_query_aware_ranking,
        test_e2e_community_boosted_ranking,
        test_e2e_top_k_extraction,
        test_e2e_convergence_behavior,
        test_e2e_empty_and_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
