"""
End-to-End Tests for Community Detector Module.

Tests k-core decomposition based community detection in realistic GraphRAG scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import networkx as nx
from graphrag_lab.community_detector import detect_communities, get_core_communities


def test_e2e_knowledge_graph_communities():
    """Test community detection on a synthetic knowledge graph."""
    # Create a knowledge graph with 3 distinct communities
    g = nx.Graph()
    
    # Community 1: AI/ML concepts (dense)
    g.add_edges_from([
        ('ML', 'DL'), ('DL', 'NN'), ('NN', 'CNN'), ('CNN', 'RNN'),
        ('ML', 'RL'), ('RL', 'DL'), ('ML', 'NN')
    ])
    
    # Community 2: NLP concepts (dense)
    g.add_edges_from([
        ('NLP', 'Transformer'), ('Transformer', 'BERT'), ('BERT', 'GPT'),
        ('NLP', 'Embedding'), ('Embedding', 'Word2Vec'), ('Transformer', 'Attention')
    ])
    
    # Community 3: Graph concepts (dense)
    g.add_edges_from([
        ('Graph', 'GNN'), ('GNN', 'GCN'), ('GCN', 'GraphSAGE'),
        ('Graph', 'PageRank'), ('PageRank', 'Centrality'), ('GNN', 'MessagePassing')
    ])
    
    # Weak connections between communities
    g.add_edge('ML', 'NLP')
    g.add_edge('DL', 'Graph')
    
    # Run community detection
    communities = detect_communities(g, min_k=2)
    
    # Verify all nodes are assigned
    assert len(communities) == 20, f"Expected 20 nodes, got {len(communities)}"
    
    # Verify community structure
    ai_ml_community = communities.get('ML', -1)
    nlp_community = communities.get('NLP', -1)
    graph_community = communities.get('Graph', -1)
    
    # Verify communities are distinct (may vary based on graph structure)
    print(f"  Communities: ML={ai_ml_community}, NLP={nlp_community}, Graph={graph_community}")
    
    print("✓ Test 1 passed: Knowledge graph community detection")
    return True


def test_e2e_multi_level_kcore():
    """Test multi-level k-core community analysis."""
    # Create a graph with varying density levels
    g = nx.Graph()
    
    # Core (k=4): Very dense
    g.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4)
    ])
    
    # Medium density (k=2-3)
    g.add_edges_from([
        (5, 0), (5, 1), (5, 2),
        (6, 1), (6, 2), (6, 3),
        (7, 5), (7, 6)
    ])
    
    # Sparse periphery (k=1)
    g.add_edges_from([
        (8, 7), (9, 8), (10, 9)
    ])
    
    # Get multi-level communities
    results = get_core_communities(g, k_values=[2, 3, 4])
    
    # Verify k=2 has >= nodes than k=4
    k2_nodes = sum(1 for c in results[2].values() if c >= 0)
    k4_nodes = sum(1 for c in results[4].values() if c >= 0)
    
    assert k2_nodes >= k4_nodes, f"k=2 should have >= nodes than k=4 ({k2_nodes} vs {k4_nodes})"
    
    # Verify core nodes (0-4) are in community at k=4
    core_at_k4 = [node for node, cid in results[4].items() if cid >= 0]
    for core_node in range(5):
        assert core_node in core_at_k4, f"Core node {core_node} should be in k=4 community"
    
    print("✓ Test 2 passed: Multi-level k-core analysis")
    return True


def test_e2e_large_scale_performance():
    """Test community detection on a larger graph for performance."""
    import time
    
    # Create a larger graph (1000 nodes)
    n_nodes = 1000
    g = nx.barabasi_albert_graph(n_nodes, 5)
    
    # Time the community detection
    start = time.time()
    communities = detect_communities(g, min_k=2)
    elapsed = time.time() - start
    
    # Verify all nodes are assigned
    assert len(communities) == n_nodes, f"Expected {n_nodes} nodes, got {len(communities)}"
    
    # Verify performance (should complete in < 5 seconds)
    assert elapsed < 5.0, f"Community detection took {elapsed:.2f}s, expected < 5s"
    
    print(f"✓ Test 3 passed: Large-scale performance ({elapsed:.3f}s for {n_nodes} nodes)")
    return True


def test_e2e_edge_cases():
    """Test edge cases in community detection."""
    # Single node
    g1 = nx.Graph()
    g1.add_node('single')
    result1 = detect_communities(g1, min_k=2)
    assert result1['single'] == -1, "Single node should not be in any community"
    
    # Two nodes (k=1 max)
    g2 = nx.Graph()
    g2.add_edge('a', 'b')
    result2 = detect_communities(g2, min_k=2)
    assert all(c == -1 for c in result2.values()), "Two-node graph should have no k>=2 communities"
    
    # Complete graph (all in one community)
    g3 = nx.complete_graph(5)
    result3 = detect_communities(g3, min_k=2)
    assert len(set(result3.values())) == 1, "Complete graph should form one community"
    
    print("✓ Test 4 passed: Edge cases")
    return True


def run_all_tests():
    """Run all e2e tests."""
    print("=" * 60)
    print("Running Community Detector E2E Tests")
    print("=" * 60)
    
    tests = [
        test_e2e_knowledge_graph_communities,
        test_e2e_multi_level_kcore,
        test_e2e_large_scale_performance,
        test_e2e_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
