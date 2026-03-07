"""
Community Detector Module for GraphRAG.

This module implements k-core decomposition based community detection
for identifying densely connected subgraphs in knowledge graphs.

Key Algorithm: k-core decomposition + community clustering
"""

from typing import Dict, Any
import networkx as nx


def detect_communities(graph: nx.Graph, min_k: int = 2) -> Dict[Any, int]:
    """
    Detect communities using k-core decomposition and clustering.

    Args:
        graph: Input graph structure (networkx.Graph)
        min_k: Minimum k-core level to consider (default: 2)

    Returns:
        Dictionary mapping node_id to community_id

    Algorithm:
        1. Perform k-core decomposition to identify dense subgraphs
        2. Extract k-cores for k >= min_k
        3. Assign community IDs based on connected components in k-cores
        4. Nodes not in any k-core get community_id = -1
    """
    if not graph.nodes():
        return {}

    # Initialize community assignment
    community_map: Dict[Any, int] = {}
    
    # Get k-core decomposition
    core_numbers = nx.core_number(graph)
    
    # Find maximum k for filtering
    max_k = max(core_numbers.values()) if core_numbers else 0
    
    if max_k < min_k:
        # No dense communities found, assign all to default
        for node in graph.nodes():
            community_map[node] = -1
        return community_map

    # Extract nodes in k-cores (k >= min_k)
    core_nodes = set(node for node, k in core_numbers.items() if k >= min_k)
    
    # Create subgraph of core nodes
    if not core_nodes:
        for node in graph.nodes():
            community_map[node] = -1
        return community_map
    
    core_subgraph = graph.subgraph(core_nodes)
    
    # Find connected components as communities
    community_id = 0
    for component in nx.connected_components(core_subgraph):
        for node in component:
            community_map[node] = community_id
        community_id += 1
    
    # Assign non-core nodes to -1 (not in any community)
    for node in graph.nodes():
        if node not in community_map:
            community_map[node] = -1

    return community_map


def get_core_communities(
    graph: nx.Graph, 
    k_values: list[int] | None = None
) -> Dict[int, Dict[Any, int]]:
    """
    Get community assignments for multiple k-core levels.

    Args:
        graph: Input graph structure
        k_values: List of k values to analyze (default: [2, 3, 4, 5])

    Returns:
        Dictionary mapping k-value to community assignment dict
    """
    if k_values is None:
        k_values = [2, 3, 4, 5]
    
    results: Dict[int, Dict[Any, int]] = {}
    for k in k_values:
        results[k] = detect_communities(graph, min_k=k)
    
    return results


# ============================================================================
# Unit Test Scaffold
# ============================================================================

def _run_tests():
    """Run basic unit tests for community detection."""
    import sys
    
    # Test 1: Empty graph
    empty_graph = nx.Graph()
    result = detect_communities(empty_graph)
    assert result == {}, "Empty graph should return empty dict"
    print("✓ Test 1 passed: Empty graph")
    
    # Test 2: Simple triangle (should form one community)
    triangle = nx.complete_graph(3)
    result = detect_communities(triangle, min_k=2)
    assert len(set(result.values())) == 1, "Triangle should form one community"
    assert all(cid >= 0 for cid in result.values()), "All nodes should be in community"
    print("✓ Test 2 passed: Triangle community")
    
    # Test 3: Disconnected components
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Component 1
    g.add_edges_from([(3, 4), (4, 5), (5, 3)])  # Component 2
    result = detect_communities(g, min_k=2)
    communities = set(result.values())
    assert len(communities) == 2, "Should have 2 communities"
    print("✓ Test 3 passed: Disconnected components")
    
    # Test 4: Sparse graph (no k-core >= 2)
    sparse = nx.path_graph(5)
    result = detect_communities(sparse, min_k=2)
    assert all(cid == -1 for cid in result.values()), "Sparse graph should have no communities"
    print("✓ Test 4 passed: Sparse graph")
    
    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    _run_tests()
