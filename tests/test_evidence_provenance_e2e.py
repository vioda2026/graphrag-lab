"""
End-to-End Tests for Evidence Provenance Graph Module.

Tests verifiable evidence tracking and reasoning audit trails in GraphRAG.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag_lab.evidence_provenance_graph import (
    EvidenceProvenanceGraph, ProvenanceNode, ProvenanceEdge,
    NodeType, EdgeType
)


def test_e2e_full_reasoning_chain():
    """Test building a complete reasoning chain from query to conclusion."""
    graph = EvidenceProvenanceGraph()
    
    # Build a reasoning chain
    query_id = graph.add_query_node("What is the relationship between ML and DL?")
    
    # Add evidence nodes
    evidence1_id = graph.add_evidence_node(
        "ML is a superset of AI techniques",
        source="textbook_ai_101",
        confidence=0.95
    )
    evidence2_id = graph.add_evidence_node(
        "DL is a subset of ML using neural networks",
        source="deep_learning_book",
        confidence=0.98
    )
    
    # Connect evidence to query
    graph.add_edge(query_id, evidence1_id, EdgeType.REFERENCES, weight=0.9)
    graph.add_edge(query_id, evidence2_id, EdgeType.REFERENCES, weight=0.95)
    
    # Add reasoning step
    reasoning_id = graph.add_reasoning_step(
        step_description="Since DL uses neural networks which are ML techniques, DL is a subset of ML",
        input_node_ids=[evidence1_id, evidence2_id],
        operation="infer",
        result="DL is a subset of ML"
    )
    
    # Add conclusion
    conclusion_id = graph.add_conclusion_node(
        conclusion="DL (Deep Learning) is a specialized subset of ML (Machine Learning)",
        supporting_node_ids=[reasoning_id],
        confidence=0.90
    )
    
    # Verify graph structure
    assert graph.graph.number_of_nodes() == 5, f"Expected 5 nodes, got {graph.graph.number_of_nodes()}"
    assert graph.graph.number_of_edges() >= 4, f"Expected >=4 edges, got {graph.graph.number_of_edges()}"
    
    # Verify conclusion has highest-level node
    assert graph.graph.out_degree(conclusion_id) == 0, "Conclusion should be a sink node"
    
    print("✓ Test 1 passed: Full reasoning chain")
    return True


def test_e2e_confidence_propagation():
    """Test confidence propagation through the provenance graph."""
    graph = EvidenceProvenanceGraph()
    
    # Build a chain with known confidences
    query_id = graph.add_query_node("Query")
    
    evidence1_id = graph.add_evidence_node("Evidence 1", source="src1", confidence=0.9)
    evidence2_id = graph.add_evidence_node("Evidence 2", source="src2", confidence=0.8)
    
    graph.add_edge(query_id, evidence1_id, EdgeType.REFERENCES)
    graph.add_edge(query_id, evidence2_id, EdgeType.REFERENCES)
    
    reasoning_id = graph.add_reasoning_step(
        step_description="Reasoning",
        input_node_ids=[evidence1_id, evidence2_id],
        operation="aggregate",
        result="Combined evidence"
    )
    
    conclusion_id = graph.add_conclusion_node(
        conclusion="Conclusion",
        supporting_node_ids=[reasoning_id],
        confidence=0.88
    )
    
    # Compute confidence propagation
    propagated_conf = graph.compute_confidence_propagation(conclusion_id)
    
    # Verify confidence is computed
    assert propagated_conf is not None, "Should compute propagated confidence"
    assert 0 < propagated_conf <= 1.0, f"Confidence {propagated_conf} should be in (0, 1]"
    
    print(f"✓ Test 2 passed: Confidence propagation (propagated: {propagated_conf:.3f})")
    return True


def test_e2e_interpretation_generation():
    """Test human-readable interpretation generation."""
    graph = EvidenceProvenanceGraph()
    
    # Build a simple reasoning graph
    query_id = graph.add_query_node("Is Python good for ML?")
    
    evidence1_id = graph.add_evidence_node(
        "Python has extensive ML libraries (TensorFlow, PyTorch)",
        source="ml_survey_2024",
        confidence=0.95
    )
    
    evidence2_id = graph.add_evidence_node(
        "Python has simple syntax and rapid prototyping",
        source="programming_best_practices",
        confidence=0.90
    )
    
    reasoning_id = graph.add_reasoning_step(
        step_description="Good library support + easy syntax = good for ML",
        input_node_ids=[evidence1_id, evidence2_id],
        operation="infer",
        result="Python is good for ML"
    )
    
    conclusion_id = graph.add_conclusion_node(
        conclusion="Yes, Python is well-suited for machine learning",
        supporting_node_ids=[reasoning_id],
        confidence=0.91
    )
    
    # Generate interpretation (returns dict)
    interpretation = graph.get_interpretation(conclusion_id)
    
    # Verify interpretation is generated
    assert interpretation is not None, "Should generate interpretation"
    assert isinstance(interpretation, dict), "Interpretation should be a dict"
    assert "node_id" in interpretation, "Should include node_id"
    assert "node_type" in interpretation, "Should include node_type"
    assert "provenance_depth" in interpretation, "Should include provenance_depth"
    assert interpretation["provenance_depth"] >= 3, "Should have depth >= 3"
    
    print(f"✓ Test 3 passed: Interpretation generation (depth={interpretation['provenance_depth']})")
    return True


def test_e2e_provenance_chain_extraction():
    """Test extracting the full provenance chain for a conclusion."""
    graph = EvidenceProvenanceGraph()
    
    # Build a multi-hop reasoning graph
    query_id = graph.add_query_node("Query")
    
    evidence1_id = graph.add_evidence_node("Evidence 1", source="src1", confidence=0.9)
    evidence2_id = graph.add_evidence_node("Evidence 2", source="src2", confidence=0.85)
    evidence3_id = graph.add_evidence_node("Evidence 3", source="src3", confidence=0.88)
    
    graph.add_edge(query_id, evidence1_id, EdgeType.REFERENCES)
    graph.add_edge(query_id, evidence2_id, EdgeType.REFERENCES)
    graph.add_edge(query_id, evidence3_id, EdgeType.REFERENCES)
    
    reasoning1_id = graph.add_reasoning_step(
        step_description="Intermediate reasoning 1",
        input_node_ids=[evidence1_id, evidence2_id],
        operation="combine",
        result="Intermediate 1"
    )
    
    reasoning2_id = graph.add_reasoning_step(
        step_description="Intermediate reasoning 2",
        input_node_ids=[reasoning1_id, evidence3_id],
        operation="infer",
        result="Intermediate 2"
    )
    
    conclusion_id = graph.add_conclusion_node(
        conclusion="Final conclusion",
        supporting_node_ids=[reasoning2_id],
        confidence=0.85
    )
    
    # Extract provenance chain (returns topological order: root to target)
    chain = graph.get_provenance_chain(conclusion_id)
    
    # Verify chain includes all ancestors
    assert len(chain) >= 6, f"Chain should include all ancestors, got {len(chain)}"
    
    # Verify chain is ordered (from root to target, so conclusion is last)
    assert chain[-1]['node_id'] == conclusion_id, "Chain should end with conclusion"
    
    # Verify all evidence nodes are in chain
    chain_ids = [node['node_id'] for node in chain]
    assert evidence1_id in chain_ids, "Evidence 1 should be in chain"
    assert evidence2_id in chain_ids, "Evidence 2 should be in chain"
    assert evidence3_id in chain_ids, "Evidence 3 should be in chain"
    
    print("✓ Test 4 passed: Provenance chain extraction")
    return True


def test_e2e_export_and_serialization():
    """Test graph export and serialization."""
    graph = EvidenceProvenanceGraph()
    
    # Build a graph
    query_id = graph.add_query_node("Test query")
    evidence_id = graph.add_evidence_node("Test evidence", source="test_source", confidence=0.9)
    graph.add_edge(query_id, evidence_id, EdgeType.REFERENCES)
    
    # Export to dict
    exported = graph.export_to_dict()
    
    # Verify export structure
    assert 'nodes' in exported, "Export should include nodes"
    assert 'edges' in exported, "Export should include edges"
    assert len(exported['nodes']) == 2, f"Expected 2 nodes in export, got {len(exported['nodes'])}"
    assert len(exported['edges']) == 1, f"Expected 1 edge in export, got {len(exported['edges'])}"
    
    # Verify JSON serialization
    import json
    json_str = json.dumps(exported, default=str)
    assert len(json_str) > 50, "JSON export should be substantial"
    
    # Verify deserialization
    reloaded = json.loads(json_str)
    assert len(reloaded['nodes']) == 2, "Reloaded graph should have same nodes"
    
    print("✓ Test 5 passed: Export and serialization")
    return True


def run_all_tests():
    """Run all e2e tests."""
    print("=" * 60)
    print("Running Evidence Provenance Graph E2E Tests")
    print("=" * 60)
    
    tests = [
        test_e2e_full_reasoning_chain,
        test_e2e_confidence_propagation,
        test_e2e_interpretation_generation,
        test_e2e_provenance_chain_extraction,
        test_e2e_export_and_serialization,
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
