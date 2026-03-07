"""
Evidence Provenance Graph Module for GraphRAG.

This module builds verifiable evidence provenance graphs that track
retrieval paths and reasoning steps with full traceability.

Key Features: Node provenance + edge weights + interpretability
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import hashlib
import json


class NodeType(Enum):
    """Types of nodes in the provenance graph."""
    QUERY = "query"
    EVIDENCE = "evidence"
    REASONING_STEP = "reasoning_step"
    CONCLUSION = "conclusion"
    SOURCE = "source"


class EdgeType(Enum):
    """Types of edges in the provenance graph."""
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFERENCES = "references"
    TRANSFORMS = "transforms"


@dataclass
class ProvenanceNode:
    """Node in the evidence provenance graph."""
    node_id: str
    node_type: NodeType
    content: Any
    source: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class ProvenanceEdge:
    """Edge in the evidence provenance graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    justification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "justification": self.justification,
            "metadata": self.metadata
        }


class EvidenceProvenanceGraph:
    """
    Build and manage evidence provenance graphs for verifiable reasoning.

    This class constructs directed graphs that track:
    1. Evidence sources and their relationships
    2. Reasoning steps and transformations
    3. Confidence propagation through the graph
    4. Full audit trail for conclusions
    """

    def __init__(self):
        """Initialize the provenance graph."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[ProvenanceEdge] = []
        self._node_counter = 0

    def _generate_node_id(self, prefix: str = "node") -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}_{hashlib.md5(str(self._node_counter).encode()).hexdigest()[:8]}"

    def add_query_node(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None
    ) -> str:
        """
        Add a query node as the root of the provenance graph.

        Args:
            query_text: The original query text
            query_embedding: Optional query embedding vector

        Returns:
            Node ID of the created query node
        """
        node_id = self._generate_node_id("query")
        node = ProvenanceNode(
            node_id=node_id,
            node_type=NodeType.QUERY,
            content=query_text,
            confidence=1.0,
            metadata={"embedding": query_embedding} if query_embedding else {}
        )
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        return node_id

    def add_evidence_node(
        self,
        content: Any,
        source: str,
        confidence: float = 1.0,
        retrieval_path: Optional[List[str]] = None
    ) -> str:
        """
        Add an evidence node from retrieval.

        Args:
            content: Evidence content (text, data, etc.)
            source: Source identifier (document ID, URL, etc.)
            confidence: Initial confidence score
            retrieval_path: Path taken to retrieve this evidence

        Returns:
            Node ID of the created evidence node
        """
        node_id = self._generate_node_id("evidence")
        node = ProvenanceNode(
            node_id=node_id,
            node_type=NodeType.EVIDENCE,
            content=content,
            source=source,
            confidence=confidence,
            metadata={"retrieval_path": retrieval_path} if retrieval_path else {}
        )
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        return node_id

    def add_reasoning_step(
        self,
        step_description: str,
        input_node_ids: List[str],
        operation: str,
        result: Any
    ) -> str:
        """
        Add a reasoning step node with connections to inputs.

        Args:
            step_description: Description of the reasoning step
            input_node_ids: IDs of nodes used as input
            operation: Type of operation (aggregate, transform, infer, etc.)
            result: Result of the reasoning step

        Returns:
            Node ID of the created reasoning node
        """
        node_id = self._generate_node_id("reasoning")
        node = ProvenanceNode(
            node_id=node_id,
            node_type=NodeType.REASONING_STEP,
            content=result,
            confidence=1.0,
            metadata={
                "description": step_description,
                "operation": operation,
                "inputs": input_node_ids
            }
        )
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        
        # Connect to input nodes
        for input_id in input_node_ids:
            self.add_edge(input_id, node_id, EdgeType.DERIVED_FROM)
        
        return node_id

    def add_conclusion_node(
        self,
        conclusion: Any,
        supporting_node_ids: List[str],
        confidence: float
    ) -> str:
        """
        Add a conclusion node with supporting evidence links.

        Args:
            conclusion: Final conclusion or answer
            supporting_node_ids: IDs of nodes supporting this conclusion
            confidence: Overall confidence in the conclusion

        Returns:
            Node ID of the created conclusion node
        """
        node_id = self._generate_node_id("conclusion")
        node = ProvenanceNode(
            node_id=node_id,
            node_type=NodeType.CONCLUSION,
            content=conclusion,
            confidence=confidence,
            metadata={"supporting_nodes": supporting_node_ids}
        )
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        
        # Connect to supporting nodes
        for support_id in supporting_node_ids:
            self.add_edge(support_id, node_id, EdgeType.SUPPORTS, 
                         weight=self.nodes[support_id].confidence)
        
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        justification: Optional[str] = None
    ) -> None:
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Edge weight (confidence/strength)
            justification: Optional explanation for the relationship
        """
        edge = ProvenanceEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            justification=justification
        )
        self.edges.append(edge)
        self.graph.add_edge(source_id, target_id, **edge.to_dict())

    def compute_confidence_propagation(self, node_id: str) -> float:
        """
        Compute propagated confidence for a node based on its ancestors.

        Args:
            node_id: Node to compute confidence for

        Returns:
            Propagated confidence score
        """
        if node_id not in self.nodes:
            return 0.0
        
        node = self.nodes[node_id]
        
        # Get all ancestors
        ancestors = nx.ancestors(self.graph, node_id)
        if not ancestors:
            return node.confidence
        
        # Compute weighted confidence from ancestors
        ancestor_confidences = []
        for ancestor_id in ancestors:
            if ancestor_id in self.nodes:
                ancestor_node = self.nodes[ancestor_id]
                # Get edge weight
                edge_data = self.graph.get_edge_data(ancestor_id, node_id)
                edge_weight = edge_data.get("weight", 1.0) if edge_data else 1.0
                ancestor_confidences.append(ancestor_node.confidence * edge_weight)
        
        if ancestor_confidences:
            # Use minimum confidence (weakest link)
            return min(ancestor_confidences)
        
        return node.confidence

    def get_provenance_chain(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get the full provenance chain for a node.

        Args:
            node_id: Node to trace

        Returns:
            List of nodes in the provenance chain (ordered from root to target)
        """
        if node_id not in self.nodes:
            return []
        
        # Get all ancestors
        ancestors = nx.ancestors(self.graph, node_id)
        ancestors.add(node_id)
        
        # Build chain in topological order
        subgraph = self.graph.subgraph(ancestors)
        ordered_nodes = list(nx.topological_sort(subgraph))
        
        chain = []
        for nid in ordered_nodes:
            if nid in self.nodes:
                chain.append(self.nodes[nid].to_dict())
        
        return chain

    def get_interpretation(self, node_id: str) -> Dict[str, Any]:
        """
        Generate human-readable interpretation of a node's provenance.

        Args:
            node_id: Node to interpret

        Returns:
            Dictionary with interpretation details
        """
        if node_id not in self.nodes:
            return {"error": "Node not found"}
        
        node = self.nodes[node_id]
        chain = self.get_provenance_chain(node_id)
        propagated_confidence = self.compute_confidence_propagation(node_id)
        
        return {
            "node_id": node_id,
            "node_type": node.node_type.value,
            "content_summary": str(node.content)[:200],
            "original_confidence": node.confidence,
            "propagated_confidence": propagated_confidence,
            "provenance_depth": len(chain),
            "sources": [
                n.get("source") for n in chain 
                if n.get("source")
            ],
            "reasoning_steps": [
                n.get("metadata", {}).get("description", "Unknown")
                for n in chain
                if n.get("node_type") == "reasoning_step"
            ]
        }

    def to_networkx(self) -> nx.DiGraph:
        """Return the underlying networkx graph."""
        return self.graph

    def export_to_dict(self) -> Dict[str, Any]:
        """Export the entire provenance graph as a dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "is_connected": nx.is_connected(self.graph.to_undirected()) 
                               if self.nodes else False
            }
        }

    def export_to_json(self, indent: int = 2) -> str:
        """Export the provenance graph as JSON string."""
        return json.dumps(self.export_to_dict(), indent=indent)


# ============================================================================
# Unit Test Scaffold
# ============================================================================

def _run_tests():
    """Run basic unit tests for evidence provenance graph."""
    # Test 1: Graph initialization
    graph = EvidenceProvenanceGraph()
    assert len(graph.nodes) == 0, "New graph should have no nodes"
    assert len(graph.edges) == 0, "New graph should have no edges"
    print("✓ Test 1 passed: Graph initialization")
    
    # Test 2: Add query node
    query_id = graph.add_query_node("What is GraphRAG?")
    assert query_id in graph.nodes, "Query node should be added"
    assert graph.nodes[query_id].node_type == NodeType.QUERY
    print("✓ Test 2 passed: Add query node")
    
    # Test 3: Add evidence node
    evidence_id = graph.add_evidence_node(
        content="GraphRAG is a graph-based retrieval system",
        source="doc_123",
        confidence=0.9
    )
    assert evidence_id in graph.nodes
    assert graph.nodes[evidence_id].source == "doc_123"
    print("✓ Test 3 passed: Add evidence node")
    
    # Test 4: Add edge
    graph.add_edge(query_id, evidence_id, EdgeType.REFERENCES, weight=0.8)
    assert len(graph.edges) == 1, "Should have one edge"
    assert graph.graph.has_edge(query_id, evidence_id)
    print("✓ Test 4 passed: Add edge")
    
    # Test 5: Add reasoning step
    reasoning_id = graph.add_reasoning_step(
        step_description="Extract key concepts",
        input_node_ids=[evidence_id],
        operation="extract",
        result=["graph", "retrieval", "RAG"]
    )
    assert reasoning_id in graph.nodes
    assert graph.nodes[reasoning_id].node_type == NodeType.REASONING_STEP
    print("✓ Test 5 passed: Add reasoning step")
    
    # Test 6: Add conclusion
    conclusion_id = graph.add_conclusion_node(
        conclusion="GraphRAG uses graph structures for retrieval",
        supporting_node_ids=[reasoning_id],
        confidence=0.85
    )
    assert conclusion_id in graph.nodes
    assert graph.nodes[conclusion_id].node_type == NodeType.CONCLUSION
    print("✓ Test 6 passed: Add conclusion")
    
    # Test 7: Provenance chain
    chain = graph.get_provenance_chain(conclusion_id)
    assert len(chain) >= 3, "Chain should include query, evidence, reasoning, conclusion"
    print("✓ Test 7 passed: Provenance chain")
    
    # Test 8: Confidence propagation
    propagated = graph.compute_confidence_propagation(conclusion_id)
    assert 0.0 <= propagated <= 1.0, "Confidence should be in [0, 1]"
    print("✓ Test 8 passed: Confidence propagation")
    
    # Test 9: Interpretation
    interpretation = graph.get_interpretation(conclusion_id)
    assert "node_type" in interpretation
    assert "provenance_depth" in interpretation
    print("✓ Test 9 passed: Interpretation")
    
    # Test 10: Export
    export_dict = graph.export_to_dict()
    assert "nodes" in export_dict
    assert "edges" in export_dict
    assert export_dict["metadata"]["node_count"] == len(graph.nodes)
    print("✓ Test 10 passed: Export")
    
    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    _run_tests()
