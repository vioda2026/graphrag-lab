from __future__ import annotations

from itertools import combinations
from typing import Iterable

from graphrag_lab.core.text import overlap_score
from graphrag_lab.core.types import Document, GraphData, GraphEdge, GraphNode
from graphrag_lab.graph_builder.base import GraphBuilder


class BaselineCooccurrenceGraphBuilder(GraphBuilder):
    """Creates one node per document and weighted links by token overlap."""

    def __init__(self, min_edge_weight: float = 0.05) -> None:
        self.min_edge_weight = min_edge_weight

    def build(self, documents: Iterable[Document]) -> GraphData:
        docs = list(documents)
        nodes = {
            d.doc_id: GraphNode(node_id=d.doc_id, text=d.text, metadata={"source": "toy"})
            for d in docs
        }
        edges = []
        for a, b in combinations(docs, 2):
            weight = overlap_score(a.text, b.text)
            if weight >= self.min_edge_weight:
                edges.append(GraphEdge(source=a.doc_id, target=b.doc_id, weight=weight))
                edges.append(GraphEdge(source=b.doc_id, target=a.doc_id, weight=weight))
        return GraphData(nodes=nodes, edges=edges)
