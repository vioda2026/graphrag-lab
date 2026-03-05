from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str


@dataclass(slots=True)
class GraphNode:
    node_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    weight: float
    relation: str = "cooccurrence"


@dataclass(slots=True)
class GraphData:
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]


@dataclass(slots=True)
class Query:
    question: str


@dataclass(slots=True)
class RetrievalResult:
    node_ids: List[str]
    passages: List[str]
    scores: List[float]


@dataclass(slots=True)
class ReadResult:
    answer: str
    supporting_passages: List[str]
