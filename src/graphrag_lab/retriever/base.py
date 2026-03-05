from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_lab.core.types import GraphData, Query, RetrievalResult


class Retriever(ABC):
    """Converts explored graph nodes into ranked evidence passages."""

    @abstractmethod
    def retrieve(self, graph: GraphData, query: Query, candidate_ids: list[str]) -> RetrievalResult:
        raise NotImplementedError
