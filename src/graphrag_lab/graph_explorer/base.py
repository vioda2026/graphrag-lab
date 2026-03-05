from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from graphrag_lab.core.types import GraphData, Query


class GraphExplorer(ABC):
    """Traverses the graph to produce candidate node ids for retrieval."""

    @abstractmethod
    def explore(self, graph: GraphData, query: Query, top_k: int = 3) -> List[str]:
        raise NotImplementedError
