from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from graphrag_lab.core.types import Document, GraphData


class GraphBuilder(ABC):
    """Builds a queryable graph structure from raw documents."""

    @abstractmethod
    def build(self, documents: Iterable[Document]) -> GraphData:
        raise NotImplementedError
