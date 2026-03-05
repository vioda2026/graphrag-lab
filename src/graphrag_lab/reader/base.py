from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_lab.core.types import Query, ReadResult, RetrievalResult


class Reader(ABC):
    """Synthesizes final answer from retrieved evidence."""

    @abstractmethod
    def read(self, query: Query, retrieved: RetrievalResult) -> ReadResult:
        raise NotImplementedError
