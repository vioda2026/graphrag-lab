from __future__ import annotations

from graphrag_lab.core.types import Query, ReadResult, RetrievalResult
from graphrag_lab.reader.base import Reader


class BaselineExtractiveReader(Reader):
    """Returns top evidence sentence as a minimal baseline answer."""

    def read(self, query: Query, retrieved: RetrievalResult) -> ReadResult:
        if not retrieved.passages:
            return ReadResult(answer="No evidence found.", supporting_passages=[])
        best = retrieved.passages[0]
        return ReadResult(answer=best, supporting_passages=retrieved.passages[:2])
