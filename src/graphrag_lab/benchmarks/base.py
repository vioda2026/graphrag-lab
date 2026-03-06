from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List

from graphrag_lab.core.types import Document, Query


@dataclass(slots=True)
class BenchmarkSample:
    sample_id: str
    query: Query
    documents: List[Document]
    expected_answer: str
    answer_aliases: List[str] = field(default_factory=list)


class BenchmarkAdapter(ABC):
    """Provides benchmark samples and computes a lightweight score."""

    @abstractmethod
    def samples(self) -> Iterable[BenchmarkSample]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, expected: str, predicted: str, answer_aliases: List[str] | None = None) -> float:
        raise NotImplementedError
