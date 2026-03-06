from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

from graphrag_lab.benchmarks.base import BenchmarkAdapter, BenchmarkSample
from graphrag_lab.core.types import Document, Query


class GraphRAGBenchAdapter(BenchmarkAdapter):
    """JSONL adapter for GraphRAGBench-like records.

    Expected record schema (one JSON per line):
    {
      "sample_id": "id-1",
      "split": "train|val|test",
      "question": "...",
      "expected_answer": "...",
      "answer_aliases": ["..."],
      "documents": [{"id": "d1", "text": "..."}]
    }
    """

    def __init__(self, data_path: Path, split: str = "test") -> None:
        self.data_path = data_path
        self.split = split

    def samples(self) -> Iterable[BenchmarkSample]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("split", "test") != self.split:
                    continue
                docs = [Document(doc_id=d["id"], text=d["text"]) for d in row.get("documents", [])]
                yield BenchmarkSample(
                    sample_id=row["sample_id"],
                    query=Query(question=row["question"]),
                    documents=docs,
                    expected_answer=row["expected_answer"],
                    answer_aliases=row.get("answer_aliases", []),
                )

    def evaluate(self, expected: str, predicted: str, answer_aliases: List[str] | None = None) -> float:
        candidates = [expected, *(answer_aliases or [])]
        return max((_token_f1(candidate, predicted) for candidate in candidates), default=0.0)


def _normalize_tokens(text: str) -> List[str]:
    cleaned = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return [t for t in cleaned.split() if t]


def _token_f1(expected: str, predicted: str) -> float:
    e_raw = expected.strip().lower()
    p_raw = predicted.strip().lower()
    if e_raw and (e_raw == p_raw):
        return 1.0

    expected_tokens = _normalize_tokens(expected)
    predicted_tokens = _normalize_tokens(predicted)
    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0

    overlap = len(set(expected_tokens) & set(predicted_tokens))
    precision = overlap / max(1, len(set(predicted_tokens)))
    recall = overlap / max(1, len(set(expected_tokens)))
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
