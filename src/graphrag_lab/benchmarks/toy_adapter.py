from __future__ import annotations

import yaml
from pathlib import Path
from typing import Iterable, List

from graphrag_lab.benchmarks.base import BenchmarkAdapter, BenchmarkSample
from graphrag_lab.core.types import Document, Query


class ToyBenchmarkAdapter(BenchmarkAdapter):
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def samples(self) -> Iterable[BenchmarkSample]:
        payload = yaml.safe_load(self.data_path.read_text(encoding="utf-8"))
        docs = [Document(doc_id=d["id"], text=d["text"]) for d in payload["documents"]]
        for q in payload["queries"]:
            yield BenchmarkSample(
                sample_id=q["id"],
                query=Query(question=q["question"]),
                documents=docs,
                expected_answer=q["expected_answer"],
            )

    def evaluate(self, expected: str, predicted: str, answer_aliases: List[str] | None = None) -> float:
        e = expected.strip().lower()
        p = predicted.strip().lower()
        if e == p or e in p or p in e:
            return 1.0
        for alias in answer_aliases or []:
            a = alias.strip().lower()
            if a and (a == p or a in p or p in a):
                return 1.0
        return 0.0
