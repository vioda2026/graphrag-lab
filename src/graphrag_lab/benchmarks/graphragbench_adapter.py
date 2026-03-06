from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from graphrag_lab.benchmarks.base import BenchmarkAdapter, BenchmarkSample
from graphrag_lab.benchmarks.metrics import lexical_f1
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
        """
        Evaluate predicted answer against expected answer using lexical F1.
        
        Args:
            expected: Ground truth answer
            predicted: Model-generated answer
            answer_aliases: Optional list of alternative correct answers
            
        Returns:
            Maximum lexical F1 score across all candidate answers
        """
        candidates = [expected, *(answer_aliases or [])]
        return max((lexical_f1(candidate, predicted) for candidate in candidates), default=0.0)
