from __future__ import annotations

from typing import List, Tuple

from graphrag_lab.core.text import overlap_score
from graphrag_lab.core.types import GraphData, Query, RetrievalResult
from graphrag_lab.retriever.base import Retriever

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class BaselineLexicalRetriever(Retriever):
    """Simple overlap scorer; uses torch topk when available."""

    def retrieve(self, graph: GraphData, query: Query, candidate_ids: list[str]) -> RetrievalResult:
        scored: List[Tuple[str, float, str]] = []
        for node_id in candidate_ids:
            node = graph.nodes[node_id]
            score = overlap_score(query.question, node.text)
            scored.append((node_id, score, node.text))

        if torch is not None and scored:
            scores_t = torch.tensor([s[1] for s in scored], dtype=torch.float32)
            values, idx = torch.topk(scores_t, k=min(3, len(scored)))
            ranking = [scored[i] for i in idx.tolist()]
            scores = values.tolist()
        else:
            ranking = sorted(scored, key=lambda x: x[1], reverse=True)[:3]
            scores = [r[1] for r in ranking]

        return RetrievalResult(
            node_ids=[r[0] for r in ranking],
            passages=[r[2] for r in ranking],
            scores=scores,
        )
