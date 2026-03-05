from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from graphrag_lab.core.text import overlap_score
from graphrag_lab.core.types import GraphData, Query
from graphrag_lab.graph_explorer.base import GraphExplorer


class BaselineBFSGraphExplorer(GraphExplorer):
    """Seeds from lexical match and expands one hop using edge weights."""

    def explore(self, graph: GraphData, query: Query, top_k: int = 3) -> List[str]:
        if not graph.nodes:
            return []

        lexical = sorted(
            graph.nodes.items(),
            key=lambda item: overlap_score(query.question, item[1].text),
            reverse=True,
        )
        seed_ids = [node_id for node_id, _ in lexical[: max(1, min(2, len(lexical)))]]

        adjacency: Dict[str, List[tuple[str, float]]] = defaultdict(list)
        for edge in graph.edges:
            adjacency[edge.source].append((edge.target, edge.weight))

        selected: List[str] = []
        visited: Set[str] = set()
        for seed in seed_ids:
            if seed in visited:
                continue
            visited.add(seed)
            selected.append(seed)
            neighbors = sorted(adjacency.get(seed, []), key=lambda x: x[1], reverse=True)
            for neighbor, _ in neighbors:
                if neighbor not in visited:
                    selected.append(neighbor)
                    visited.add(neighbor)
                if len(selected) >= top_k:
                    return selected[:top_k]
            if len(selected) >= top_k:
                return selected[:top_k]

        return selected[:top_k]
