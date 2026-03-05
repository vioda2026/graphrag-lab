from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from graphrag_lab.benchmarks.base import BenchmarkAdapter
from graphrag_lab.benchmarks.graphragbench_adapter import GraphRAGBenchAdapter
from graphrag_lab.benchmarks.toy_adapter import ToyBenchmarkAdapter
from graphrag_lab.configs.loader import dump_config
from graphrag_lab.configs.schema import AppConfig
from graphrag_lab.core.logging import write_json
from graphrag_lab.graph_builder.baseline import BaselineCooccurrenceGraphBuilder
from graphrag_lab.graph_explorer.baseline import BaselineBFSGraphExplorer
from graphrag_lab.reader.baseline import BaselineExtractiveReader
from graphrag_lab.retriever.baseline import BaselineLexicalRetriever


def _build_adapter(config: AppConfig) -> BenchmarkAdapter:
    if config.benchmark.name == "graphragbench":
        return GraphRAGBenchAdapter(
            data_path=config.data.graphragbench_data_path,
            split=config.benchmark.split,
        )
    return ToyBenchmarkAdapter(config.data.toy_data_path)


def run_toy_pipeline(config: AppConfig) -> Dict[str, object]:
    random.seed(config.runtime.seed)

    adapter = _build_adapter(config)
    builder = BaselineCooccurrenceGraphBuilder(min_edge_weight=config.builder.min_edge_weight)
    explorer = BaselineBFSGraphExplorer()
    retriever = BaselineLexicalRetriever()
    reader = BaselineExtractiveReader()

    run_dir = config.runtime.output_dir / config.runtime.mode
    run_dir.mkdir(parents=True, exist_ok=True)

    samples_out: List[dict] = []
    scores: List[float] = []
    for sample in adapter.samples():
        graph = builder.build(sample.documents)
        candidates = explorer.explore(graph, sample.query, top_k=config.explorer.top_k)
        retrieved = retriever.retrieve(graph, sample.query, candidates[: config.retriever.top_k])
        read_result = reader.read(sample.query, retrieved)
        score = adapter.evaluate(sample.expected_answer, read_result.answer)
        scores.append(score)
        samples_out.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.query.question,
                "expected_answer": sample.expected_answer,
                "predicted_answer": read_result.answer,
                "candidate_ids": candidates,
                "retrieved_node_ids": retrieved.node_ids,
                "scores": retrieved.scores,
                "metric": score,
            }
        )

    summary = {
        "mode": config.runtime.mode,
        "avg_score": sum(scores) / max(1, len(scores)),
        "num_samples": len(scores),
    }
    report = {"config": dump_config(config), "summary": summary, "samples": samples_out}

    write_json(run_dir / "report.json", report)
    return report
