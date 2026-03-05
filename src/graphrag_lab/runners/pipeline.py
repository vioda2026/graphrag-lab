from __future__ import annotations

import csv
import json
import random
import uuid
from datetime import datetime, timezone
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


def _append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_csv(path: Path, row: Dict[str, object], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": config.runtime.mode,
        "benchmark": config.benchmark.name,
        "split": config.benchmark.split,
        "seed": config.runtime.seed,
        "avg_score": sum(scores) / max(1, len(scores)),
        "num_samples": len(scores),
    }
    report = {"config": dump_config(config), "summary": summary, "samples": samples_out}

    write_json(run_dir / "report.json", report)

    tracker_row = {
        "run_id": summary["run_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "mode": summary["mode"],
        "benchmark": summary["benchmark"],
        "split": summary["split"],
        "seed": summary["seed"],
        "avg_score": summary["avg_score"],
        "num_samples": summary["num_samples"],
    }
    tracker_root = config.runtime.output_dir
    _append_jsonl(tracker_root / "experiment_runs.jsonl", tracker_row)
    _append_csv(
        tracker_root / "experiment_runs.csv",
        tracker_row,
        fieldnames=["run_id", "timestamp_utc", "mode", "benchmark", "split", "seed", "avg_score", "num_samples"],
    )
    return report
