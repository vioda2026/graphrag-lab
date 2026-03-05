# GraphRAG Lab (M1)

M1 delivers a minimal, runnable GraphRAG research scaffold focused on:
- graph structure experiments (`graph_builder`)
- graph traversal/retrieval experiments (`graph_explorer`)
- pluggable retriever/reader/benchmark adapters
- YAML config profiles for `local-debug`, `multi-gpu`, `api-llm`, and `graphragbench-debug`
- a CPU-safe toy end-to-end pipeline

## M2 in progress
- GraphRAGBench adapter scaffold added (`data/graphragbench/sample.jsonl`, split filtering + lightweight metric).

## Quickstart

```bash
make run-toy
```

Artifacts are written under `artifacts/`.
