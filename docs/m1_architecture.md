# M1 Architecture Note

## Goals
- Keep a strict module boundary for graph structure vs graph exploration research.
- Ensure the default path runs on local CPU with no GPU assumptions.
- Preserve PyTorch-first integration points (retriever scoring path can use `torch.topk` when installed).

## Module Layout
- `graph_builder`: converts raw documents into a graph (`GraphBuilder` + `BaselineCooccurrenceGraphBuilder`).
- `graph_explorer`: traverses graph to produce candidate nodes (`GraphExplorer` + `BaselineBFSGraphExplorer`).
- `retriever`: ranks passages from candidate nodes (`Retriever` + `BaselineLexicalRetriever`).
- `reader`: synthesizes final answer (`Reader` + `BaselineExtractiveReader`).
- `benchmarks`: adapter boundary for datasets and metrics (`BenchmarkAdapter` + `ToyBenchmarkAdapter`).
- `configs`: YAML profile + dataclass schema with profile merge.
- `runners`: pipeline orchestration and artifact writing.

## Runtime Modes
- `local-debug`: smallest defaults for quick local iteration.
- `multi-gpu`: profile stub for distributed training-oriented knobs.
- `api-llm`: profile stub for API-based reader behavior.

## Artifacts
Each run writes `artifacts/<mode>/report.json` with config snapshot, summary metric, and per-sample outputs.
