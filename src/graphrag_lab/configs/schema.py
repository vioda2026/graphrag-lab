from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RuntimeConfig:
    mode: str
    seed: int
    output_dir: Path


@dataclass(slots=True)
class BuilderConfig:
    min_edge_weight: float


@dataclass(slots=True)
class ExplorerConfig:
    top_k: int


@dataclass(slots=True)
class RetrieverConfig:
    top_k: int


@dataclass(slots=True)
class ReaderConfig:
    mode: str


@dataclass(slots=True)
class DataConfig:
    toy_data_path: Path
    graphragbench_data_path: Path


@dataclass(slots=True)
class BenchmarkConfig:
    name: str
    split: str


@dataclass(slots=True)
class AppConfig:
    runtime: RuntimeConfig
    builder: BuilderConfig
    explorer: ExplorerConfig
    retriever: RetrieverConfig
    reader: ReaderConfig
    data: DataConfig
    benchmark: BenchmarkConfig
