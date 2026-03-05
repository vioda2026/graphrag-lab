from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import yaml

from graphrag_lab.configs.schema import (
    AppConfig,
    BenchmarkConfig,
    BuilderConfig,
    DataConfig,
    ExplorerConfig,
    ReaderConfig,
    RetrieverConfig,
    RuntimeConfig,
)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _resolve_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(mode: str, config_root: Path | None = None) -> AppConfig:
    """YAML + dataclass config loader with OmegaConf-like profile merge."""
    cfg_root = (config_root or Path.cwd() / "configs").resolve()
    base = yaml.safe_load((cfg_root / "base.yaml").read_text(encoding="utf-8"))
    profile = yaml.safe_load((cfg_root / "profiles" / f"{mode}.yaml").read_text(encoding="utf-8"))
    merged = _deep_merge(base, profile)

    runtime = RuntimeConfig(
        mode=merged["runtime"]["mode"],
        seed=int(merged["runtime"]["seed"]),
        output_dir=_resolve_path(merged["runtime"]["output_dir"], Path.cwd()),
    )
    benchmark_cfg = merged.get("benchmark", {"name": "toy", "split": "test"})
    return AppConfig(
        runtime=runtime,
        builder=BuilderConfig(min_edge_weight=float(merged["builder"]["min_edge_weight"])),
        explorer=ExplorerConfig(top_k=int(merged["explorer"]["top_k"])),
        retriever=RetrieverConfig(top_k=int(merged["retriever"]["top_k"])),
        reader=ReaderConfig(mode=str(merged["reader"]["mode"])),
        data=DataConfig(
            toy_data_path=_resolve_path(merged["data"]["toy_data_path"], Path.cwd()),
            graphragbench_data_path=_resolve_path(
                merged["data"].get("graphragbench_data_path", "data/graphragbench/sample.jsonl"),
                Path.cwd(),
            ),
        ),
        benchmark=BenchmarkConfig(
            name=str(benchmark_cfg.get("name", "toy")),
            split=str(benchmark_cfg.get("split", "test")),
        ),
    )


def dump_config(config: AppConfig) -> Dict[str, Any]:
    data = asdict(config)
    data["runtime"]["output_dir"] = str(config.runtime.output_dir)
    data["data"]["toy_data_path"] = str(config.data.toy_data_path)
    data["data"]["graphragbench_data_path"] = str(config.data.graphragbench_data_path)
    return data
