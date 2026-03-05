from __future__ import annotations

import argparse
import json

from graphrag_lab.configs.loader import load_config
from graphrag_lab.runners.pipeline import run_toy_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG Lab M1 runner")
    parser.add_argument(
        "--mode",
        default="local-debug",
        choices=["local-debug", "multi-gpu", "api-llm", "graphragbench-debug"],
        help="Run profile from configs/profiles",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.mode)
    report = run_toy_pipeline(cfg)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
