from __future__ import annotations

import argparse
import json

from graphrag_lab.configs.loader import load_config
from graphrag_lab.runners.pipeline import run_seed_sweep, run_toy_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG Lab M1 runner")
    parser.add_argument(
        "--mode",
        default="local-debug",
        choices=["local-debug", "multi-gpu", "api-llm", "graphragbench-debug"],
        help="Run profile from configs/profiles",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seed sweep, e.g. 7,11,13",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.mode)

    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        report = run_seed_sweep(cfg, seeds)
        print(json.dumps(report["aggregate"], indent=2))
        return

    report = run_toy_pipeline(cfg)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
