from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_lab.configs.loader import load_config
from graphrag_lab.configs.schema import RetrieverTrainingConfig
from graphrag_lab.runners.pipeline import run_seed_sweep, run_toy_pipeline
from graphrag_lab.runners.retriever_training import run_retriever_seed_sweep, run_retriever_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG Lab runner")
    parser.add_argument(
        "--mode",
        default="local-debug",
        choices=["local-debug", "multi-gpu", "api-llm", "graphragbench-debug"],
        help="Run profile from configs/profiles",
    )
    parser.add_argument(
        "--command",
        default="pipeline",
        choices=["pipeline", "train-retriever"],
        help="Command to run",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seed sweep, e.g. 7,11,13",
    )
    # Retriever training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (for train-retriever)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training (for train-retriever)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (for train-retriever)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Checkpoint directory (for train-retriever)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.mode)

    if args.command == "train-retriever":
        # Create retriever training config
        train_config = RetrieverTrainingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            checkpoint_dir=Path(args.checkpoint_dir),
            warmup_ratio=0.1,
            max_length=512,
            margin=0.3,
        )
        
        if args.seeds.strip():
            # Run seed sweep for retriever training
            seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
            report = run_retriever_seed_sweep(
                cfg,
                train_config,
                seeds,
                num_train_samples=50,
                num_val_samples=10,
            )
            print(json.dumps(report["aggregate"], indent=2))
        else:
            # Single training run
            result = run_retriever_training(
                cfg,
                train_config,
                num_train_samples=50,
                num_val_samples=10,
            )
            print(json.dumps(result["summary"], indent=2))
        return

    # Default: run pipeline
    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        report = run_seed_sweep(cfg, seeds)
        print(json.dumps(report["aggregate"], indent=2))
        return

    report = run_toy_pipeline(cfg)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
