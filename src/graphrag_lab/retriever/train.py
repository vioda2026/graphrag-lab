#!/usr/bin/env python3
"""
Minimal training entry point for retriever.

Usage:
    python -m graphrag_lab.retriever.train --epochs 2 --batch-size 4
    
This creates mock data, trains for specified epochs, and saves checkpoints.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from graphrag_lab.configs.schema import RetrieverTrainingConfig
from graphrag_lab.retriever.mock_dataset import create_mock_training_samples, create_mock_validation_samples
from graphrag_lab.retriever.trainer import RetrieverTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train retriever with mock data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of mock training samples",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=10,
        help="Number of mock validation samples",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for training logs",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting retriever training")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Training samples: {args.num_samples}")
    print(f"   Validation samples: {args.num_val_samples}")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    
    # Create config
    config = RetrieverTrainingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        warmup_ratio=0.1,
        max_length=512,
        margin=0.3,
    )
    
    # Create mock data
    print("\n📦 Creating mock dataset...")
    train_samples = create_mock_training_samples(args.num_samples)
    val_samples = create_mock_validation_samples(args.num_val_samples)
    print(f"   Created {len(train_samples)} training samples")
    print(f"   Created {len(val_samples)} validation samples")
    
    # Initialize trainer
    print("\n🏋️ Initializing trainer...")
    try:
        trainer = RetrieverTrainer(config)
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        print("   Install with: pip install torch")
        print("   Creating mock training run for testing...")
        
        # Create mock training log for testing without PyTorch
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-mock"
        training_log = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "mock",
            "config": {
                "model_name": config.model_name,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
            },
            "training_history": [],
            "message": "Mock training - PyTorch not installed",
        }
        
        log_path = output_dir / f"training_log_{run_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(training_log, f, indent=2)
        
        print(f"   Mock training log saved to: {log_path}")
        return
    
    # Resume from checkpoint if specified
    resume_from = Path(args.resume_from) if args.resume_from else None
    if resume_from:
        print(f"\n📂 Resuming from checkpoint: {resume_from}")
    
    # Train
    print("\n🎯 Starting training loop...")
    training_result = trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        resume_from=resume_from,
    )
    
    # Save training log
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{len(train_samples)}samples"
    training_log = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        **training_result,
    }
    
    log_path = output_dir / f"training_log_{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"   Final checkpoint: {training_result['final_checkpoint']}")
    print(f"   Training log: {log_path}")
    
    # Print summary
    if training_result["training_history"]:
        final_epoch = training_result["training_history"][-1]
        print(f"\n📊 Final metrics:")
        print(f"   Epoch: {final_epoch['epoch']}")
        print(f"   Train loss: {final_epoch['train_loss']:.4f}")
        if "val_loss" in final_epoch:
            print(f"   Val loss: {final_epoch['val_loss']:.4f}")


if __name__ == "__main__":
    main()
