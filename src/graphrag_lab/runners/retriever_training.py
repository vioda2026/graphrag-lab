"""Runner for retriever training with experiment tracking."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from graphrag_lab.configs.schema import AppConfig, RetrieverTrainingConfig
from graphrag_lab.retriever.mock_dataset import create_mock_training_samples, create_mock_validation_samples
from graphrag_lab.retriever.trainer import RetrieverTrainer


def _append_jsonl(path: Path, row: Dict[str, object]) -> None:
    """Append a row to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_retriever_training(
    config: AppConfig,
    train_config: RetrieverTrainingConfig,
    num_train_samples: int = 50,
    num_val_samples: int = 10,
    resume_from: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Run retriever training with experiment tracking.
    
    Args:
        config: Main app config
        train_config: Retriever training config
        num_train_samples: Number of training samples (mock)
        num_val_samples: Number of validation samples (mock)
        resume_from: Optional checkpoint path to resume from
        
    Returns:
        Training result dictionary
    """
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = config.runtime.output_dir / "retriever_training" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting retriever training run: {run_id}")
    
    # Create mock data
    train_samples = create_mock_training_samples(num_train_samples)
    val_samples = create_mock_validation_samples(num_val_samples)
    
    # Initialize trainer
    try:
        trainer = RetrieverTrainer(train_config)
    except ImportError as e:
        print(f"⚠️  PyTorch not available: {e}")
        print("   Creating mock training run for testing...")
        
        # Create mock experiment record
        experiment_record = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "type": "retriever_training",
            "seed": config.runtime.seed,
            "score": 0.0,
            "checkpoint_path": str(train_config.checkpoint_dir / "mock_checkpoint.pt"),
            "train_loss": None,
            "val_loss": None,
            "num_epochs": train_config.num_epochs,
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "status": "mock",
        }
        
        # Save to experiment_runs.jsonl
        tracker_path = config.runtime.output_dir / "experiment_runs.jsonl"
        _append_jsonl(tracker_path, experiment_record)
        
        # Save mock training log
        training_log = {
            "run_id": run_id,
            "status": "mock",
            "message": "Mock training - PyTorch not installed",
            "config": {
                "model_name": train_config.model_name,
                "batch_size": train_config.batch_size,
                "learning_rate": train_config.learning_rate,
                "num_epochs": train_config.num_epochs,
            },
        }
        
        log_path = run_dir / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(training_log, f, indent=2)
        
        print(f"   Mock training log saved to: {log_path}")
        
        return {
            "run_id": run_id,
            "summary": experiment_record,
            "training_result": training_log,
        }
    
    # Train
    training_result = trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        resume_from=resume_from,
    )
    
    # Get final metrics
    final_epoch = training_result["training_history"][-1] if training_result["training_history"] else {}
    final_score = final_epoch.get("val_loss", final_epoch.get("train_loss", 0.0))
    
    # Create experiment run record
    experiment_record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "type": "retriever_training",
        "seed": config.runtime.seed,
        "score": final_score,
        "checkpoint_path": training_result["final_checkpoint"],
        "train_loss": final_epoch.get("train_loss", None),
        "val_loss": final_epoch.get("val_loss", None),
        "num_epochs": train_config.num_epochs,
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
    }
    
    # Save to experiment_runs.jsonl
    tracker_path = config.runtime.output_dir / "experiment_runs.jsonl"
    _append_jsonl(tracker_path, experiment_record)
    
    # Save full training log
    training_log = {
        "run_id": run_id,
        "config": {
            "model_name": train_config.model_name,
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "num_epochs": train_config.num_epochs,
        },
        **training_result,
    }
    
    log_path = run_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    
    print(f"✅ Training completed: {run_id}")
    print(f"   Final score: {final_score:.4f}")
    print(f"   Checkpoint: {training_result['final_checkpoint']}")
    print(f"   Log: {log_path}")
    
    return {
        "run_id": run_id,
        "summary": experiment_record,
        "training_result": training_result,
    }


def run_retriever_seed_sweep(
    config: AppConfig,
    train_config: RetrieverTrainingConfig,
    seeds: List[int],
    num_train_samples: int = 50,
    num_val_samples: int = 10,
) -> Dict[str, object]:
    """
    Run retriever training with multiple seeds.
    
    Args:
        config: Main app config
        train_config: Retriever training config
        seeds: List of seeds to run
        num_train_samples: Number of training samples per run
        num_val_samples: Number of validation samples per run
        
    Returns:
        Sweep summary with aggregate metrics
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    
    print(f"🎯 Starting retriever seed sweep with {len(seeds)} seeds: {seeds}")
    
    runs: List[Dict[str, object]] = []
    scores: List[float] = []
    
    for seed in seeds:
        # Update config with seed
        from dataclasses import replace
        seeded_cfg = replace(config, runtime=replace(config.runtime, seed=seed))
        
        # Also update train_config checkpoint dir to include seed
        seeded_train_config = replace(
            train_config,
            checkpoint_dir=train_config.checkpoint_dir / f"seed_{seed}",
        )
        
        try:
            # Run training
            result = run_retriever_training(
                seeded_cfg,
                seeded_train_config,
                num_train_samples=num_train_samples,
                num_val_samples=num_val_samples,
            )
            
            runs.append(result["summary"])
            scores.append(result["summary"]["score"])
        except ImportError as e:
            print(f"⚠️  Seed {seed} failed (PyTorch not available): {e}")
            # Create mock record for seed sweep
            mock_record = {
                "run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-seed{seed}-mock",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "type": "retriever_training",
                "seed": seed,
                "score": 0.0,
                "checkpoint_path": str(seeded_train_config.checkpoint_dir / "mock_checkpoint.pt"),
                "status": "mock",
            }
            runs.append(mock_record)
            scores.append(0.0)
    
    # Calculate aggregate metrics
    import statistics
    aggregate = {
        "type": "retriever_seed_sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "num_runs": len(runs),
        "mean_score": statistics.fmean(scores),
        "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "best_seed": seeds[scores.index(min(scores))],  # Lower loss is better
        "best_score": min(scores),
    }
    
    # Save sweep summary
    sweep_summary = {
        "aggregate": aggregate,
        "runs": runs,
    }
    
    sweep_path = config.runtime.output_dir / f"seed_sweep_retriever_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2)
    
    print(f"\n✅ Seed sweep completed!")
    print(f"   Mean score: {aggregate['mean_score']:.4f} ± {aggregate['std_score']:.4f}")
    print(f"   Best seed: {aggregate['best_seed']} (score: {aggregate['best_score']:.4f})")
    print(f"   Summary: {sweep_path}")
    
    return sweep_summary
