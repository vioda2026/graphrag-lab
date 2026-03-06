# Retriever Training Module

This module provides training infrastructure for the GraphRAG retriever component.

## Components

### `trainer.py`
Core training loop with PyTorch DataLoader and checkpointing support.
- `RetrieverTrainer`: Main training class with train/validate/checkpoint methods
- `TrainingSample`: Data structure for training samples (query, positive, negatives)
- `RetrieverDataset`: PyTorch Dataset wrapper

### `mock_dataset.py`
Mock data generation for testing and development.
- `create_mock_training_samples()`: Generate synthetic training data
- `create_mock_validation_samples()`: Generate synthetic validation data

### `train.py`
Standalone training script for quick experimentation.

## Usage

### Training with Mock Data

```bash
# Basic training (2 epochs, batch size 4)
python -m graphrag_lab.retriever.train --epochs 2 --batch-size 4

# Custom configuration
python -m graphrag_lab.retriever.train \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --num-samples 100 \
  --checkpoint-dir artifacts/checkpoints/my_experiment
```

### Via CLI

```bash
# Single training run
python -m graphrag_lab.cli --command train-retriever --epochs 2

# Seed sweep
python -m graphrag_lab.cli \
  --command train-retriever \
  --epochs 2 \
  --seeds 7,11,13
```

### Programmatic Usage

```python
from graphrag_lab.configs.schema import RetrieverTrainingConfig
from graphrag_lab.retriever import RetrieverTrainer, create_mock_training_samples

# Create config
config = RetrieverTrainingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=4,
    learning_rate=2e-5,
    num_epochs=3,
    checkpoint_dir=Path("artifacts/checkpoints"),
    warmup_ratio=0.1,
    max_length=512,
    margin=0.3,
)

# Create mock data
train_samples = create_mock_training_samples(num_samples=100)
val_samples = create_mock_training_samples(num_samples=20)

# Initialize trainer and train
trainer = RetrieverTrainer(config)
result = trainer.train(train_samples, val_samples)
```

## Checkpoint Format

Checkpoints are saved as JSON files with the following structure:

```json
{
  "epoch": 1,
  "step": 100,
  "loss": 0.5,
  "config": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 4,
    "learning_rate": 2e-05
  }
}
```

## Experiment Tracking

All training runs are logged to `artifacts/experiment_runs.jsonl` with fields:
- `run_id`: Unique identifier
- `timestamp_utc`: ISO timestamp
- `type`: "retriever_training"
- `seed`: Random seed used
- `score`: Final validation score (lower is better)
- `checkpoint_path`: Path to saved checkpoint
- `train_loss`, `val_loss`: Training/validation losses
- `num_epochs`, `batch_size`, `learning_rate`: Hyperparameters

Seed sweeps generate summary files: `artifacts/seed_sweep_retriever_*.json` with:
- `aggregate`: Mean/std score, best seed
- `runs`: Individual run details

## Requirements

- PyTorch (required for actual training)
- Mock mode works without PyTorch for testing

## Tests

```bash
# Run retriever training tests
python -m unittest tests.test_retriever_trainer
python -m unittest tests.test_retriever_training_e2e
```
