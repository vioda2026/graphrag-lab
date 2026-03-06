# M2 TODO (GraphRAGBench + UnifiedMem-Inspired)

1. ✅ Integrate GraphRAGBench adapter with train/val/test split filtering (official metric parity pending; currently lexical-F1 + exact/contains fallback).
2. ✅ Retriever training loop scaffold with PyTorch dataloaders and checkpointing (2026-03-06 11:30, updated 2026-03-06 12:15).
   - ✅ Training config schema (`RetrieverTrainingConfig`)
   - ✅ `RetrieverDataset` (PyTorch Dataset)
   - ✅ `RetrieverTrainer` class with:
     - ✅ `train_step()` method (batch-level training)
     - ✅ `validate()` method (returns metrics dict)
     - ✅ `save_checkpoint()` / `load_checkpoint()`
   - ✅ Unit tests (`test_retriever_trainer.py`) - 6 tests (1 passing, 5 skipped pending PyTorch)
   - ⏳ Actual model integration (pending PyTorch install)
3. Add distributed launcher hooks for 4xA800 (`torchrun`, gradient accumulation, mixed precision).
4. Implement API-LLM reader client interface with retry, rate-limit, and response caching.
5. Add UnifiedMem-inspired memory module interface:
   - memory write/read primitives
   - controller policy stubs
   - memory-aware retrieval features
6. ✅ Expand experiment tracking with run IDs, seed sweep support, and CSV/JSONL aggregation.
7. ✅ Add regression tests for module contracts and benchmark adapter compatibility (12 tests total, 7 passing, 5 skipped pending PyTorch).
