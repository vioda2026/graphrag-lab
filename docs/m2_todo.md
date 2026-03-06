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
3. ✅ Distributed launcher hooks for 4xA800 (`torchrun`, gradient accumulation, mixed precision) (2026-03-06 13:33).
   - ✅ `DistributedConfig` schema (world_size, rank, backend, mixed_precision, etc.)
   - ✅ `get_distributed_config_from_env()` - env var parsing
   - ✅ `is_main_process()` - rank 0 detection
   - ✅ `get_effective_batch_size()` - effective BS calculation
   - ✅ `setup_distributed_training()` / `cleanup_distributed_training()` scaffolds
   - ✅ `A800HardwareProfile` - 4xA800 optimized settings (bf16, 512 effective BS)
   - ✅ Unit tests (`test_distributed_launcher.py`) - 16 tests passing
4. ✅ API-LLM reader client interface with retry, rate-limit, and response caching (2026-03-06 12:30).
   - ✅ `APILLMReaderConfig` schema
   - ✅ `ResponseCache` (file-based, 24h TTL)
   - ✅ `RateLimiter` (sliding window)
   - ✅ `APILLMReader` with retry logic (exponential backoff)
   - ✅ Unit tests (`test_api_llm_reader.py`) - 10 tests passing
   - ✅ Environment variable configuration support
5. 🔄 UnifiedMem-inspired memory module interface (2026-03-06 14:49 scaffold, 3 tests failing).
   - ✅ `MemoryRecord`, `MemoryQuery`, `MemoryResult` dataclasses
   - ✅ `InMemoryStorage` (write/read/update/delete)
   - ✅ `SimpleMemoryController` (eviction policy stubs)
   - ✅ `MemoryAwareRetriever` (hybrid scoring)
   - ✅ `MemoryManager` (high-level API)
   - ⚠️ 3 tests failing:
     - `test_update_access`: access_count double counting
     - `test_remember_triggers_eviction`: eviction not triggering
     - `test_save_and_load`: persistence not implemented
   - ⏳ Pending: bug fixes + persistence implementation
6. ✅ Expand experiment tracking with run IDs, seed sweep support, and CSV/JSONL aggregation.
7. ✅ Add regression tests for module contracts and benchmark adapter compatibility (57 tests total, 49 passing, 5 skipped pending PyTorch, 3 failing in unified_memory).
