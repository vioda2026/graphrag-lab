# M2 TODO (GraphRAGBench + UnifiedMem-Inspired)

1. ✅ Integrate GraphRAGBench adapter with train/val/test split filtering (official metric parity pending; currently lexical-F1 + exact/contains fallback).
2. ✅ Retriever training loop scaffold with API-LLM verification path (2026-03-06 11:30, updated 2026-03-06 20:30 **API-LLM path**).
   - ✅ Training config schema (`RetrieverTrainingConfig`)
   - ✅ `RetrieverDataset` (PyTorch Dataset)
   - ✅ `RetrieverTrainer` class with:
     - ✅ `train_step()` method (batch-level training)
     - ✅ `validate()` method (returns metrics dict)
     - ✅ `save_checkpoint()` / `load_checkpoint()`
   - ✅ Unit tests (`test_retriever_trainer.py`) - 6 tests (1 passing, 5 skipped pending PyTorch)
   - ✅ **API-LLM verification path** (Phase 1 design, no code implementation):
     - Primary model: bailian/qwen3.5-plus (1M context, low cost)
     - Backup model: minimax-cn/MiniMax-M2.5 (200k context)
     - Phase A (this week): Prompt validation (10-20 query-document pairs)
     - Phase B (next week): API wrapper interface design
     - Phase C (week 3): Ablation study design
   - ⏳ PyTorch integration deferred to Phase 3 (per 20:15 strategy adjustment)
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
5. ✅ UnifiedMem-inspired memory module interface (2026-03-06 14:49 scaffold, 2026-03-06 17:54 bug fixes).
   - ✅ `MemoryRecord`, `MemoryQuery`, `MemoryResult` dataclasses
   - ✅ `InMemoryStorage` (write/read/update/delete)
   - ✅ `SimpleMemoryController` (eviction policy stubs)
   - ✅ `MemoryAwareRetriever` (hybrid scoring)
   - ✅ `MemoryManager` (high-level API)
   - ✅ Persistence (JSON serialization for save/load)
   - ✅ All 19 tests passing
6. ✅ Expand experiment tracking with run IDs, seed sweep support, and CSV/JSONL aggregation.
7. ✅ Add regression tests for module contracts and benchmark adapter compatibility (64 tests total, 55 passing, 9 skipped pending PyTorch).
8. ✅ **Stage 1 P0 Hypothesis Prototypes** (2026-03-08 00:57, commit 39c3c79):
   - ✅ `community_detector.py`: k-core decomposition for community detection
     - `detect_communities()`: k-core + connected components
     - `get_core_communities()`: multi-level k-core analysis
     - 4 unit tests passing
   - ✅ `node_ranker.py`: SPRIG personalized PageRank for node ranking
     - `sprig_rank()`: query-aware personalization
     - `rank_with_community_boost()`: community-enhanced ranking
     - `get_top_k_nodes()`: top-k extraction utility
     - 4 unit tests passing
   - ✅ `retriever_controller.py`: confidence-based adaptive termination
     - `RetrieverController`: dynamic threshold adjustment
     - `check_convergence()`: window-based convergence detection
     - `check_early_stop()`: patience-based early stopping
     - 5 unit tests passing
   - ✅ `evidence_provenance_graph.py`: verifiable evidence tracking
     - `ProvenanceNode`/`ProvenanceEdge`: typed graph elements
     - `EvidenceProvenanceGraph`: full reasoning audit trail
     - `compute_confidence_propagation()`: weakest-link confidence
     - `get_interpretation()`: human-readable explanations
     - 10 unit tests passing
   - **Total**: 23 new unit tests, ~1105 lines (430 production + 675 tests)
   - **Dependencies installed**: networkx 2.4, numpy 1.21.5 (via apt)
