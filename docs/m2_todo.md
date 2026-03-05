# M2 TODO (GraphRAGBench + UnifiedMem-Inspired)

1. ✅ Integrate GraphRAGBench adapter with train/val/test split filtering (official metric parity pending; currently lexical-F1 + exact/contains fallback).
2. Add retriever training loop with PyTorch dataloaders and checkpointing.
3. Add distributed launcher hooks for 4xA800 (`torchrun`, gradient accumulation, mixed precision).
4. Implement API-LLM reader client interface with retry, rate-limit, and response caching.
5. Add UnifiedMem-inspired memory module interface:
   - memory write/read primitives
   - controller policy stubs
   - memory-aware retrieval features
6. Expand experiment tracking with run IDs, seed sweep support, and CSV/JSONL aggregation.
7. Add regression tests for module contracts and benchmark adapter compatibility.
