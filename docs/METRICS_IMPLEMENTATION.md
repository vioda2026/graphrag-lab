# GraphRAGBench Metrics Implementation

## Overview

This directory contains the official GraphRAGBench evaluation metrics implementation, aligned with the [GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) repository.

## Implemented Metrics

### 1. `lexical_f1()` (Token-level F1)

**Status:** ✅ Fully implemented and tested

The primary metric for fact retrieval and complex reasoning tasks. Computes token-level F1 score between expected and predicted answers.

**Algorithm:**
1. Normalize both texts (lowercase, remove articles/punctuation)
2. Compute token sets
3. Calculate precision and recall from token overlap
4. Return F1 score (harmonic mean)

**Usage:**
```python
from graphrag_lab.benchmarks.metrics import lexical_f1

score = lexical_f1("Paris", "The capital is Paris")  # Returns 0.5
```

**Test coverage:** `tests/adapter/test_metrics_parity.py::TestLexicalF1`

### 2. `rouge_l()` (ROUGE-L)

**Status:** ✅ Implemented (requires `rouge_score` package)

Used for complex reasoning and fact retrieval tasks. Wraps the official `rouge_score` library with stemming enabled.

**Usage:**
```python
from graphrag_lab.benchmarks.metrics import rouge_l

score = rouge_l("The quick brown fox", "The quick fox")  # Returns ~0.75
```

**Dependencies:** `pip install rouge_score`

**Test coverage:** `tests/adapter/test_metrics_parity.py::TestRougeL`

### 3. `answer_correctness()` (Answer Correctness)

**Status:** ⚠️ Placeholder (requires LLM + embeddings)

Combines factuality scoring (TP/FP/FN classification) with semantic similarity. Used for all question types in the official benchmark.

**Algorithm:**
1. Generate atomic statements from answer and ground truth using LLM
2. Classify statements as TP/FP/FN using LLM
3. Compute F-beta score for factuality
4. Compute cosine similarity using embeddings
5. Return weighted average (default: 0.75 factuality + 0.25 similarity)

**Current fallback:** Returns `lexical_f1()` with warning

**Full implementation requires:**
- LangChain LLM (e.g., ChatOpenAI)
- LangChain Embeddings (e.g., HuggingFaceBgeEmbeddings)

**Reference:** `Evaluation/metrics/answer_accuracy.py` in official repo

### 4. `coverage_score()` (Coverage)

**Status:** ⚠️ Placeholder (requires LLM)

Measures what percentage of reference answer facts are covered in the response. Used for contextual summarization and creative generation.

**Algorithm:**
1. Extract factual statements from reference using LLM
2. Check which facts are covered in response using LLM
3. Return percentage of covered facts

**Current fallback:** Returns `np.nan` with warning

**Reference:** `Evaluation/metrics/coverage.py` in official repo

### 5. `faithfulness_score()` (Faithfulness)

**Status:** ⚠️ Placeholder (requires LLM)

Measures what percentage of answer statements are supported by retrieved context. Used for creative generation.

**Algorithm:**
1. Break down answer into atomic statements using LLM
2. Check which statements are supported by context using LLM
3. Return percentage of supported statements

**Current fallback:** Returns `np.nan` with warning

**Reference:** `Evaluation/metrics/faithfulness.py` in official repo

## Metric Configuration by Question Type

Following the official GraphRAGBench evaluation pipeline:

| Question Type | Metrics |
|--------------|---------|
| Fact Retrieval | `lexical_f1`, `rouge_l`, `answer_correctness` |
| Complex Reasoning | `lexical_f1`, `rouge_l`, `answer_correctness` |
| Contextual Summarization | `answer_correctness`, `coverage_score` |
| Creative Generation | `answer_correctness`, `coverage_score`, `faithfulness_score` |

## Backward Compatibility

The `graphragbench_adapter.py` uses `lexical_f1()` by default, maintaining backward compatibility with existing code. The `evaluate()` method signature remains unchanged.

## Testing

Run the parity tests:
```bash
cd /path/to/graphrag-lab
source /home/jiaxin/temp/bin/activate
python -m pytest tests/adapter/test_metrics_parity.py -v
```

Run all benchmark adapter tests:
```bash
python -m pytest tests/test_graphragbench_adapter.py -v
```

## Differences from Official Implementation

### Implemented Locally (No External Dependencies)
- ✅ `lexical_f1`: Our implementation matches the official tokenization and F1 computation exactly
- ✅ `_normalize_tokens`: Matches official preprocessing

### Requires External Libraries
- ✅ `rouge_l`: Wraps official `rouge_score` library (same as official)
- ⚠️ `answer_correctness`: Requires LangChain + LLM + embeddings (same as official)
- ⚠️ `coverage_score`: Requires LangChain + LLM (same as official)
- ⚠️ `faithfulness_score`: Requires LangChain + LLM (same as official)

### Key Design Decisions

1. **Token-based F1**: Uses set-based token comparison (not sequence-based), matching the official implementation's approach for lexical metrics.

2. **Article Removal**: Removes "a", "an", "the" during normalization, as per official GraphRAGBench preprocessing.

3. **Graceful Degradation**: LLM-dependent metrics return `np.nan` or fall back to simpler metrics with warnings, allowing the system to function without full dependencies.

4. **Type Hints**: Full type annotations for better IDE support and type checking.

## Future Work

1. **Full LLM Integration**: Implement complete `answer_correctness`, `coverage_score`, and `faithfulness_score` with configurable LLM backends.

2. **Batch Evaluation**: Add batch evaluation utilities for processing entire datasets efficiently.

3. **Caching**: Implement statement caching to reduce LLM API calls during evaluation.

4. **Additional Metrics**: Add `context_relevance` and `evidence_recall` for retrieval evaluation.

## References

- Official GraphRAG-Benchmark: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- Evaluation Code: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main/Evaluation
- Paper: https://arxiv.org/abs/2506.05690
