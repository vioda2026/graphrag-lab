# M2 Stage 2 Experiment Plan

**Created**: 2026-03-08 09:45 GMT+8  
**Status**: 🔄 Planning  
**Target DDL**: 2026-03-15 (Stage 2 completion)

---

## Overview

Stage 2 focuses on **full experiment implementation and validation** of the 4 P0 hypotheses from Stage 1 prototypes.

### Stage 1 → Stage 2 Transition Checklist

| Condition | Target | Current | Status |
|-----------|--------|---------|--------|
| Paper survey | ≥10 篇 | 10+ 篇 | ✅ |
| Innovation hypotheses | ≥15 项 | 10 条 (4 P0) | ✅ |
| P0 prototypes | 4 modules | 4 complete (23 tests) | ✅ |
| API-LLM Phase A | 15 samples | 15/15 complete | ✅ |
| API-LLM Phase B | Design doc | ✅ Complete | ✅ |

**Decision**: All Stage 1 conditions met. Proceed to Stage 2.

---

## P0 Hypotheses & Experiments

### Exp #1: Core-based Community Detection (k-core vs Leiden)

**Hypothesis**: k-core decomposition provides faster community detection with comparable quality.

| Metric | Baseline (Leiden) | Target (k-core) | Expected Gain |
|--------|-------------------|-----------------|---------------|
| Indexing time | ~30s | ~10s | ↓66% |
| Community quality (NMI) | 1.0 | 0.85-0.95 | -5~-15% |
| Query latency | ~500ms | ~350ms | ↓30% |

**Implementation**:
- [ ] `src/graphrag_lab/community_detector.py` ✅ (prototype complete)
- [ ] Integration with graph builder pipeline
- [ ] Benchmark: GraphRAGBench HotpotQA subset (100 queries)
- [ ] Metrics: NMI, ARI, indexing time, query latency

**Files to create**:
- `tests/test_community_detector_e2e.py`
- `experiments/exp01_community_comparison/`

---

### Exp #2: SPRIG Personalized PageRank (Linear-time Node Ranking)

**Hypothesis**: SPRIG's seeded propagation achieves linear-time ranking with comparable recall.

| Metric | Baseline (PageRank) | Target (SPRIG) | Expected Gain |
|--------|---------------------|----------------|---------------|
| Ranking time | ~5s | ~0.5s | ↓90% |
| Recall@10 | 1.0 | 0.90-0.95 | -5~-10% |
| CPU usage | High | Low | ↓70% |

**Implementation**:
- [ ] `src/graphrag_lab/node_ranker.py` ✅ (prototype complete)
- [ ] Integration with retriever pipeline
- [ ] Benchmark: Multi-hop QA (200 queries)
- [ ] Metrics: Recall@K, NDCG, ranking time

**Files to create**:
- `tests/test_node_ranker_e2e.py`
- `experiments/exp02_ranking_comparison/`

---

### Exp #3: Confidence-based Adaptive Termination

**Hypothesis**: Dynamic threshold adjustment reduces exploration steps without sacrificing accuracy.

| Metric | Baseline (Fixed 3 rounds) | Target (Adaptive) | Expected Gain |
|--------|---------------------------|-------------------|---------------|
| Avg exploration rounds | 3.0 | 1.5-2.0 | ↓33-50% |
| LLM calls per query | ~15 | ~8-10 | ↓33-47% |
| Answer accuracy | 1.0 | 0.95-1.0 | -0~-5% |

**Implementation**:
- [ ] `src/graphrag_lab/retriever_controller.py` ✅ (prototype complete)
- [ ] Confidence threshold tuning (grid search: 0.5-0.9)
- [ ] Benchmark: GraphRAGBench full test set
- [ ] Metrics: Accuracy, LLM calls, latency

**Files to create**:
- `tests/test_retriever_controller_e2e.py`
- `experiments/exp03_adaptive_termination/`

---

### Exp #4: Evidence Provenance Graph (Verifiable Reasoning)

**Hypothesis**: Explicit provenance tracking improves answer verifiability without significant overhead.

| Metric | Baseline (No provenance) | Target (With provenance) | Expected Gain |
|--------|--------------------------|--------------------------|---------------|
| Verifiability score | 0.5 | 0.8-0.9 | ↑60-80% |
| Overhead | 0% | 5-10% | +5-10% |
| User trust (survey) | 3.0/5 | 4.0-4.5/5 | ↑33-50% |

**Implementation**:
- [ ] `src/graphrag_lab/evidence_provenance_graph.py` ✅ (prototype complete)
- [ ] Integration with answer generation
- [ ] Human evaluation: 50 samples, 3 annotators
- [ ] Metrics: Verifiability score, overhead, trust survey

**Files to create**:
- `tests/test_evidence_provenance_e2e.py`
- `experiments/exp04_provenance_evaluation/`
- `evals/human_survey_template.md`

---

## Experiment Infrastructure

### 4.1 Distributed Training Setup (4xA800)

**Configuration**:
- Framework: PyTorch + torchrun
- Mixed precision: BF16
- Effective batch size: 512
- Gradient accumulation: 4 steps

**Files to create**:
- `configs/distributed/4xa800.yaml`
- `scripts/run_distributed.sh`
- `src/graphrag_lab/distributed/training_loop.py`

### 4.2 Experiment Tracking

**Tools**:
- Run IDs: UUID-based
- Metrics: CSV + JSONL aggregation
- Visualization: Matplotlib/Seaborn

**Files to create**:
- `src/graphrag_lab/experiment_tracker.py`
- `scripts/aggregate_results.py`

### 4.3 GraphRAGBench Integration

**Datasets**:
- HotpotQA (multi-hop reasoning)
- 2WikiMultihopQA (explicit multi-hop)
- Musique (challenging multi-hop)

**Metrics**:
- Exact Match (EM)
- F1 (lexical)
- Precision/Recall
- NDCG (ranking quality)

---

## Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| Week 1 (03-08 to 03-15) | Exp #1-2 complete | Community + ranking benchmarks |
| Week 2 (03-15 to 03-22) | Exp #3-4 complete | Termination + provenance eval |
| Week 3 (03-22 to 03-29) | Ablation studies | All combinations tested |
| Week 4 (03-29 to 04-05) | Paper draft | Results writeup |

---

## Immediate Next Steps (03-08 to 03-10)

1. **Create experiment directory structure**
   ```bash
   mkdir -p experiments/{exp01_community,exp02_ranking,exp03_termination,exp04_provenance}
   ```

2. **Create e2e test files for all 4 P0 modules**
   - `tests/test_community_detector_e2e.py`
   - `tests/test_node_ranker_e2e.py`
   - `tests/test_retriever_controller_e2e.py`
   - `tests/test_evidence_provenance_e2e.py`

3. **Set up GraphRAGBench data loader**
   - Download HotpotQA subset
   - Create data preprocessing script

4. **Run baseline measurements**
   - Leiden community detection (Exp #1 baseline)
   - Standard PageRank (Exp #2 baseline)
   - Fixed 3-round exploration (Exp #3 baseline)
   - No provenance (Exp #4 baseline)

---

## Success Criteria

- [ ] All 4 experiments complete with statistical significance (p < 0.05)
- [ ] At least 3/4 hypotheses validated (expected gains achieved)
- [ ] Full reproducibility (code + data + configs public)
- [ ] Paper draft ready for submission (ICLR/ACL/EMNLP target)

---

## Notes

- **Priority**: Exp #1 and #3 have highest expected impact (latency + LLM cost reduction)
- **Risk**: Human evaluation for Exp #4 may take longer than expected
- **Dependency**: 4xA800 access confirmation needed for distributed training tests
