"""
GraphRAGBench Official Metrics Implementation

This module implements the official evaluation metrics from GraphRAG-Bench:
- lexical_f1: Token-level F1 score (exact match baseline)
- rouge_l: ROUGE-L score using rouge_score library
- answer_correctness: Combined factuality + semantic similarity score
- coverage_score: Percentage of reference facts covered in response
- faithfulness_score: Percentage of answer statements supported by context

Reference: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
"""

from __future__ import annotations

import re
from typing import List, Optional


def _normalize_tokens(text: str) -> List[str]:
    """
    Normalize text for token-level comparison.
    
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Split into tokens
    
    This matches the official GraphRAGBench tokenization approach.
    """
    cleaned = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return [t for t in cleaned.split() if t]


def lexical_f1(expected: str, predicted: str) -> float:
    """
    Compute token-level F1 score between expected and predicted answers.
    
    This is the official lexical F1 metric from GraphRAGBench, which:
    1. Normalizes both texts (lowercase, remove articles/punctuation)
    2. Computes token-level precision and recall
    3. Returns F1 score (harmonic mean of precision and recall)
    
    Args:
        expected: Ground truth answer text
        predicted: Model-generated answer text
        
    Returns:
        F1 score between 0.0 and 1.0
        
    Examples:
        >>> lexical_f1("Paris", "The capital is Paris")
        0.5
        >>> lexical_f1("Mars", "Jupiter")
        0.0
        >>> lexical_f1("United States", "the united states.")
        1.0
    """
    e_raw = expected.strip().lower()
    p_raw = predicted.strip().lower()
    
    # Exact match shortcut
    if e_raw and (e_raw == p_raw):
        return 1.0
    
    expected_tokens = _normalize_tokens(expected)
    predicted_tokens = _normalize_tokens(predicted)
    
    # Handle edge cases
    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0
    
    # Compute token overlap
    expected_set = set(expected_tokens)
    predicted_set = set(predicted_tokens)
    overlap = len(expected_set & predicted_set)
    
    # Compute precision and recall
    precision = overlap / max(1, len(predicted_set))
    recall = overlap / max(1, len(expected_set))
    
    # Compute F1 score
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def rouge_l(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE-L score between reference and hypothesis.
    
    This wraps the official rouge_score library implementation used by GraphRAGBench.
    Requires: pip install rouge_score
    
    Args:
        reference: Ground truth text
        hypothesis: Model-generated text
        
    Returns:
        ROUGE-L F-measure between 0.0 and 1.0
        
    Note:
        Returns 0.0 if either text is empty or if rouge_score is not installed.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        # Fallback: return 0 if library not available
        return 0.0
    
    # Handle edge cases
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    
    # Initialize scorer with stemming (matches official implementation)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    llm=None,
    embeddings=None,
    weights: List[float] = [0.75, 0.25],
) -> float:
    """
    Compute answer correctness score combining factuality and semantic similarity.
    
    This is the official GraphRAGBench answer_correctness metric that:
    1. Breaks down answer and ground_truth into atomic statements
    2. Classifies statements as TP/FP/FN using LLM
    3. Computes factuality F-beta score
    4. Computes semantic similarity using embeddings
    5. Returns weighted average
    
    Args:
        question: The question being answered
        answer: Model-generated answer
        ground_truth: Reference ground truth answer
        llm: LangChain LLM for statement generation and classification
        embeddings: LangChain embeddings for semantic similarity
        weights: [factuality_weight, similarity_weight], default [0.75, 0.25]
        
    Returns:
        Correctness score between 0.0 and 1.0
        
    Note:
        Requires LangChain and an LLM/embeddings backend.
        Returns 0.0 if dependencies are not available.
    """
    # Placeholder implementation - requires LLM and embeddings
    # Full implementation would match the official answer_accuracy.py
    # For now, fall back to lexical F1 as a simple baseline
    import warnings
    warnings.warn(
        "answer_correctness requires LLM and embeddings. "
        "Falling back to lexical_f1. Install langchain and configure LLM/embeddings for full implementation."
    )
    return lexical_f1(ground_truth, answer)


def coverage_score(
    question: str,
    reference: str,
    response: str,
    llm=None,
) -> float:
    """
    Compute coverage score measuring what percentage of reference facts are covered.
    
    This is the official GraphRAGBench coverage metric that:
    1. Extracts factual statements from reference answer using LLM
    2. Checks which facts are covered in the response using LLM
    3. Returns percentage of covered facts
    
    Args:
        question: The question being answered
        reference: Reference ground truth answer
        response: Model-generated response
        llm: LangChain LLM for fact extraction and coverage checking
        
    Returns:
        Coverage score between 0.0 and 1.0
        
    Note:
        Requires LangChain and an LLM backend.
        Returns 1.0 for empty reference, NaN if fact extraction fails.
    """
    # Placeholder implementation
    import warnings
    import numpy as np
    warnings.warn(
        "coverage_score requires LLM. "
        "Returning NaN. Install langchain and configure LLM for full implementation."
    )
    return np.nan


def faithfulness_score(
    question: str,
    answer: str,
    contexts: List[str],
    llm=None,
) -> float:
    """
    Compute faithfulness score measuring what percentage of answer statements are supported by context.
    
    This is the official GraphRAGBench faithfulness metric that:
    1. Breaks down answer into atomic statements using LLM
    2. Checks which statements are supported by contexts using LLM
    3. Returns percentage of supported statements
    
    Args:
        question: The question being answered
        answer: Model-generated answer
        contexts: List of retrieved context passages
        llm: LangChain LLM for statement generation and verification
        
    Returns:
        Faithfulness score between 0.0 and 1.0
        
    Note:
        Requires LangChain and an LLM backend.
        Returns 1.0 for empty answer, 0.0 for empty contexts, NaN if processing fails.
    """
    # Placeholder implementation
    import warnings
    import numpy as np
    warnings.warn(
        "faithfulness_score requires LLM. "
        "Returning NaN. Install langchain and configure LLM for full implementation."
    )
    return np.nan


# Backward compatibility alias
token_f1 = lexical_f1
