"""
Retriever Controller Module for GraphRAG.

This module implements confidence-based adaptive termination control
for retrieval processes with dynamic threshold adjustment and early stopping.

Key Logic: Dynamic threshold adjustment + early stopping mechanism
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TerminationReason(Enum):
    """Reasons for retrieval termination."""
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    LOW_CONFIDENCE = "low_confidence"
    EARLY_STOP = "early_stop"
    MANUAL = "manual"


@dataclass
class RetrievalResult:
    """Container for retrieval results with confidence scores."""
    node_id: Any
    score: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TerminationDecision:
    """Decision output from retriever controller."""
    should_terminate: bool
    reason: TerminationReason
    final_results: List[RetrievalResult]
    iterations_completed: int
    confidence_trend: List[float]


class RetrieverController:
    """
    Adaptive retrieval controller with confidence-based termination.

    This controller manages iterative retrieval processes by:
    1. Tracking confidence scores across iterations
    2. Dynamically adjusting acceptance thresholds
    3. Implementing early stopping when convergence detected
    4. Providing termination decisions with provenance
    """

    def __init__(
        self,
        initial_threshold: float = 0.7,
        min_threshold: float = 0.3,
        max_threshold: float = 0.95,
        convergence_window: int = 3,
        convergence_delta: float = 0.02,
        max_iterations: int = 10,
        early_stop_patience: int = 2
    ):
        """
        Initialize the retriever controller.

        Args:
            initial_threshold: Starting confidence threshold (default: 0.7)
            min_threshold: Minimum allowed threshold (default: 0.3)
            max_threshold: Maximum allowed threshold (default: 0.95)
            convergence_window: Window size for convergence detection (default: 3)
            convergence_delta: Delta for convergence check (default: 0.02)
            max_iterations: Maximum retrieval iterations (default: 10)
            early_stop_patience: Iterations without improvement before stop (default: 2)
        """
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.convergence_window = convergence_window
        self.convergence_delta = convergence_delta
        self.max_iterations = max_iterations
        self.early_stop_patience = early_stop_patience
        
        # Runtime state
        self.current_threshold = initial_threshold
        self.confidence_history: List[float] = []
        self.best_score: float = 0.0
        self.no_improvement_count: int = 0

    def reset(self):
        """Reset controller state for new retrieval session."""
        self.current_threshold = self.initial_threshold
        self.confidence_history = []
        self.best_score = 0.0
        self.no_improvement_count = 0

    def update_threshold(self, avg_confidence: float) -> float:
        """
        Dynamically adjust threshold based on retrieval confidence.

        Args:
            avg_confidence: Average confidence of current iteration results

        Returns:
            Updated threshold value
        """
        # Adaptive adjustment: lower threshold if confidence is consistently low
        if len(self.confidence_history) >= self.convergence_window:
            recent_avg = sum(self.confidence_history[-self.convergence_window:]) / self.convergence_window
            
            if recent_avg < self.current_threshold - 0.1:
                # Confidence too low, relax threshold
                self.current_threshold = max(
                    self.min_threshold,
                    self.current_threshold - 0.05
                )
            elif recent_avg > self.current_threshold + 0.1:
                # Confidence high, tighten threshold
                self.current_threshold = min(
                    self.max_threshold,
                    self.current_threshold + 0.05
                )
        
        return self.current_threshold

    def check_convergence(self) -> bool:
        """
        Check if retrieval has converged based on confidence trend.

        Returns:
            True if converged, False otherwise
        """
        if len(self.confidence_history) < self.convergence_window:
            return False
        
        recent = self.confidence_history[-self.convergence_window:]
        max_recent = max(recent)
        min_recent = min(recent)
        
        return (max_recent - min_recent) < self.convergence_delta

    def check_early_stop(self, current_score: float) -> bool:
        """
        Check if early stopping should be triggered.

        Args:
            current_score: Current iteration's best score

        Returns:
            True if should stop early, False otherwise
        """
        if current_score > self.best_score:
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        return self.no_improvement_count >= self.early_stop_patience

    def decide_termination(
        self,
        results: List[RetrievalResult],
        iteration: int
    ) -> TerminationDecision:
        """
        Make termination decision based on current results.

        Args:
            results: Current iteration retrieval results
            iteration: Current iteration number

        Returns:
            TerminationDecision with should_terminate flag and metadata
        """
        # Record confidence
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
            self.confidence_history.append(avg_confidence)
            self.update_threshold(avg_confidence)
            
            best_score = max(r.score for r in results)
            should_early_stop = self.check_early_stop(best_score)
        else:
            avg_confidence = 0.0
            should_early_stop = False
        
        # Check termination conditions
        should_terminate = False
        reason = TerminationReason.MANUAL
        
        if iteration >= self.max_iterations:
            should_terminate = True
            reason = TerminationReason.MAX_ITERATIONS
        elif self.check_convergence():
            should_terminate = True
            reason = TerminationReason.CONVERGED
        elif should_early_stop and iteration >= 3:
            should_terminate = True
            reason = TerminationReason.EARLY_STOP
        elif results and all(r.confidence < self.min_threshold for r in results):
            should_terminate = True
            reason = TerminationReason.LOW_CONFIDENCE
        
        # Filter results by current threshold
        final_results = [
            r for r in results 
            if r.confidence >= self.current_threshold
        ]
        
        return TerminationDecision(
            should_terminate=should_terminate,
            reason=reason,
            final_results=final_results,
            iterations_completed=iteration,
            confidence_trend=self.confidence_history.copy()
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics for monitoring."""
        return {
            "current_threshold": self.current_threshold,
            "iterations": len(self.confidence_history),
            "best_score": self.best_score,
            "avg_confidence": (
                sum(self.confidence_history) / len(self.confidence_history)
                if self.confidence_history else 0.0
            ),
            "convergence_detected": self.check_convergence()
        }


# ============================================================================
# Unit Test Scaffold
# ============================================================================

def _run_tests():
    """Run basic unit tests for retriever controller."""
    # Test 1: Initialization
    controller = RetrieverController()
    assert controller.current_threshold == 0.7, "Initial threshold should be 0.7"
    assert controller.max_iterations == 10, "Max iterations should be 10"
    print("✓ Test 1 passed: Initialization")
    
    # Test 2: Threshold update
    controller.reset()
    new_threshold = controller.update_threshold(0.5)
    assert new_threshold == 0.7, "Threshold should not change without history"
    print("✓ Test 2 passed: Threshold update")
    
    # Test 3: Termination decision - max iterations
    controller = RetrieverController(max_iterations=3)
    results = [RetrievalResult(node_id=1, score=0.8, confidence=0.75)]
    
    for i in range(3):
        decision = controller.decide_termination(results, iteration=i+1)
    
    assert decision.should_terminate, "Should terminate at max iterations"
    assert decision.reason == TerminationReason.MAX_ITERATIONS
    print("✓ Test 3 passed: Max iterations termination")
    
    # Test 4: Convergence detection
    controller = RetrieverController(convergence_window=3, convergence_delta=0.02)
    controller.confidence_history = [0.75, 0.76, 0.75, 0.76]
    assert controller.check_convergence(), "Should detect convergence"
    print("✓ Test 4 passed: Convergence detection")
    
    # Test 5: Early stop
    controller = RetrieverController(early_stop_patience=2)
    controller.best_score = 0.9
    controller.no_improvement_count = 2
    assert controller.check_early_stop(0.85), "Should trigger early stop"
    print("✓ Test 5 passed: Early stop")
    
    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    _run_tests()
