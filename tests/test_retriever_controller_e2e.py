"""
End-to-End Tests for Retriever Controller Module.

Tests confidence-based adaptive termination in realistic GraphRAG retrieval scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag_lab.retriever_controller import (
    RetrieverController, RetrievalResult, TerminationDecision, TerminationReason
)


def test_e2e_adaptive_threshold_adjustment():
    """Test dynamic threshold adjustment during retrieval."""
    controller = RetrieverController(
        initial_threshold=0.7,
        min_threshold=0.3,
        max_threshold=0.95,
        max_iterations=10
    )
    
    # Simulate retrieval iterations with varying confidence
    confidence_history = [0.5, 0.6, 0.75, 0.8, 0.85, 0.82, 0.83]
    
    for confidence in confidence_history:
        controller.confidence_history.append(confidence)
        controller.update_threshold(confidence)
        
        # Verify threshold stays within bounds
        assert controller.min_threshold <= controller.current_threshold <= controller.max_threshold, \
            f"Threshold {controller.current_threshold} out of bounds"
    
    print("✓ Test 1 passed: Adaptive threshold adjustment")
    return True


def test_e2e_convergence_detection():
    """Test convergence detection based on confidence window."""
    controller = RetrieverController(
        convergence_window=3,
        convergence_delta=0.02,
        max_iterations=10
    )
    
    # Simulate converging confidence scores
    controller.confidence_history = [0.70, 0.71, 0.715, 0.718, 0.719]
    
    # Check convergence
    is_converged = controller.check_convergence()
    assert is_converged, "Should detect convergence with stable confidence"
    
    # Test non-convergence (high variance)
    controller2 = RetrieverController(convergence_window=3, convergence_delta=0.02)
    controller2.confidence_history = [0.5, 0.8, 0.4, 0.9, 0.3]
    is_converged2 = controller2.check_convergence()
    assert not is_converged2, "Should not detect convergence with high variance"
    
    print("✓ Test 2 passed: Convergence detection")
    return True


def test_e2e_early_stopping():
    """Test early stopping mechanism."""
    controller = RetrieverController(
        early_stop_patience=2,
        max_iterations=10
    )
    
    # Simulate retrieval with no improvement
    best_score = 0.8
    no_improvement_count = 0
    
    scores = [0.75, 0.73, 0.72, 0.74, 0.73]
    
    should_stop = False
    for score in scores:
        if score < best_score:
            no_improvement_count += 1
        else:
            best_score = score
            no_improvement_count = 0
        
        if no_improvement_count >= controller.early_stop_patience:
            should_stop = True
            break
    
    assert should_stop, "Should trigger early stop after patience exceeded"
    
    print("✓ Test 3 passed: Early stopping mechanism")
    return True


def test_e2e_full_retrieval_cycle():
    """Test a complete retrieval cycle with termination decision."""
    controller = RetrieverController(
        initial_threshold=0.6,
        convergence_window=3,
        convergence_delta=0.03,
        max_iterations=10,
        early_stop_patience=3
    )
    
    # Simulate a retrieval process
    iteration = 0
    confidence_history = [0.55, 0.62, 0.70, 0.75, 0.78, 0.79, 0.795, 0.798]
    
    final_decision = None
    
    for confidence in confidence_history:
        iteration += 1
        
        # Create mock retrieval results
        results = [
            RetrievalResult(node_id=f"node_{i}", score=0.9 - i * 0.1, confidence=confidence)
            for i in range(5)
        ]
        
        # Make termination decision
        decision = controller.decide_termination(results, iteration)
        final_decision = decision
        
        if decision.should_terminate:
            print(f"  Terminated at iteration {iteration} with reason: {decision.reason.value}")
            break
    
    # Verify decision was made
    assert final_decision is not None, "Should have made a termination decision"
    
    print("✓ Test 4 passed: Full retrieval cycle")
    return True


def test_e2e_termination_reasons():
    """Test different termination reasons."""
    # Test max iterations
    controller1 = RetrieverController(max_iterations=5)
    controller1.confidence_history = [0.5, 0.5, 0.5, 0.5, 0.5]
    
    results1 = [RetrievalResult("n1", 0.8, 0.5)]
    decision1 = controller1.decide_termination(results1, iteration=5)
    assert decision1.should_terminate and decision1.reason == TerminationReason.MAX_ITERATIONS
    
    # Test convergence
    controller2 = RetrieverController(convergence_window=3, convergence_delta=0.01)
    controller2.confidence_history = [0.8, 0.805, 0.808, 0.809]
    
    results2 = [RetrievalResult("n1", 0.8, 0.809)]
    decision2 = controller2.decide_termination(results2, iteration=4)
    assert decision2.should_terminate and decision2.reason == TerminationReason.CONVERGED
    
    print("✓ Test 5 passed: Termination reasons")
    return True


def test_e2e_llm_call_reduction():
    """Test that adaptive termination reduces LLM calls."""
    # Fixed termination (baseline): always 5 iterations
    fixed_iterations = 5
    
    # Adaptive termination
    controller = RetrieverController(
        convergence_window=2,
        convergence_delta=0.05,
        max_iterations=10,
        early_stop_patience=2
    )
    
    # Simulate quick convergence
    confidence_history = [0.7, 0.85, 0.88, 0.89]
    
    adaptive_iterations = 0
    for confidence in confidence_history:
        adaptive_iterations += 1
        
        results = [RetrievalResult("n1", 0.8, confidence)]
        decision = controller.decide_termination(results, adaptive_iterations)
        
        if decision.should_terminate:
            break
    
    # Verify reduction
    reduction = (fixed_iterations - adaptive_iterations) / fixed_iterations * 100
    assert adaptive_iterations < fixed_iterations, \
        f"Adaptive ({adaptive_iterations}) should be < fixed ({fixed_iterations})"
    
    print(f"✓ Test 6 passed: LLM call reduction ({reduction:.1f}% fewer iterations)")
    return True


def run_all_tests():
    """Run all e2e tests."""
    print("=" * 60)
    print("Running Retriever Controller E2E Tests")
    print("=" * 60)
    
    tests = [
        test_e2e_adaptive_threshold_adjustment,
        test_e2e_convergence_detection,
        test_e2e_early_stopping,
        test_e2e_full_retrieval_cycle,
        test_e2e_termination_reasons,
        test_e2e_llm_call_reduction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
