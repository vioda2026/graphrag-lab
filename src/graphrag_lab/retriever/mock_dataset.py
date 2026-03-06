"""Mock dataset for retriever training - minimal runnable example."""
from __future__ import annotations

from typing import List

from graphrag_lab.retriever.trainer import TrainingSample


def create_mock_training_samples(num_samples: int = 100) -> List[TrainingSample]:
    """
    Create mock training samples for testing/training.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of TrainingSample objects
    """
    samples = []
    for i in range(num_samples):
        sample = TrainingSample(
            query=f"Mock query {i} about topic {i % 10}",
            positive_node_id=f"positive_node_{i}",
            positive_text=f"This is positive text for query {i}. It contains relevant information about topic {i % 10}.",
            negative_node_ids=[f"negative_node_{i}_1", f"negative_node_{i}_2"],
            negative_texts=[
                f"This is negative text 1 for query {i}. It is unrelated.",
                f"This is negative text 2 for query {i}. Also unrelated.",
            ],
        )
        samples.append(sample)
    return samples


def create_mock_validation_samples(num_samples: int = 20) -> List[TrainingSample]:
    """
    Create mock validation samples.
    
    Args:
        num_samples: Number of validation samples
        
    Returns:
        List of TrainingSample objects
    """
    return create_mock_training_samples(num_samples)
