"""Distributed launcher hooks for 4xA800 multi-GPU training."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedConfig:
    """Configuration for distributed training on 4xA800."""
    # World size and rank
    world_size: int = 4  # 4xA800
    rank: int = 0
    local_rank: int = 0
    
    # Communication backend
    backend: str = "nccl"  # NCCL for GPU, gloo for CPU
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 4
    
    # Mixed precision
    mixed_precision: str = "fp16"  # fp16, bf16, or fp32
    
    # Communication settings
    init_method: str = "env://"  # env://, tcp://, file://
    timeout_minutes: int = 30


def get_distributed_config_from_env() -> DistributedConfig:
    """
    Create DistributedConfig from environment variables.
    
    Standard torchrun environment variables:
    - WORLD_SIZE: total number of processes
    - RANK: global rank of this process
    - LOCAL_RANK: local rank on this node
    - MASTER_ADDR: address of the master node
    - MASTER_PORT: port of the master node
    """
    return DistributedConfig(
        world_size=int(os.getenv("WORLD_SIZE", "4")),
        rank=int(os.getenv("RANK", "0")),
        local_rank=int(os.getenv("LOCAL_RANK", "0")),
        backend=os.getenv("DIST_BACKEND", "nccl"),
        gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")),
        mixed_precision=os.getenv("MIXED_PRECISION", "fp16"),
        init_method=os.getenv("DIST_INIT_METHOD", "env://"),
        timeout_minutes=int(os.getenv("DIST_TIMEOUT_MINUTES", "30")),
    )


def is_main_process(config: Optional[DistributedConfig] = None) -> bool:
    """Check if this is the main process (rank 0)."""
    if config is None:
        config = get_distributed_config_from_env()
    return config.rank == 0


def get_effective_batch_size(per_device_batch_size: int, config: Optional[DistributedConfig] = None) -> int:
    """
    Calculate effective batch size across all GPUs with gradient accumulation.
    
    effective_batch_size = per_device_batch_size * world_size * gradient_accumulation_steps
    """
    if config is None:
        config = get_distributed_config_from_env()
    return per_device_batch_size * config.world_size * config.gradient_accumulation_steps


def setup_distributed_training(config: Optional[DistributedConfig] = None) -> None:
    """
    Initialize distributed training environment.
    
    This is a scaffold - actual torch.distributed.init_process_group
    will be called when PyTorch is available.
    
    Usage with torchrun:
        torchrun --nproc_per_node=4 --nnodes=1 \\
            --master_addr=localhost --master_port=29500 \\
            -m graphrag_lab.train_retriever
    """
    if config is None:
        config = get_distributed_config_from_env()
    
    # Placeholder for torch.distributed.init_process_group
    # This will be implemented when PyTorch is installed
    print(f"Distributed training scaffold initialized:")
    print(f"  - World size: {config.world_size}")
    print(f"  - Rank: {config.rank}")
    print(f"  - Local rank: {config.local_rank}")
    print(f"  - Backend: {config.backend}")
    print(f"  - Mixed precision: {config.mixed_precision}")
    print(f"  - Gradient accumulation steps: {config.gradient_accumulation_steps}")


def cleanup_distributed_training() -> None:
    """
    Cleanup distributed training environment.
    
    This is a scaffold - actual torch.distributed.destroy_process_group
    will be called when PyTorch is available.
    """
    # Placeholder for torch.distributed.destroy_process_group
    print("Distributed training cleanup called")


@dataclass
class A800HardwareProfile:
    """Hardware profile for 4xA800 setup."""
    # GPU specs
    gpu_memory_gb: int = 80  # A800 80GB
    num_gpus: int = 4
    
    # Recommended settings for A800
    recommended_batch_size_per_gpu: int = 32
    recommended_gradient_accumulation: int = 4
    recommended_mixed_precision: str = "bf16"  # A800 supports bf16 well
    
    @property
    def total_gpu_memory_gb(self) -> int:
        return self.gpu_memory_gb * self.num_gpus
    
    @property
    def effective_batch_size(self) -> int:
        return (
            self.recommended_batch_size_per_gpu *
            self.num_gpus *
            self.recommended_gradient_accumulation
        )
    
    def get_training_config(self) -> DistributedConfig:
        """Get recommended distributed config for 4xA800."""
        return DistributedConfig(
            world_size=self.num_gpus,
            backend="nccl",
            gradient_accumulation_steps=self.recommended_gradient_accumulation,
            mixed_precision=self.recommended_mixed_precision,
        )


# Global hardware profile (can be overridden)
DEFAULT_HARDWARE_PROFILE = A800HardwareProfile()


def get_recommended_config_for_hardware() -> DistributedConfig:
    """Get recommended distributed config for current hardware."""
    return DEFAULT_HARDWARE_PROFILE.get_training_config()
