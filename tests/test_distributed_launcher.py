"""Tests for distributed launcher hooks."""
import unittest
import os
from unittest.mock import patch

from graphrag_lab.distributed.launcher import (
    DistributedConfig,
    get_distributed_config_from_env,
    is_main_process,
    get_effective_batch_size,
    A800HardwareProfile,
    DEFAULT_HARDWARE_PROFILE,
    get_recommended_config_for_hardware,
)


class DistributedConfigTest(unittest.TestCase):
    """Test DistributedConfig dataclass."""
    
    def test_default_config(self):
        """Test default distributed config values."""
        config = DistributedConfig()
        
        self.assertEqual(config.world_size, 4)
        self.assertEqual(config.rank, 0)
        self.assertEqual(config.local_rank, 0)
        self.assertEqual(config.backend, "nccl")
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.mixed_precision, "fp16")
        self.assertEqual(config.init_method, "env://")
        self.assertEqual(config.timeout_minutes, 30)
    
    def test_custom_config(self):
        """Test custom distributed config values."""
        config = DistributedConfig(
            world_size=8,
            rank=3,
            local_rank=1,
            backend="gloo",
            gradient_accumulation_steps=8,
            mixed_precision="bf16",
        )
        
        self.assertEqual(config.world_size, 8)
        self.assertEqual(config.rank, 3)
        self.assertEqual(config.local_rank, 1)
        self.assertEqual(config.backend, "gloo")
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertEqual(config.mixed_precision, "bf16")


class GetDistributedConfigFromEnvTest(unittest.TestCase):
    """Test environment variable parsing."""
    
    @patch.dict(os.environ, {
        "WORLD_SIZE": "8",
        "RANK": "3",
        "LOCAL_RANK": "1",
        "DIST_BACKEND": "gloo",
        "GRADIENT_ACCUMULATION_STEPS": "8",
        "MIXED_PRECISION": "bf16",
    })
    def test_env_config_parsing(self):
        """Test config parsing from environment variables."""
        config = get_distributed_config_from_env()
        
        self.assertEqual(config.world_size, 8)
        self.assertEqual(config.rank, 3)
        self.assertEqual(config.local_rank, 1)
        self.assertEqual(config.backend, "gloo")
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertEqual(config.mixed_precision, "bf16")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_default_env_config(self):
        """Test default values when environment variables not set."""
        config = get_distributed_config_from_env()
        
        self.assertEqual(config.world_size, 4)
        self.assertEqual(config.rank, 0)
        self.assertEqual(config.local_rank, 0)


class IsMainProcessTest(unittest.TestCase):
    """Test main process detection."""
    
    def test_main_process_rank_0(self):
        """Test rank 0 is main process."""
        config = DistributedConfig(rank=0)
        self.assertTrue(is_main_process(config))
    
    def test_not_main_process_rank_1(self):
        """Test rank > 0 is not main process."""
        config = DistributedConfig(rank=1)
        self.assertFalse(is_main_process(config))
    
    def test_not_main_process_rank_3(self):
        """Test rank 3 is not main process."""
        config = DistributedConfig(rank=3)
        self.assertFalse(is_main_process(config))


class GetEffectiveBatchSizeTest(unittest.TestCase):
    """Test effective batch size calculation."""
    
    def test_effective_batch_size_4gpu(self):
        """Test effective batch size with 4 GPUs."""
        config = DistributedConfig(world_size=4, gradient_accumulation_steps=4)
        effective_bs = get_effective_batch_size(32, config)
        
        # 32 * 4 * 4 = 512
        self.assertEqual(effective_bs, 512)
    
    def test_effective_batch_size_8gpu(self):
        """Test effective batch size with 8 GPUs."""
        config = DistributedConfig(world_size=8, gradient_accumulation_steps=8)
        effective_bs = get_effective_batch_size(16, config)
        
        # 16 * 8 * 8 = 1024
        self.assertEqual(effective_bs, 1024)
    
    def test_effective_batch_size_default_config(self):
        """Test effective batch size with default config."""
        # Default: world_size=4, gradient_accumulation_steps=4
        effective_bs = get_effective_batch_size(32)
        
        # 32 * 4 * 4 = 512
        self.assertEqual(effective_bs, 512)


class A800HardwareProfileTest(unittest.TestCase):
    """Test A800 hardware profile."""
    
    def setUp(self):
        self.profile = A800HardwareProfile()
    
    def test_gpu_specs(self):
        """Test GPU specifications."""
        self.assertEqual(self.profile.gpu_memory_gb, 80)
        self.assertEqual(self.profile.num_gpus, 4)
    
    def test_total_gpu_memory(self):
        """Test total GPU memory calculation."""
        self.assertEqual(self.profile.total_gpu_memory_gb, 320)  # 80 * 4
    
    def test_recommended_settings(self):
        """Test recommended training settings."""
        self.assertEqual(self.profile.recommended_batch_size_per_gpu, 32)
        self.assertEqual(self.profile.recommended_gradient_accumulation, 4)
        self.assertEqual(self.profile.recommended_mixed_precision, "bf16")
    
    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        # 32 * 4 * 4 = 512
        self.assertEqual(self.profile.effective_batch_size, 512)
    
    def test_get_training_config(self):
        """Test getting distributed config from hardware profile."""
        config = self.profile.get_training_config()
        
        self.assertEqual(config.world_size, 4)
        self.assertEqual(config.backend, "nccl")
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.mixed_precision, "bf16")


class GetRecommendedConfigForHardwareTest(unittest.TestCase):
    """Test getting recommended config for current hardware."""
    
    def test_returns_a800_config(self):
        """Test returns A800-optimized config."""
        config = get_recommended_config_for_hardware()
        
        self.assertEqual(config.world_size, 4)
        self.assertEqual(config.backend, "nccl")
        self.assertEqual(config.mixed_precision, "bf16")


if __name__ == "__main__":
    unittest.main()
