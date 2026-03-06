"""End-to-end tests for retriever training, checkpoint, and seed sweep."""
import unittest
import tempfile
import shutil
import json
from pathlib import Path

from graphrag_lab.configs.schema import RetrieverTrainingConfig, RuntimeConfig, AppConfig
from graphrag_lab.retriever.trainer import RetrieverTrainer, TrainingSample
from graphrag_lab.retriever.mock_dataset import create_mock_training_samples
from graphrag_lab.runners.retriever_training import run_retriever_training, run_retriever_seed_sweep


class RetrieverTrainingE2ETest(unittest.TestCase):
    """End-to-end tests for retriever training workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = RetrieverTrainingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=4,
            learning_rate=2e-5,
            num_epochs=2,
            checkpoint_dir=self.checkpoint_dir,
            warmup_ratio=0.1,
            max_length=512,
            margin=0.3,
        )
        
        self.app_config = AppConfig(
            runtime=RuntimeConfig(
                mode="test",
                seed=42,
                output_dir=self.output_dir,
            ),
            builder=None,
            explorer=None,
            retriever=None,
            reader=None,
            data=None,
            benchmark=None,
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_training_mock_data_creation(self):
        """Test that mock training data can be created."""
        samples = create_mock_training_samples(num_samples=10)
        
        self.assertEqual(len(samples), 10)
        self.assertIsInstance(samples[0], TrainingSample)
        self.assertIn("Mock query", samples[0].query)
    
    def test_checkpoint_save_and_load_cycle(self):
        """Test checkpoint can be saved and loaded successfully."""
        try:
            trainer = RetrieverTrainer(self.config)
            
            # Save checkpoint
            checkpoint = trainer.save_checkpoint(epoch=1, step=50, loss=0.75)
            
            # Verify checkpoint file exists
            self.assertTrue(checkpoint.checkpoint_path.parent.exists())
            
            # Load checkpoint
            loaded = trainer.load_checkpoint(checkpoint.checkpoint_path)
            
            # Verify loaded data matches
            self.assertEqual(loaded.epoch, 1)
            self.assertEqual(loaded.step, 50)
            self.assertAlmostEqual(loaded.loss, 0.75)
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_training_one_step(self):
        """Test that training can run for at least one step."""
        try:
            trainer = RetrieverTrainer(self.config)
            
            # Create small batch
            batch = [
                TrainingSample(
                    query="Test query",
                    positive_node_id="node_1",
                    positive_text="Positive text",
                    negative_node_ids=["node_2"],
                    negative_texts=["Negative text"],
                )
            ]
            
            # Mock model
            class MockModel:
                def train(self): pass
                def eval(self): pass
            
            trainer.model = MockModel()
            
            # Run one training step
            loss = trainer.train_step(batch)
            
            self.assertIsInstance(loss, float)
            self.assertGreaterEqual(loss, 0.0)
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_training_two_epochs_mock(self):
        """Test training loop can run for 2 epochs with mock data."""
        try:
            trainer = RetrieverTrainer(self.config)
            
            # Create mock samples
            train_samples = create_mock_training_samples(num_samples=8)
            
            # Create dataloader
            dataloader = trainer.create_dataloader(train_samples, shuffle=True)
            
            # Train for 2 epochs
            for epoch in range(2):
                avg_loss, batch_logs = trainer.train_epoch(dataloader, epoch)
                
                # Verify epoch completed
                self.assertIsInstance(avg_loss, float)
                self.assertGreater(len(batch_logs), 0)
                
                # Save checkpoint
                checkpoint = trainer.save_checkpoint(epoch, len(batch_logs), avg_loss)
                self.assertTrue(checkpoint.checkpoint_path.parent.exists())
            
            # Verify checkpoints were created
            checkpoint_files = list(self.checkpoint_dir.glob("*.json"))
            self.assertGreater(len(checkpoint_files), 0)
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_resume_from_checkpoint(self):
        """Test training can resume from a checkpoint."""
        try:
            trainer = RetrieverTrainer(self.config)
            
            # Create and save initial checkpoint
            initial_checkpoint = trainer.save_checkpoint(epoch=0, step=10, loss=1.0)
            
            # Load checkpoint
            loaded = trainer.load_checkpoint(initial_checkpoint.checkpoint_path)
            
            self.assertEqual(loaded.epoch, 0)
            self.assertEqual(loaded.step, 10)
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_retriever_training_runner(self):
        """Test the retriever training runner with experiment tracking."""
        try:
            from graphrag_lab.configs.schema import BuilderConfig, ExplorerConfig, RetrieverConfig, ReaderConfig, DataConfig, BenchmarkConfig
            
            # Create minimal app config
            app_config = AppConfig(
                runtime=RuntimeConfig(
                    mode="test",
                    seed=42,
                    output_dir=self.output_dir,
                ),
                builder=BuilderConfig(min_edge_weight=0.5),
                explorer=ExplorerConfig(top_k=3),
                retriever=RetrieverConfig(top_k=3),
                reader=ReaderConfig(mode="extractive"),
                data=DataConfig(toy_data_path=self.temp_dir, graphragbench_data_path=self.temp_dir),
                benchmark=BenchmarkConfig(name="toy", split="test"),
            )
            
            # Run training
            result = run_retriever_training(
                app_config,
                self.config,
                num_train_samples=10,
                num_val_samples=4,
            )
            
            # Verify result structure
            self.assertIn("run_id", result)
            self.assertIn("summary", result)
            self.assertIn("training_result", result)
            
            # Verify experiment record was logged
            experiment_runs_path = self.output_dir / "experiment_runs.jsonl"
            self.assertTrue(experiment_runs_path.exists())
            
            with open(experiment_runs_path, "r") as f:
                lines = f.readlines()
            
            self.assertGreater(len(lines), 0)
            last_record = json.loads(lines[-1])
            self.assertIn("run_id", last_record)
            self.assertIn("checkpoint_path", last_record)
            self.assertIn("score", last_record)
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_seed_sweep_runner(self):
        """Test seed sweep runner with multiple seeds."""
        try:
            from graphrag_lab.configs.schema import BuilderConfig, ExplorerConfig, RetrieverConfig, ReaderConfig, DataConfig, BenchmarkConfig
            
            # Create minimal app config
            app_config = AppConfig(
                runtime=RuntimeConfig(
                    mode="test",
                    seed=42,
                    output_dir=self.output_dir,
                ),
                builder=BuilderConfig(min_edge_weight=0.5),
                explorer=ExplorerConfig(top_k=3),
                retriever=RetrieverConfig(top_k=3),
                reader=ReaderConfig(mode="extractive"),
                data=DataConfig(toy_data_path=self.temp_dir, graphragbench_data_path=self.temp_dir),
                benchmark=BenchmarkConfig(name="toy", split="test"),
            )
            
            # Run seed sweep with 2 seeds
            seeds = [7, 11]
            result = run_retriever_seed_sweep(
                app_config,
                self.config,
                seeds,
                num_train_samples=8,
                num_val_samples=4,
            )
            
            # Verify result structure
            self.assertIn("aggregate", result)
            self.assertIn("runs", result)
            
            aggregate = result["aggregate"]
            self.assertEqual(aggregate["num_runs"], 2)
            self.assertIn("mean_score", aggregate)
            self.assertIn("std_score", aggregate)
            self.assertIn("best_seed", aggregate)
            
            # Verify sweep summary file was created
            sweep_files = list(self.output_dir.glob("seed_sweep_retriever_*.json"))
            self.assertGreater(len(sweep_files), 0)
            
            # Verify experiment runs were logged
            experiment_runs_path = self.output_dir / "experiment_runs.jsonl"
            self.assertTrue(experiment_runs_path.exists())
            
            with open(experiment_runs_path, "r") as f:
                lines = f.readlines()
            
            # Should have at least 2 runs logged
            self.assertGreaterEqual(len(lines), 2)
        except ImportError:
            self.skipTest("PyTorch not available")


if __name__ == "__main__":
    unittest.main()
