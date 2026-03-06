"""Tests for retriever training loop scaffold."""
import unittest
from pathlib import Path
import tempfile
import shutil

from graphrag_lab.configs.schema import RetrieverTrainingConfig
from graphrag_lab.retriever.trainer import RetrieverTrainer, TrainingSample


class RetrieverTrainerTest(unittest.TestCase):
    """Test retriever training loop scaffold."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = RetrieverTrainingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=4,
            learning_rate=2e-5,
            num_epochs=2,
            checkpoint_dir=self.temp_dir / "checkpoints",
            warmup_ratio=0.1,
            max_length=512,
            margin=0.3,
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized (skip if torch not available)."""
        try:
            trainer = RetrieverTrainer(self.config)
            self.assertIsNotNone(trainer)
            self.assertEqual(trainer.config.model_name, self.config.model_name)
        except ImportError:
            # PyTorch not available - skip test
            self.skipTest("PyTorch not available")
    
    def test_training_sample_creation(self):
        """Test training sample data structure."""
        sample = TrainingSample(
            query="What is GraphRAG?",
            positive_node_id="node_1",
            positive_text="GraphRAG is a graph-based retrieval method",
            negative_node_ids=["node_2", "node_3"],
            negative_texts=["Unrelated text 1", "Unrelated text 2"],
        )
        
        self.assertEqual(sample.query, "What is GraphRAG?")
        self.assertEqual(sample.positive_node_id, "node_1")
        self.assertEqual(len(sample.negative_node_ids), 2)
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load cycle."""
        try:
            trainer = RetrieverTrainer(self.config)
            checkpoint = trainer.save_checkpoint(epoch=1, step=100, loss=0.5)
            
            self.assertEqual(checkpoint.epoch, 1)
            self.assertEqual(checkpoint.step, 100)
            self.assertAlmostEqual(checkpoint.loss, 0.5)
            self.assertTrue(checkpoint.checkpoint_path.parent.exists())
            
            # Load back
            loaded = trainer.load_checkpoint(checkpoint.checkpoint_path)
            self.assertEqual(loaded.epoch, checkpoint.epoch)
            self.assertEqual(loaded.step, checkpoint.step)
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_dataloader_creation(self):
        """Test DataLoader can be created from samples."""
        try:
            trainer = RetrieverTrainer(self.config)
            
            samples = [
                TrainingSample(
                    query=f"Query {i}",
                    positive_node_id=f"node_{i}",
                    positive_text=f"Positive text {i}",
                    negative_node_ids=[f"neg_{i}_1"],
                    negative_texts=[f"Negative text {i}"],
                )
                for i in range(10)
            ]
            
            dataloader = trainer.create_dataloader(samples, shuffle=True)
            self.assertEqual(len(dataloader), 3)  # 10 samples / batch_size 4 = 3 batches
            
            # Iterate through dataloader
            for batch in dataloader:
                self.assertIsInstance(batch, list)
                self.assertLessEqual(len(batch), self.config.batch_size)
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_train_step(self):
        """Test train_step method for batch training."""
        try:
            trainer = RetrieverTrainer(self.config)
            
            # Create a mock model for testing
            class MockModel:
                def train(self):
                    pass
                def eval(self):
                    pass
            
            trainer.model = MockModel()
            
            batch = [
                TrainingSample(
                    query="Test query",
                    positive_node_id="node_1",
                    positive_text="Positive text",
                    negative_node_ids=["node_2"],
                    negative_texts=["Negative text"],
                )
            ]
            
            loss = trainer.train_step(batch)
            self.assertIsInstance(loss, float)
            self.assertGreaterEqual(loss, 0.0)
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_validate(self):
        """Test validate method returns metrics dict."""
        try:
            trainer = RetrieverTrainer(self.config)
            
            # Create a mock model for testing
            class MockModel:
                def train(self):
                    pass
                def eval(self):
                    pass
            
            trainer.model = MockModel()
            
            samples = [
                TrainingSample(
                    query=f"Query {i}",
                    positive_node_id=f"node_{i}",
                    positive_text=f"Positive text {i}",
                    negative_node_ids=[f"neg_{i}_1"],
                    negative_texts=[f"Negative text {i}"],
                )
                for i in range(4)
            ]
            
            metrics = trainer.validate(samples)
            self.assertIsInstance(metrics, dict)
            self.assertIn("val_loss", metrics)
            self.assertIsInstance(metrics["val_loss"], float)
        except ImportError:
            self.skipTest("PyTorch not available")


if __name__ == "__main__":
    unittest.main()
