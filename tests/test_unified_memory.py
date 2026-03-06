"""Tests for unified memory module."""
import unittest
import time
import tempfile
import shutil
from pathlib import Path

from graphrag_lab.memory.unified_mem import (
    MemoryRecord,
    MemoryQuery,
    MemoryResult,
    InMemoryStorage,
    SimpleMemoryController,
    MemoryAwareRetriever,
    MemoryManager,
)


class MemoryRecordTest(unittest.TestCase):
    """Test MemoryRecord dataclass."""
    
    def test_memory_record_creation(self):
        """Test creating a memory record."""
        record = MemoryRecord(
            key="test_key",
            content="test content",
            created_at=1234567890.0,
            last_accessed=1234567890.0,
            access_count=0,
            metadata={"source": "test"}
        )
        
        self.assertEqual(record.key, "test_key")
        self.assertEqual(record.content, "test content")
        self.assertEqual(record.metadata["source"], "test")


class InMemoryStorageTest(unittest.TestCase):
    """Test InMemoryStorage implementation."""
    
    def setUp(self):
        self.storage = InMemoryStorage()
    
    def test_write_and_read(self):
        """Test writing and reading records."""
        record = self.storage.write("key1", "content1", {"meta": "value"})
        
        self.assertEqual(record.key, "key1")
        self.assertEqual(record.content, "content1")
        self.assertEqual(record.access_count, 0)
        
        # Read should increment access count
        read_record = self.storage.read("key1")
        self.assertEqual(read_record.access_count, 1)
    
    def test_read_nonexistent(self):
        """Test reading nonexistent key."""
        result = self.storage.read("nonexistent")
        self.assertIsNone(result)
    
    def test_delete(self):
        """Test deleting records."""
        self.storage.write("key1", "content1")
        self.assertTrue(self.storage.delete("key1"))
        self.assertFalse(self.storage.delete("key1"))  # Already deleted
    
    def test_update_access(self):
        """Test updating access time."""
        self.storage.write("key1", "content1")
        record = self.storage.read("key1")
        initial_count = record.access_count
        
        self.storage.update_access("key1")
        updated = self.storage.read("key1")
        self.assertEqual(updated.access_count, initial_count + 1)


class SimpleMemoryControllerTest(unittest.TestCase):
    """Test SimpleMemoryController."""
    
    def setUp(self):
        self.storage = InMemoryStorage()
        self.controller = SimpleMemoryController(self.storage, max_records=3)
    
    def test_init(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.max_records, 3)
        self.assertEqual(self.controller.eviction_policy, "lru")
    
    def test_should_evict_below_limit(self):
        """Test should_evict when below limit."""
        self.storage.write("key1", "content1")
        self.assertFalse(self.controller.should_evict())
    
    def test_should_evict_above_limit(self):
        """Test should_evict when above limit."""
        for i in range(4):
            self.storage.write(f"key{i}", f"content{i}")
        self.assertTrue(self.controller.should_evict())
    
    def test_evict(self):
        """Test evicting records."""
        # Add records
        for i in range(4):
            self.storage.write(f"key{i}", f"content{i}")
        
        evicted = self.controller.evict(count=1)
        self.assertEqual(len(evicted), 1)
        self.assertIn(evicted[0], ["key0", "key1", "key2", "key3"])
    
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        for i in range(2):
            self.storage.write(f"key{i}", f"content{i}")
        
        stats = self.controller.get_memory_stats()
        self.assertEqual(stats["total_records"], 2)
        self.assertEqual(stats["max_records"], 3)
        self.assertFalse(stats["needs_eviction"])


class MemoryAwareRetrieverTest(unittest.TestCase):
    """Test MemoryAwareRetriever."""
    
    def setUp(self):
        self.storage = InMemoryStorage()
        self.controller = SimpleMemoryController(self.storage, max_records=10)
        self.retriever = MemoryAwareRetriever(self.storage, self.controller, memory_weight=0.4)
    
    def test_load_memory_context(self):
        """Test loading memory context."""
        self.storage.write("key1", "content1")
        self.storage.write("key2", "content2")
        
        context = self.retriever.load_memory_context("test query")
        self.assertEqual(len(context), 2)
        self.assertIn("content1", context)
        self.assertIn("content2", context)
    
    def test_compute_hybrid_score(self):
        """Test computing hybrid score."""
        graph_score = 0.8
        memory_score = 0.6
        # hybrid = 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
        hybrid = self.retriever.compute_hybrid_score(graph_score, memory_score)
        self.assertAlmostEqual(hybrid, 0.72, places=5)
    
    def test_should_use_memory(self):
        """Test memory usage decision."""
        self.assertTrue(self.retriever.should_use_memory("test query"))


class MemoryManagerTest(unittest.TestCase):
    """Test MemoryManager."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = MemoryManager(storage_path=self.temp_dir, max_records=3)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_remember(self):
        """Test remembering a record."""
        record = self.manager.remember("key1", "content1", {"meta": "value"})
        
        self.assertEqual(record.key, "key1")
        self.assertEqual(record.content, "content1")
        self.assertEqual(self.manager.recall("key1").content, "content1")
    
    def test_remember_triggers_eviction(self):
        """Test that remember triggers eviction."""
        # Add 3 records (at limit)
        for i in range(3):
            self.manager.remember(f"key{i}", f"content{i}")
        
        # Add one more - should trigger eviction
        new_record = self.manager.remember("key3", "content3")
        
        # Should have evicted one
        stats = self.manager.get_stats()
        self.assertLessEqual(stats["total_records"], stats["max_records"])
    
    def test_recall(self):
        """Test recalling a record."""
        self.manager.remember("key1", "content1")
        record = self.manager.recall("key1")
        
        self.assertEqual(record.key, "key1")
        self.assertEqual(record.content, "content1")
    
    def test_recall_similar(self):
        """Test recall similar records."""
        self.manager.remember("key1", "content about graphrag")
        self.manager.remember("key2", "content about rag")
        
        result = self.manager.recall_similar("query about rag")
        self.assertIsInstance(result, MemoryResult)
    
    def test_get_stats(self):
        """Test getting memory statistics."""
        stats = self.manager.get_stats()
        
        self.assertIn("total_records", stats)
        self.assertIn("max_records", stats)
        self.assertIn("retriever_memory_weight", stats)
    
    def test_save_and_load(self):
        """Test saving and loading memory."""
        self.manager.remember("key1", "content1")
        self.manager.remember("key2", "content2")
        
        save_path = self.temp_dir / "memory.dat"
        self.manager.save(save_path)
        self.assertTrue(save_path.exists())
        
        # Load in new manager
        new_manager = MemoryManager(storage_path=self.temp_dir)
        new_manager.load(save_path)


if __name__ == "__main__":
    unittest.main()
