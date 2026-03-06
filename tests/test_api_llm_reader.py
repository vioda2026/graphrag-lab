"""Tests for API-LLM Reader."""
import unittest
from pathlib import Path
import tempfile
import shutil
import os

from graphrag_lab.core.types import Query, RetrievalResult
from graphrag_lab.reader.api_llm_reader import (
    APILLMReader,
    APILLMReaderConfig,
    ResponseCache,
    RateLimiter,
)


class ResponseCacheTest(unittest.TestCase):
    """Test response cache functionality."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = ResponseCache(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_miss(self):
        """Test cache returns None for missing key."""
        result = self.cache.get("test prompt", "test-model")
        self.assertIsNone(result)
    
    def test_cache_set_get(self):
        """Test cache set and get."""
        self.cache.set("test prompt", "test-model", "test response")
        result = self.cache.get("test prompt", "test-model")
        self.assertEqual(result, "test response")
    
    def test_cache_key_uniqueness(self):
        """Test different prompts get different cache keys."""
        self.cache.set("prompt 1", "model", "response 1")
        self.cache.set("prompt 2", "model", "response 2")
        
        result1 = self.cache.get("prompt 1", "model")
        result2 = self.cache.get("prompt 2", "model")
        
        self.assertEqual(result1, "response 1")
        self.assertEqual(result2, "response 2")
    
    def test_cache_model_differentiation(self):
        """Test same prompt with different models gets different cache entries."""
        self.cache.set("same prompt", "model-1", "response 1")
        self.cache.set("same prompt", "model-2", "response 2")
        
        result1 = self.cache.get("same prompt", "model-1")
        result2 = self.cache.get("same prompt", "model-2")
        
        self.assertEqual(result1, "response 1")
        self.assertEqual(result2, "response 2")


class RateLimiterTest(unittest.TestCase):
    """Test rate limiter functionality."""
    
    def test_rate_limiter_allows_under_limit(self):
        """Test rate limiter allows calls under limit."""
        limiter = RateLimiter(max_calls=5, window_seconds=60)
        
        # Should allow 5 calls immediately
        for i in range(5):
            start = __import__('time').time()
            limiter.acquire()
            elapsed = __import__('time').time() - start
            self.assertLess(elapsed, 0.1)  # Should be nearly instant
    
    def test_rate_limiter_blocks_over_limit(self):
        """Test rate limiter blocks when over limit."""
        limiter = RateLimiter(max_calls=2, window_seconds=1)
        
        # Use up the limit
        limiter.acquire()
        limiter.acquire()
        
        # Next call should wait
        start = __import__('time').time()
        limiter.acquire()
        elapsed = __import__('time').time() - start
        
        # Should have waited close to 1 second
        self.assertGreaterEqual(elapsed, 0.8)


class APILLMReaderConfigTest(unittest.TestCase):
    """Test API-LLM Reader config."""
    
    def test_config_creation(self):
        """Test config can be created."""
        config = APILLMReaderConfig(
            api_base="https://api.test.com",
            api_key="test-key",
            model="test-model",
            max_tokens=1024,
            temperature=0.5,
            timeout_seconds=30,
            max_retries=3,
            retry_delay_seconds=1.0,
            rate_limit_calls=60,
            rate_limit_window_seconds=60,
            cache_dir=None,
            use_cache=False,
        )
        
        self.assertEqual(config.api_base, "https://api.test.com")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.max_tokens, 1024)


class APILLMReaderTest(unittest.TestCase):
    """Test API-LLM Reader (without actual API calls)."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = APILLMReaderConfig(
            api_base="https://api.test.com",
            api_key="test-key",
            model="test-model",
            max_tokens=1024,
            temperature=0.3,
            timeout_seconds=30,
            max_retries=3,
            retry_delay_seconds=0.1,
            rate_limit_calls=60,
            rate_limit_window_seconds=60,
            cache_dir=self.temp_dir,
            use_cache=True,
        )
        self.reader = APILLMReader(self.config)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_prompt_building(self):
        """Test prompt is built correctly from query and evidence."""
        query = Query(question="What is GraphRAG?")
        retrieved = RetrievalResult(
            node_ids=["node1", "node2"],
            passages=["GraphRAG is a graph-based RAG method.", "It uses knowledge graphs."],
            scores=[0.9, 0.8],
        )
        
        prompt = self.reader._build_prompt(query, retrieved)
        
        self.assertIn("What is GraphRAG?", prompt)
        self.assertIn("[Evidence 1]: GraphRAG is a graph-based RAG method.", prompt)
        self.assertIn("[Evidence 2]: It uses knowledge graphs.", prompt)
    
    def test_read_with_cache(self):
        """Test read uses cache when available."""
        query = Query(question="Test question?")
        retrieved = RetrievalResult(
            node_ids=["node1"],
            passages=["Test passage"],
            scores=[0.9],
        )
        
        # Pre-populate cache
        prompt = self.reader._build_prompt(query, retrieved)
        self.reader.cache.set(prompt, self.config.model, "Cached answer")
        
        # Read should use cache
        result = self.reader.read(query, retrieved)
        self.assertEqual(result.answer, "Cached answer")
    
    def test_reader_initialization(self):
        """Test reader can be initialized."""
        self.assertIsNotNone(self.reader)
        self.assertIsNotNone(self.reader.rate_limiter)
        self.assertIsNotNone(self.reader.cache)


if __name__ == "__main__":
    unittest.main()
