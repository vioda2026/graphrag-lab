"""API-LLM Reader client with retry, rate-limit, and response caching."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from graphrag_lab.core.types import Query, ReadResult, RetrievalResult
from graphrag_lab.reader.base import Reader


@dataclass
class APILLMReaderConfig:
    """Configuration for API-LLM Reader."""
    api_base: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    timeout_seconds: int
    max_retries: int
    retry_delay_seconds: float
    rate_limit_calls: int  # Max calls per window
    rate_limit_window_seconds: int
    cache_dir: Optional[Path]
    use_cache: bool


class ResponseCache:
    """Simple file-based response cache for LLM API calls."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if exists."""
        key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Check if cache is still valid (within 24 hours)
                    if time.time() - data.get("timestamp", 0) < 86400:
                        return data.get("response")
            except (json.JSONDecodeError, IOError):
                pass
        return None
    
    def set(self, prompt: str, model: str, response: str) -> None:
        """Cache a response."""
        key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(key)
        
        data = {
            "prompt": prompt,
            "model": model,
            "response": response,
            "timestamp": time.time(),
        }
        
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.call_times: List[float] = []
    
    def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.time()
        
        # Remove old calls outside the window
        self.call_times = [t for t in self.call_times if now - t < self.window_seconds]
        
        # If at limit, wait until oldest call expires
        if len(self.call_times) >= self.max_calls:
            wait_time = self.window_seconds - (now - self.call_times[0])
            if wait_time > 0:
                time.sleep(wait_time)
                # Clean up again after waiting
                now = time.time()
                self.call_times = [t for t in self.call_times if now - t < self.window_seconds]
        
        # Record this call
        self.call_times.append(now)


class APILLMReader(Reader):
    """LLM-based reader using external API with retry, rate-limit, and caching."""
    
    def __init__(self, config: APILLMReaderConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            config.rate_limit_calls,
            config.rate_limit_window_seconds
        )
        self.cache = ResponseCache(config.cache_dir) if config.cache_dir and config.use_cache else None
    
    def _build_prompt(self, query: Query, retrieved: RetrievalResult) -> str:
        """Build prompt for LLM from query and retrieved evidence."""
        evidence_text = "\n\n".join([
            f"[Evidence {i+1}]: {passage}"
            for i, passage in enumerate(retrieved.passages)
        ])
        
        prompt = f"""You are a research assistant. Answer the question based on the provided evidence.

Question: {query.question}

Evidence:
{evidence_text}

Instructions:
- Answer based ONLY on the provided evidence
- If the evidence doesn't contain enough information, say so
- Be concise but complete
- Cite which evidence supports your answer

Answer:"""
        
        return prompt
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to LLM with retry logic."""
        import urllib.request
        import urllib.error
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.acquire()
                
                # Make request
                req = urllib.request.Request(
                    f"{self.config.api_base}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    return result["choices"][0]["message"]["content"]
                    
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429:  # Rate limit exceeded
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                elif e.code >= 500:  # Server error, retry
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    break  # Client error, don't retry
                    
            except Exception as e:
                last_error = e
                wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                time.sleep(wait_time)
                continue
        
        # All retries exhausted
        error_msg = f"API call failed after {self.config.max_retries} attempts"
        if last_error:
            error_msg += f": {str(last_error)}"
        raise RuntimeError(error_msg)
    
    def read(self, query: Query, retrieved: RetrievalResult) -> ReadResult:
        """
        Generate answer using LLM API.
        
        Args:
            query: The query to answer
            retrieved: Retrieved evidence passages
        
        Returns:
            ReadResult with LLM-generated answer and supporting passages
        """
        # Build prompt
        prompt = self._build_prompt(query, retrieved)
        
        # Check cache first
        if self.cache:
            cached_response = self.cache.get(prompt, self.config.model)
            if cached_response:
                return ReadResult(
                    answer=cached_response,
                    supporting_passages=retrieved.passages[:2]
                )
        
        # Call API
        response = self._call_api(prompt)
        
        # Cache response
        if self.cache:
            self.cache.set(prompt, self.config.model, response)
        
        return ReadResult(
            answer=response,
            supporting_passages=retrieved.passages[:2]
        )


def create_api_llm_reader_from_env() -> APILLMReader:
    """Create APILLMReader from environment variables."""
    import os
    
    config = APILLMReaderConfig(
        api_base=os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT", "30")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        retry_delay_seconds=float(os.getenv("LLM_RETRY_DELAY", "1.0")),
        rate_limit_calls=int(os.getenv("LLM_RATE_LIMIT_CALLS", "60")),
        rate_limit_window_seconds=int(os.getenv("LLM_RATE_LIMIT_WINDOW", "60")),
        cache_dir=Path(os.getenv("LLM_CACHE_DIR", "~/.cache/graphrag_lab/reader")).expanduser(),
        use_cache=os.getenv("LLM_USE_CACHE", "true").lower() == "true",
    )
    
    if not config.api_key:
        raise ValueError("LLM_API_KEY environment variable is required")
    
    return APILLMReader(config)
