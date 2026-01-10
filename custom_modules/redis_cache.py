"""
Redis-based caching implementation for Letta performance optimization.
This module provides a drop-in replacement for Letta's in-memory caching.
"""

import json
import os
from typing import Any, Dict, Optional, Union
import redis
from redis import asyncio as aioredis
import pickle
import hashlib
from functools import wraps
import asyncio


class LettaRedisCache:
    """Redis cache implementation for Letta configuration and state caching."""
    
    def __init__(
        self,
        redis_url: str = None,
        prefix: str = "letta:",
        ttl: int = 3600,
        enable_pickle: bool = True
    ):
        """
        Initialize the Redis cache.
        
        Args:
            redis_url: Redis connection URL (defaults to env var REDIS_URL or localhost)
            prefix: Prefix for all cache keys
            ttl: Default time-to-live in seconds (1 hour)
            enable_pickle: Whether to use pickle for complex objects
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.prefix = prefix
        self.ttl = ttl
        self.enable_pickle = enable_pickle
        
        # Initialize sync and async Redis clients
        self._sync_client = None
        self._async_client = None
    
    @property
    def sync_client(self) -> redis.Redis:
        """Get or create sync Redis client."""
        if self._sync_client is None:
            self._sync_client = redis.from_url(
                self.redis_url,
                decode_responses=not self.enable_pickle
            )
        return self._sync_client
    
    @property
    async def async_client(self) -> aioredis.Redis:
        """Get or create async Redis client."""
        if self._async_client is None:
            self._async_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=not self.enable_pickle
            )
        return self._async_client
    
    def _make_key(self, key: str) -> str:
        """Generate prefixed cache key."""
        return f"{self.prefix}{key}"
    
    def _serialize(self, value: Any) -> Union[str, bytes]:
        """Serialize value for storage."""
        if self.enable_pickle:
            return pickle.dumps(value)
        return json.dumps(value)
    
    def _deserialize(self, value: Union[str, bytes]) -> Any:
        """Deserialize value from storage."""
        if value is None:
            return None
        if self.enable_pickle:
            return pickle.loads(value)
        return json.loads(value)
    
    # Sync methods
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = self.sync_client.get(self._make_key(key))
        return self._deserialize(value)
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with optional TTL."""
        return self.sync_client.set(
            self._make_key(key),
            self._serialize(value),
            ex=ttl or self.ttl
        )
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return bool(self.sync_client.delete(self._make_key(key)))
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        keys = self.sync_client.keys(self._make_key(pattern))
        if keys:
            return self.sync_client.delete(*keys)
        return 0
    
    # Async methods
    async def get_async(self, key: str) -> Optional[Any]:
        """Async get value from cache."""
        client = await self.async_client
        value = await client.get(self._make_key(key))
        return self._deserialize(value)
    
    async def set_async(self, key: str, value: Any, ttl: int = None) -> bool:
        """Async set value in cache with optional TTL."""
        client = await self.async_client
        return await client.set(
            self._make_key(key),
            self._serialize(value),
            ex=ttl or self.ttl
        )
    
    async def delete_async(self, key: str) -> bool:
        """Async delete key from cache."""
        client = await self.async_client
        return bool(await client.delete(self._make_key(key)))
    
    # Decorator for caching
    def cache(self, ttl: int = None, key_func=None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result
            
            return wrapper
        return decorator
    
    def cache_async(self, ttl: int = None, key_func=None):
        """Decorator for caching async function results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # Try to get from cache
                result = await self.get_async(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set_async(cache_key, result, ttl=ttl)
                return result
            
            return wrapper
        return decorator


class LettaConfigCache(LettaRedisCache):
    """Specialized cache for Letta LLM and embedding configurations."""
    
    def __init__(self, redis_url: str = None):
        super().__init__(
            redis_url=redis_url,
            prefix="letta:config:",
            ttl=3600,  # 1 hour TTL for configs
            enable_pickle=True
        )
        
        # Track cache stats
        self.stats_key = "stats"
    
    def get_llm_config(self, actor_id: str, **kwargs) -> Optional[Dict]:
        """Get cached LLM configuration."""
        key = self._make_config_key("llm", actor_id, **kwargs)
        config = self.get(key)
        if config:
            self._increment_stat("llm_hits")
        else:
            self._increment_stat("llm_misses")
        return config
    
    def set_llm_config(self, actor_id: str, config: Dict, **kwargs) -> bool:
        """Cache LLM configuration."""
        key = self._make_config_key("llm", actor_id, **kwargs)
        return self.set(key, config)
    
    def get_embedding_config(self, actor_id: str, **kwargs) -> Optional[Dict]:
        """Get cached embedding configuration."""
        key = self._make_config_key("embedding", actor_id, **kwargs)
        config = self.get(key)
        if config:
            self._increment_stat("embedding_hits")
        else:
            self._increment_stat("embedding_misses")
        return config
    
    def set_embedding_config(self, actor_id: str, config: Dict, **kwargs) -> bool:
        """Cache embedding configuration."""
        key = self._make_config_key("embedding", actor_id, **kwargs)
        return self.set(key, config)
    
    async def get_llm_config_async(self, actor_id: str, **kwargs) -> Optional[Dict]:
        """Async get cached LLM configuration."""
        key = self._make_config_key("llm", actor_id, **kwargs)
        config = await self.get_async(key)
        if config:
            await self._increment_stat_async("llm_hits")
        else:
            await self._increment_stat_async("llm_misses")
        return config
    
    async def set_llm_config_async(self, actor_id: str, config: Dict, **kwargs) -> bool:
        """Async cache LLM configuration."""
        key = self._make_config_key("llm", actor_id, **kwargs)
        return await self.set_async(key, config)
    
    def _make_config_key(self, config_type: str, actor_id: str, **kwargs) -> str:
        """Generate a cache key for configurations."""
        key_parts = [config_type, actor_id]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()
    
    def _increment_stat(self, stat_name: str):
        """Increment cache statistics."""
        self.sync_client.hincrby(self._make_key(self.stats_key), stat_name, 1)
    
    async def _increment_stat_async(self, stat_name: str):
        """Async increment cache statistics."""
        client = await self.async_client
        await client.hincrby(self._make_key(self.stats_key), stat_name, 1)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        stats = self.sync_client.hgetall(self._make_key(self.stats_key))
        return {k.decode() if isinstance(k, bytes) else k: int(v) for k, v in stats.items()}
    
    def clear_all_configs(self):
        """Clear all cached configurations."""
        return self.clear_pattern("*")


# Drop-in replacement functions for Letta's caching
def make_key(**kwargs) -> str:
    """Generate cache key compatible with Letta's existing code."""
    key_parts = []
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    return hashlib.md5(":".join(key_parts).encode()).hexdigest()


# Example usage and integration
if __name__ == "__main__":
    # Example: Initialize cache
    cache = LettaConfigCache(redis_url="redis://localhost:6379/0")
    
    # Example: Cache LLM config
    test_config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    # Set config
    cache.set_llm_config("user123", test_config, model="gpt-4")
    
    # Get config
    retrieved = cache.get_llm_config("user123", model="gpt-4")
    print(f"Retrieved config: {retrieved}")
    
    # Get stats
    print(f"Cache stats: {cache.get_stats()}")