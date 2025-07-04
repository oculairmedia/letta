"""
Redis-based Agent State Caching for Letta

This module provides high-performance caching for agent states, memory blocks,
and message histories to dramatically reduce database load and improve response times.
"""

import asyncio
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from functools import wraps

from letta.schemas.agent import AgentState as PydanticAgentState
from letta.schemas.block import Block as PydanticBlock
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.memory import Memory
from letta.schemas.tool import Tool as PydanticTool

from .redis_cache import LettaRedisCache


class AgentStateCache(LettaRedisCache):
    """Specialized cache for Letta agent states and related data."""
    
    def __init__(self, redis_url: str = None, agent_ttl: int = 3600, message_ttl: int = 1800):
        """
        Initialize the agent state cache.
        
        Args:
            redis_url: Redis connection URL
            agent_ttl: TTL for agent state (default 1 hour)
            message_ttl: TTL for message cache (default 30 minutes)
        """
        super().__init__(
            redis_url=redis_url,
            prefix="letta:agent:",
            ttl=agent_ttl,
            enable_pickle=True
        )
        self.agent_ttl = agent_ttl
        self.message_ttl = message_ttl
        
    # === Agent State Caching ===
    
    def cache_agent_state(self, agent_id: str, state: PydanticAgentState) -> bool:
        """Cache complete agent state."""
        key = f"state:{agent_id}"
        # Convert to dict for JSON serialization
        state_dict = state.model_dump()
        return self.set(key, state_dict, ttl=self.agent_ttl)
    
    async def cache_agent_state_async(self, agent_id: str, state: PydanticAgentState) -> bool:
        """Async version of cache_agent_state."""
        key = f"state:{agent_id}"
        state_dict = state.model_dump()
        return await self.set_async(key, state_dict, ttl=self.agent_ttl)
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached agent state."""
        key = f"state:{agent_id}"
        return self.get(key)
    
    async def get_agent_state_async(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Async version of get_agent_state."""
        key = f"state:{agent_id}"
        return await self.get_async(key)
    
    def invalidate_agent_state(self, agent_id: str) -> bool:
        """Invalidate agent state cache."""
        key = f"state:{agent_id}"
        return self.delete(key)
    
    async def invalidate_agent_state_async(self, agent_id: str) -> bool:
        """Async version of invalidate_agent_state."""
        key = f"state:{agent_id}"
        return await self.delete_async(key)
    
    # === Memory Block Caching ===
    
    def cache_memory_blocks(self, agent_id: str, blocks: List[PydanticBlock]) -> bool:
        """Cache memory blocks for an agent."""
        key = f"blocks:{agent_id}"
        block_data = [block.model_dump() for block in blocks]
        return self.set(key, block_data, ttl=self.agent_ttl)
    
    async def cache_memory_blocks_async(self, agent_id: str, blocks: List[PydanticBlock]) -> bool:
        """Async version of cache_memory_blocks."""
        key = f"blocks:{agent_id}"
        block_data = [block.model_dump() for block in blocks]
        return await self.set_async(key, block_data, ttl=self.agent_ttl)
    
    def get_memory_blocks(self, agent_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached memory blocks."""
        key = f"blocks:{agent_id}"
        return self.get(key)
    
    async def get_memory_blocks_async(self, agent_id: str) -> Optional[List[Dict[str, Any]]]:
        """Async version of get_memory_blocks."""
        key = f"blocks:{agent_id}"
        return await self.get_async(key)
    
    def cache_single_block(self, agent_id: str, block_id: str, block: PydanticBlock) -> bool:
        """Cache individual memory block."""
        key = f"block:{agent_id}:{block_id}"
        return self.set(key, block.model_dump(), ttl=self.agent_ttl)
    
    async def cache_single_block_async(self, agent_id: str, block_id: str, block: PydanticBlock) -> bool:
        """Async version of cache_single_block."""
        key = f"block:{agent_id}:{block_id}"
        return await self.set_async(key, block.model_dump(), ttl=self.agent_ttl)
    
    # === Message History Caching ===
    
    def cache_recent_messages(self, agent_id: str, messages: List[PydanticMessage], limit: int = 100) -> bool:
        """Cache recent messages for an agent."""
        key = f"messages:recent:{agent_id}"
        # Keep only the most recent messages
        recent_messages = messages[-limit:] if len(messages) > limit else messages
        message_data = [msg.model_dump() for msg in recent_messages]
        return self.set(key, message_data, ttl=self.message_ttl)
    
    async def cache_recent_messages_async(self, agent_id: str, messages: List[PydanticMessage], limit: int = 100) -> bool:
        """Async version of cache_recent_messages."""
        key = f"messages:recent:{agent_id}"
        recent_messages = messages[-limit:] if len(messages) > limit else messages
        message_data = [msg.model_dump() for msg in recent_messages]
        return await self.set_async(key, message_data, ttl=self.message_ttl)
    
    def get_recent_messages(self, agent_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached recent messages."""
        key = f"messages:recent:{agent_id}"
        return self.get(key)
    
    async def get_recent_messages_async(self, agent_id: str) -> Optional[List[Dict[str, Any]]]:
        """Async version of get_recent_messages."""
        key = f"messages:recent:{agent_id}"
        return await self.get_async(key)
    
    def append_message(self, agent_id: str, message: PydanticMessage, limit: int = 100) -> bool:
        """Append a new message to the cache."""
        messages = self.get_recent_messages(agent_id) or []
        messages.append(message.model_dump())
        # Keep only recent messages
        if len(messages) > limit:
            messages = messages[-limit:]
        key = f"messages:recent:{agent_id}"
        return self.set(key, messages, ttl=self.message_ttl)
    
    async def append_message_async(self, agent_id: str, message: PydanticMessage, limit: int = 100) -> bool:
        """Async version of append_message."""
        messages = await self.get_recent_messages_async(agent_id) or []
        messages.append(message.model_dump())
        if len(messages) > limit:
            messages = messages[-limit:]
        key = f"messages:recent:{agent_id}"
        return await self.set_async(key, messages, ttl=self.message_ttl)
    
    # === Context Window Caching ===
    
    def cache_context_window(self, agent_id: str, context_hash: str, context_data: Dict[str, Any]) -> bool:
        """Cache compiled context window."""
        key = f"context:{agent_id}:{context_hash}"
        return self.set(key, context_data, ttl=self.message_ttl)
    
    async def cache_context_window_async(self, agent_id: str, context_hash: str, context_data: Dict[str, Any]) -> bool:
        """Async version of cache_context_window."""
        key = f"context:{agent_id}:{context_hash}"
        return await self.set_async(key, context_data, ttl=self.message_ttl)
    
    def get_context_window(self, agent_id: str, context_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context window."""
        key = f"context:{agent_id}:{context_hash}"
        return self.get(key)
    
    async def get_context_window_async(self, agent_id: str, context_hash: str) -> Optional[Dict[str, Any]]:
        """Async version of get_context_window."""
        key = f"context:{agent_id}:{context_hash}"
        return await self.get_async(key)
    
    # === Session Management ===
    
    def create_session(self, agent_id: str, session_data: Dict[str, Any], session_ttl: int = 900) -> bool:
        """Create or update an active session (default 15 min TTL)."""
        key = f"session:{agent_id}"
        session_data["last_activity"] = datetime.utcnow().isoformat()
        return self.set(key, session_data, ttl=session_ttl)
    
    async def create_session_async(self, agent_id: str, session_data: Dict[str, Any], session_ttl: int = 900) -> bool:
        """Async version of create_session."""
        key = f"session:{agent_id}"
        session_data["last_activity"] = datetime.utcnow().isoformat()
        return await self.set_async(key, session_data, ttl=session_ttl)
    
    def get_session(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve active session."""
        key = f"session:{agent_id}"
        return self.get(key)
    
    async def get_session_async(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Async version of get_session."""
        key = f"session:{agent_id}"
        return await self.get_async(key)
    
    def touch_session(self, agent_id: str, session_ttl: int = 900) -> bool:
        """Update session last activity time."""
        session = self.get_session(agent_id)
        if session:
            session["last_activity"] = datetime.utcnow().isoformat()
            key = f"session:{agent_id}"
            return self.set(key, session, ttl=session_ttl)
        return False
    
    async def touch_session_async(self, agent_id: str, session_ttl: int = 900) -> bool:
        """Async version of touch_session."""
        session = await self.get_session_async(agent_id)
        if session:
            session["last_activity"] = datetime.utcnow().isoformat()
            key = f"session:{agent_id}"
            return await self.set_async(key, session, ttl=session_ttl)
        return False
    
    def end_session(self, agent_id: str) -> bool:
        """End an active session."""
        key = f"session:{agent_id}"
        return self.delete(key)
    
    async def end_session_async(self, agent_id: str) -> bool:
        """Async version of end_session."""
        key = f"session:{agent_id}"
        return await self.delete_async(key)
    
    # === Batch Operations ===
    
    def invalidate_agent_cache(self, agent_id: str) -> int:
        """Invalidate all caches for an agent."""
        pattern = f"{agent_id}:*"
        keys = list(self._redis.scan_iter(match=f"{self.prefix}{pattern}"))
        if keys:
            return self._redis.delete(*keys)
        return 0
    
    async def invalidate_agent_cache_async(self, agent_id: str) -> int:
        """Async version of invalidate_agent_cache."""
        pattern = f"{agent_id}:*"
        keys = []
        async for key in self._redis_async.scan_iter(match=f"{self.prefix}{pattern}"):
            keys.append(key)
        if keys:
            return await self._redis_async.delete(*keys)
        return 0
    
    # === Cache Warming ===
    
    async def warm_cache_for_agent(self, agent_id: str, state: PydanticAgentState, 
                                   messages: List[PydanticMessage] = None) -> Dict[str, bool]:
        """Warm all caches for an agent."""
        results = {}
        
        # Cache agent state
        results["state"] = await self.cache_agent_state_async(agent_id, state)
        
        # Cache memory blocks
        if state.memory and state.memory.blocks:
            results["blocks"] = await self.cache_memory_blocks_async(agent_id, state.memory.blocks)
        
        # Cache recent messages if provided
        if messages:
            results["messages"] = await self.cache_recent_messages_async(agent_id, messages)
        
        # Create session
        session_data = {
            "agent_id": agent_id,
            "created": datetime.utcnow().isoformat(),
            "warm_cache": True
        }
        results["session"] = await self.create_session_async(agent_id, session_data)
        
        return results
    
    # === Statistics ===
    
    def get_agent_cache_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get cache statistics for a specific agent."""
        stats = {
            "agent_id": agent_id,
            "cached_items": {}
        }
        
        # Check what's cached
        stats["cached_items"]["state"] = bool(self.get_agent_state(agent_id))
        stats["cached_items"]["blocks"] = bool(self.get_memory_blocks(agent_id))
        stats["cached_items"]["messages"] = bool(self.get_recent_messages(agent_id))
        stats["cached_items"]["session"] = bool(self.get_session(agent_id))
        
        # Get cache sizes
        pattern = f"{agent_id}:*"
        keys = list(self._redis.scan_iter(match=f"{self.prefix}{pattern}"))
        stats["total_keys"] = len(keys)
        
        return stats


# Singleton instance
_agent_cache_instance = None


def get_agent_cache(redis_url: str = None) -> AgentStateCache:
    """Get singleton agent cache instance."""
    global _agent_cache_instance
    if _agent_cache_instance is None:
        import os
        redis_url = redis_url or os.getenv('LETTA_REDIS_URL', 'redis://redis:6379/0')
        _agent_cache_instance = AgentStateCache(redis_url=redis_url)
    return _agent_cache_instance


# Decorators for caching

def cache_agent_state(ttl: int = None):
    """Decorator to cache agent state results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, agent_id: str, *args, **kwargs):
            cache = get_agent_cache()
            
            # Try to get from cache
            cached = await cache.get_agent_state_async(agent_id)
            if cached:
                cache.stats["cache_hits"] += 1
                # Convert dict back to Pydantic model
                from letta.schemas.agent import AgentState
                return AgentState(**cached)
            
            # Cache miss - call original function
            cache.stats["cache_misses"] += 1
            result = await func(self, agent_id, *args, **kwargs)
            
            # Cache the result
            if result:
                await cache.cache_agent_state_async(agent_id, result)
            
            return result
        return wrapper
    return decorator