#!/usr/bin/env python3
"""Test Redis cache integration with Letta"""

import sys
sys.path.append('/app')

from letta.cache.redis_cache import LettaConfigCache
import os

def test_redis_cache():
    """Test Redis cache functionality"""
    
    redis_url = os.getenv('LETTA_REDIS_URL', 'redis://redis:6379/0')
    print(f"Testing Redis cache with URL: {redis_url}")
    
    try:
        # Initialize cache
        cache = LettaConfigCache(redis_url=redis_url)
        print("✓ Cache initialized successfully")
        
        # Test basic operations
        test_config = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        # Set config
        success = cache.set_llm_config("test_user_123", test_config, model="gpt-4")
        print(f"✓ Set LLM config: {success}")
        
        # Get config
        retrieved = cache.get_llm_config("test_user_123", model="gpt-4")
        print(f"✓ Retrieved config: {retrieved}")
        
        # Check if configs match
        if retrieved == test_config:
            print("✓ Cache working correctly - configs match!")
        else:
            print("✗ Cache error - configs don't match")
            
        # Get stats
        stats = cache.get_stats()
        print(f"✓ Cache stats: {stats}")
        
        # Test embedding config
        embed_config = {"model": "text-embedding-ada-002", "dimensions": 1536}
        cache.set_embedding_config("test_user_123", embed_config)
        retrieved_embed = cache.get_embedding_config("test_user_123")
        print(f"✓ Embedding config cached: {retrieved_embed}")
        
        # Final stats
        final_stats = cache.get_stats()
        print(f"\nFinal cache statistics: {final_stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_redis_cache()
    sys.exit(0 if success else 1)