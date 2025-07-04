#!/usr/bin/env python3
"""
Apply Redis caching patch to Letta server.py
This script modifies the server to use Redis caching when available.
"""

import os
import shutil
from datetime import datetime

SERVER_PATH = "/app/letta/server/server.py"
BACKUP_PATH = f"/app/letta/server/server.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def apply_redis_patch():
    """Apply Redis caching modifications to server.py"""
    
    # Create backup
    if os.path.exists(SERVER_PATH):
        shutil.copy2(SERVER_PATH, BACKUP_PATH)
        print(f"Created backup: {BACKUP_PATH}")
    
    # Read the original file
    with open(SERVER_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "from letta.cache.redis_cache import LettaConfigCache" in content:
        print("Redis patch already applied!")
        return
    
    # Add import
    import_line = "from letta.schemas.embedding_config import EmbeddingConfig"
    if import_line in content:
        content = content.replace(
            import_line,
            import_line + "\ntry:\n    from letta.cache.redis_cache import LettaConfigCache\n    REDIS_AVAILABLE = True\nexcept ImportError:\n    REDIS_AVAILABLE = False"
        )
    
    # Modify cache initialization
    old_cache_init = """        # TODO: Remove these in memory caches
        self._llm_config_cache = {}
        self._embedding_config_cache = {}"""
    
    new_cache_init = """        # TODO: Remove these in memory caches
        # Use Redis cache if available
        redis_url = os.getenv('LETTA_REDIS_URL', os.getenv('REDIS_URL'))
        if REDIS_AVAILABLE and redis_url:
            try:
                self._config_cache = LettaConfigCache(redis_url=redis_url)
                print(f"[Redis Cache] Initialized with URL: {redis_url}")
            except Exception as e:
                print(f"[Redis Cache] Failed to initialize: {e}")
                self._llm_config_cache = {}
                self._embedding_config_cache = {}
        else:
            self._llm_config_cache = {}
            self._embedding_config_cache = {}"""
    
    content = content.replace(old_cache_init, new_cache_init)
    
    # Modify get_cached_llm_config
    old_llm_method = """    @trace_method
    def get_cached_llm_config(self, actor: User, **kwargs):
        key = make_key(**kwargs)
        if key not in self._llm_config_cache:
            self._llm_config_cache[key] = self.get_llm_config_from_handle(actor=actor, **kwargs)
        return self._llm_config_cache[key]"""
    
    new_llm_method = """    @trace_method
    def get_cached_llm_config(self, actor: User, **kwargs):
        if hasattr(self, '_config_cache'):
            config = self._config_cache.get_llm_config(actor.id, **kwargs)
            if config is None:
                config = self.get_llm_config_from_handle(actor=actor, **kwargs)
                self._config_cache.set_llm_config(actor.id, config, **kwargs)
            return config
        else:
            key = make_key(**kwargs)
            if key not in self._llm_config_cache:
                self._llm_config_cache[key] = self.get_llm_config_from_handle(actor=actor, **kwargs)
            return self._llm_config_cache[key]"""
    
    content = content.replace(old_llm_method, new_llm_method)
    
    # Modify get_cached_embedding_config
    old_embed_method = """    @trace_method
    def get_cached_embedding_config(self, actor: User, **kwargs):
        key = make_key(**kwargs)
        if key not in self._embedding_config_cache:
            self._embedding_config_cache[key] = self.get_embedding_config_from_handle(actor=actor, **kwargs)
        return self._embedding_config_cache[key]"""
    
    new_embed_method = """    @trace_method
    def get_cached_embedding_config(self, actor: User, **kwargs):
        if hasattr(self, '_config_cache'):
            config = self._config_cache.get_embedding_config(actor.id, **kwargs)
            if config is None:
                config = self.get_embedding_config_from_handle(actor=actor, **kwargs)
                self._config_cache.set_embedding_config(actor.id, config, **kwargs)
            return config
        else:
            key = make_key(**kwargs)
            if key not in self._embedding_config_cache:
                self._embedding_config_cache[key] = self.get_embedding_config_from_handle(actor=actor, **kwargs)
            return self._embedding_config_cache[key]"""
    
    content = content.replace(old_embed_method, new_embed_method)
    
    # Write the patched file
    with open(SERVER_PATH, 'w') as f:
        f.write(content)
    
    print("Redis patch applied successfully!")
    print("The server will now use Redis caching when LETTA_REDIS_URL is set.")

if __name__ == "__main__":
    apply_redis_patch()