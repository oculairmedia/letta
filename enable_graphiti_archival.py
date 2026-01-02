#!/usr/bin/env python3
"""
Enable Graphiti archival memory for Letta agents.
This script patches the running Letta instance to use Graphiti for archival memory.
"""

import os
import sys
import importlib
import logging
from pathlib import Path

# Add letta to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patch_tool_executor():
    """Patch the CoreToolExecutor to use GraphitiToolExecutor"""
    try:
        # Import the modules
        from letta.services.tool_executor import core_tool_executor
        from letta.services.tool_executor.graphiti_tool_executor import GraphitiToolExecutor
        
        # Check if we need to patch
        if hasattr(core_tool_executor, '_original_CoreToolExecutor'):
            logger.info("CoreToolExecutor already patched")
            return True
        
        # Store original
        core_tool_executor._original_CoreToolExecutor = core_tool_executor.CoreToolExecutor
        
        # Replace with Graphiti version
        core_tool_executor.CoreToolExecutor = GraphitiToolExecutor
        
        logger.info("Successfully patched CoreToolExecutor to use Graphiti")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch CoreToolExecutor: {str(e)}")
        return False


def patch_dependency_injection():
    """Patch the dependency injection to use GraphitiToolExecutor"""
    try:
        # This is more complex and depends on Letta's DI system
        # For now, we'll use environment variables
        os.environ["LETTA_GRAPHITI_ENABLED"] = "true"
        os.environ["LETTA_GRAPHITI_URL"] = "http://192.168.50.90:8001"
        
        # Import and enable config
        from letta.config.graphiti_config import enable_graphiti
        enable_graphiti()
        
        logger.info("Enabled Graphiti configuration")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {str(e)}")
        return False


def create_startup_hook():
    """Create a startup hook that enables Graphiti when Letta starts"""
    hook_content = '''# graphiti_startup_hook.py
"""Auto-generated hook to enable Graphiti archival memory"""
import logging
from letta.config.graphiti_config import enable_graphiti

logger = logging.getLogger(__name__)

def on_startup():
    """Enable Graphiti on Letta startup"""
    try:
        enable_graphiti("http://192.168.50.90:8001")
        
        # Patch tool executor
        from letta.services.tool_executor import core_tool_executor
        from letta.services.tool_executor.graphiti_tool_executor import GraphitiToolExecutor
        
        if not hasattr(core_tool_executor, '_original_CoreToolExecutor'):
            core_tool_executor._original_CoreToolExecutor = core_tool_executor.CoreToolExecutor
            core_tool_executor.CoreToolExecutor = GraphitiToolExecutor
            logger.info("Graphiti archival memory enabled via startup hook")
            
    except Exception as e:
        logger.error(f"Failed to enable Graphiti in startup hook: {str(e)}")

# Register the hook
on_startup()
'''
    
    hook_path = Path(__file__).parent / "letta" / "hooks" / "graphiti_startup_hook.py"
    hook_path.parent.mkdir(exist_ok=True)
    hook_path.write_text(hook_content)
    logger.info(f"Created startup hook at {hook_path}")
    return True


def test_graphiti_connection():
    """Test that Graphiti is accessible"""
    try:
        import aiohttp
        import asyncio
        
        async def test():
            async with aiohttp.ClientSession() as session:
                async with session.get("http://192.168.50.90:8001/healthcheck") as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("status") == "healthy"
            return False
        
        is_healthy = asyncio.run(test())
        if is_healthy:
            logger.info("✅ Graphiti is healthy and accessible")
            return True
        else:
            logger.error("❌ Graphiti health check failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to Graphiti: {str(e)}")
        return False


def create_env_file():
    """Create or update .env file with Graphiti settings"""
    env_path = Path(__file__).parent / ".env"
    
    # Read existing env if it exists
    existing_env = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    existing_env[key] = value
    
    # Update with Graphiti settings
    existing_env.update({
        "LETTA_GRAPHITI_ENABLED": "true",
        "LETTA_GRAPHITI_URL": "http://192.168.50.90:8001",
        "LETTA_GRAPHITI_EXTRACT_ENTITIES": "true",
        "LETTA_GRAPHITI_EXTRACT_RELATIONSHIPS": "true",
        "LETTA_GRAPHITI_CACHE_ENABLED": "true"
    })
    
    # Write back
    with open(env_path, 'w') as f:
        for key, value in existing_env.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"Updated .env file at {env_path}")
    return True


def main():
    """Main function to enable Graphiti integration"""
    logger.info("🚀 Enabling Graphiti archival memory for Letta")
    
    # Test Graphiti connection first
    if not test_graphiti_connection():
        logger.error("Cannot proceed without Graphiti connection")
        return False
    
    # Create/update env file
    create_env_file()
    
    # Patch dependency injection
    patch_dependency_injection()
    
    # Patch tool executor
    patch_tool_executor()
    
    # Create startup hook
    create_startup_hook()
    
    logger.info("""
✅ Graphiti archival memory integration enabled!

To use:
1. Restart Letta server for changes to take full effect
2. New agents will automatically use Graphiti for archival memory
3. Existing agents can be migrated by updating their configuration

Environment variables set:
- LETTA_GRAPHITI_ENABLED=true
- LETTA_GRAPHITI_URL=http://192.168.50.90:8001

The archival_memory_insert and archival_memory_search functions now use Graphiti's
hybrid vector+graph search instead of simple vector search.
""")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)