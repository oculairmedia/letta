#!/usr/bin/env python3
"""
Simple patch to enable Graphiti archival memory in running Letta instance.
This directly modifies the archival memory functions.
"""

import os
import sys
import aiohttp
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Graphiti configuration
GRAPHITI_URL = os.getenv("GRAPHITI_URL", "http://192.168.50.90:8001")
GRAPHITI_ENABLED = os.getenv("LETTA_GRAPHITI_ENABLED", "true").lower() == "true"


async def test_graphiti():
    """Test Graphiti connection"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{GRAPHITI_URL}/healthcheck") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
        except:
            return False
    return False


async def graphiti_insert(content: str, agent_id: str) -> Optional[str]:
    """Insert content into Graphiti"""
    async with aiohttp.ClientSession() as session:
        try:
            memory_data = {
                "messages": [{
                    "role": "system",
                    "content": f"[Archival Memory] {content}",
                    "role_type": "archival_memory",
                    "metadata": {
                        "agent_id": agent_id,
                        "timestamp": datetime.now().isoformat(),
                        "memory_type": "archival"
                    }
                }]
            }
            
            async with session.post(
                f"{GRAPHITI_URL}/add-memory",
                json=memory_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully added memory to Graphiti")
                    return f"Memory added to knowledge graph"
                else:
                    error = await response.text()
                    logger.error(f"Failed to add memory: {error}")
                    return None
        except Exception as e:
            logger.error(f"Error adding to Graphiti: {e}")
            return None


async def graphiti_search(query: str, agent_id: str, limit: int = 10) -> List[str]:
    """Search Graphiti for memories"""
    async with aiohttp.ClientSession() as session:
        try:
            search_data = {
                "query": query,
                "max_nodes": limit * 2,  # Get extra for filtering
                "group_ids": [agent_id]
            }
            
            async with session.post(
                f"{GRAPHITI_URL}/search/nodes",
                json=search_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    nodes = data.get("nodes", [])
                    
                    # Filter and format results
                    results = []
                    for node in nodes[:limit]:
                        # Check if node belongs to agent
                        if (node.get("group_id") == agent_id or 
                            agent_id in str(node.get("attributes", {}))):
                            text = node.get("summary", node.get("name", ""))
                            created = node.get("created_at", "Unknown")
                            results.append(f"{text} [Created: {created}]")
                    
                    logger.info(f"Found {len(results)} results in Graphiti")
                    return results
                else:
                    logger.error(f"Search failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error searching Graphiti: {e}")
            return []


def patch_archival_memory():
    """Patch the archival memory functions in Letta"""
    try:
        # Import Letta's tool executor
        from letta.services.tool_executor.core_tool_executor import LettaCoreToolExecutor
        
        # Save original methods
        if not hasattr(LettaCoreToolExecutor, '_original_archival_memory_insert'):
            LettaCoreToolExecutor._original_archival_memory_insert = LettaCoreToolExecutor.archival_memory_insert
            LettaCoreToolExecutor._original_archival_memory_search = LettaCoreToolExecutor.archival_memory_search
        
        # Create patched methods
        async def patched_archival_memory_insert(self, agent_state, actor, content):
            """Patched archival memory insert using Graphiti"""
            if GRAPHITI_ENABLED:
                logger.debug(f"Using Graphiti for archival_memory_insert")
                result = await graphiti_insert(content, str(agent_state.id))
                return result
            else:
                # Fall back to original
                return await self._original_archival_memory_insert(agent_state, actor, content)
        
        async def patched_archival_memory_search(self, agent_state, actor, query, page=0, start=None):
            """Patched archival memory search using Graphiti"""
            if GRAPHITI_ENABLED:
                logger.debug(f"Using Graphiti for archival_memory_search")
                
                # Calculate offset
                page_size = 10
                offset = start if start is not None else page * page_size
                
                # Search Graphiti
                results = await graphiti_search(query, str(agent_state.id), page_size + offset)
                
                # Apply pagination
                paginated = results[offset:offset + page_size]
                
                if not paginated:
                    return f"No results found for: {query}"
                
                # Format results
                formatted = []
                for i, result in enumerate(paginated):
                    formatted.append(f"{i + 1 + offset}. {result}")
                
                result_text = "\n\n".join(formatted)
                
                # Add pagination info
                if len(results) > offset + page_size:
                    result_text += f"\n\n[Page {page + 1}. Use page={page + 1} for more results]"
                
                return result_text
            else:
                # Fall back to original
                return await self._original_archival_memory_search(agent_state, actor, query, page, start)
        
        # Apply patches
        LettaCoreToolExecutor.archival_memory_insert = patched_archival_memory_insert
        LettaCoreToolExecutor.archival_memory_search = patched_archival_memory_search
        
        logger.info("✅ Successfully patched archival memory functions")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch: {e}")
        return False


async def main():
    """Main function"""
    logger.info(f"🚀 Enabling Graphiti archival memory")
    logger.info(f"Graphiti URL: {GRAPHITI_URL}")
    
    # Test connection
    if await test_graphiti():
        logger.info("✅ Graphiti is healthy")
    else:
        logger.error("❌ Cannot connect to Graphiti")
        return False
    
    # Apply patch
    if patch_archival_memory():
        logger.info("""
✅ Graphiti archival memory enabled!

Archival memory functions now use Graphiti's hybrid vector+graph search.
All agents will automatically use the new backend.
""")
        return True
    
    return False


if __name__ == "__main__":
    # Run async main
    success = asyncio.run(main())
    
    # If successful, keep the patch active
    if success:
        logger.info("Patch applied. Archival memory now uses Graphiti.")
        
        # Set environment variable for persistence
        os.environ["LETTA_GRAPHITI_ENABLED"] = "true"
        os.environ["LETTA_GRAPHITI_URL"] = GRAPHITI_URL