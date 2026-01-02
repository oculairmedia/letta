#!/usr/bin/env python3
"""
Script to modify Letta's core_tool_executor.py to use Graphiti for archival memory
"""

import re

# The new imports to add
NEW_IMPORTS = """
# Graphiti integration imports
import aiohttp
import os
from datetime import datetime

# Graphiti configuration
GRAPHITI_URL = os.getenv("GRAPHITI_URL", "http://192.168.50.90:8001")
GRAPHITI_ENABLED = os.getenv("LETTA_GRAPHITI_ENABLED", "true").lower() == "true"
"""

# New archival_memory_insert implementation
NEW_INSERT_FUNCTION = '''    async def archival_memory_insert(self, agent_state: AgentState, actor: User, content: str) -> Optional[str]:
        """
        Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

        Args:
            content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        if GRAPHITI_ENABLED:
            # Use Graphiti for storage
            try:
                async with aiohttp.ClientSession() as session:
                    memory_data = {
                        "messages": [{
                            "role": "system",
                            "content": f"[Archival Memory] {content}",
                            "role_type": "archival_memory",
                            "metadata": {
                                "agent_id": str(agent_state.id),
                                "agent_name": agent_state.name,
                                "user_id": str(actor.id),
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
                            logger.info(f"Successfully added memory to Graphiti for agent {agent_state.id}")
                        else:
                            logger.error(f"Failed to add to Graphiti: {response.status}")
                            # Fall back to original implementation
                            raise Exception("Graphiti storage failed")
                            
            except Exception as e:
                logger.error(f"Graphiti insert error: {e}, falling back to original")
                # Fall back to original implementation
                await PassageManager().insert_passage_async(
                    agent_state=agent_state,
                    agent_id=agent_state.id,
                    text=content,
                    actor=actor,
                )
        else:
            # Original implementation
            await PassageManager().insert_passage_async(
                agent_state=agent_state,
                agent_id=agent_state.id,
                text=content,
                actor=actor,
            )
            
        await AgentManager().rebuild_system_prompt_async(agent_id=agent_state.id, actor=actor, force=True)
        return None'''

# New archival_memory_search implementation
NEW_SEARCH_FUNCTION = '''    async def archival_memory_search(
        self, agent_state: AgentState, actor: User, query: str, page: Optional[int] = 0, start: Optional[int] = 0
    ) -> Optional[str]:
        """
        Search archival memory using semantic (embedding-based) search.

        Args:
            query (str): String to search for.
            page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).
            start (Optional[int]): Starting index for the search results. Defaults to 0.

        Returns:
            str: Query result string
        """
        if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
            page = 0
        try:
            page = int(page)
        except:
            raise ValueError(f"'page' argument must be an integer")

        count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE

        if GRAPHITI_ENABLED:
            # Use Graphiti for search
            try:
                async with aiohttp.ClientSession() as session:
                    search_data = {
                        "query": query,
                        "max_nodes": count * 2,  # Get extra for filtering
                        "group_ids": [str(agent_state.id)]
                    }
                    
                    async with session.post(
                        f"{GRAPHITI_URL}/search/nodes",
                        json=search_data,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            nodes = result.get("nodes", [])
                            
                            # Filter to this agent's memories
                            agent_memories = []
                            for node in nodes:
                                if (str(agent_state.id) in str(node.get("group_id", "")) or 
                                    str(agent_state.id) in str(node.get("attributes", {}))):
                                    text = node.get("summary", node.get("name", ""))
                                    created = node.get("created_at", "Unknown")
                                    agent_memories.append(f"{text}")
                            
                            # Apply pagination
                            offset = start if start is not None else page * count
                            paginated = agent_memories[offset:offset + count]
                            
                            if len(paginated) == 0:
                                return None
                            
                            results = [f"{i+1+offset}. {r}" for i, r in enumerate(paginated)]
                            results_str = "\\n".join(results)
                            
                            if len(agent_memories) > offset + count:
                                results_str += f"\\n\\n[Page {page + 1}. Use page={page + 1} for more results.]"
                            
                            return results_str
                        else:
                            logger.error(f"Graphiti search failed: {response.status}")
                            # Fall back to original
                            raise Exception("Graphiti search failed")
                            
            except Exception as e:
                logger.error(f"Graphiti search error: {e}, falling back to original")
                # Fall through to original implementation
        
        # Original implementation (also used as fallback)
        try:
            # Get results using passage manager
            all_results = await AgentManager().list_agent_passages_async(
                actor=actor,
                agent_id=agent_state.id,
                query_text=query,
                limit=count + start,  # Request enough results to handle offset
                embedding_config=agent_state.embedding_config,'''

def modify_file(input_file, output_file):
    """Modify the core_tool_executor.py file to use Graphiti"""
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Add imports after the existing imports
    import_pos = content.find('from typing import')
    if import_pos > 0:
        # Find the end of imports section
        import_end = content.find('\n\n', import_pos)
        content = content[:import_end] + '\n' + NEW_IMPORTS + content[import_end:]
    
    # Replace archival_memory_insert
    insert_pattern = r'(    async def archival_memory_insert\(self.*?\n(?:.*?\n)*?        return None)'
    content = re.sub(insert_pattern, NEW_INSERT_FUNCTION, content, flags=re.DOTALL)
    
    # Replace archival_memory_search (up to the try block)
    search_pattern = r'(    async def archival_memory_search\(.*?\n(?:.*?\n)*?        try:\n            # Get results using passage manager)'
    content = re.sub(search_pattern, NEW_SEARCH_FUNCTION, content, flags=re.DOTALL)
    
    # Write modified content
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Modified file written to: {output_file}")

if __name__ == "__main__":
    modify_file("/tmp/core_tool_executor.py", "/tmp/core_tool_executor_modified.py")