# graphiti_archival_memory.py
"""
Graphiti-backed archival memory implementation for Letta.
Replaces vector-only search with Graphiti's hybrid vector+graph search.
"""

import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging

from letta.schemas.agent import AgentState
from letta.schemas.user import User
from letta.schemas.memory import ArchivalMemorySummary
from letta.orm.passage import AgentPassage
from letta.services.tool_executor.base import Tool
from letta.schemas.tool import ToolCreate
from letta.constants import JSON_ENSURE_ASCII

logger = logging.getLogger(__name__)


class GraphitiArchivalMemory:
    """Graphiti-backed archival memory implementation"""
    
    def __init__(self, graphiti_url: str = "http://192.168.50.90:8001"):
        self.graphiti_url = graphiti_url
        self.session = None
        self._ensure_session()
    
    def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def insert(self, agent_state: AgentState, actor: User, content: str) -> Optional[str]:
        """
        Insert content into Graphiti graph as archival memory.
        
        Args:
            agent_state: Current agent state
            actor: User performing the action
            content: Text to store in memory
            
        Returns:
            Success message or None if failed
        """
        self._ensure_session()
        
        try:
            # Prepare memory data for Graphiti
            memory_data = {
                "messages": [{
                    "role": "system",
                    "content": f"[Archival Memory] {content}",
                    "role_type": "archival_memory",
                    "metadata": {
                        "agent_id": agent_state.id,
                        "agent_name": agent_state.name,
                        "user_id": actor.id,
                        "timestamp": datetime.now().isoformat(),
                        "memory_type": "archival"
                    }
                }]
            }
            
            # Add to Graphiti via add-memory endpoint
            async with self.session.post(
                f"{self.graphiti_url}/add-memory",
                json=memory_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully added memory to Graphiti for agent {agent_state.id}")
                    return f"Memory added successfully to knowledge graph"
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to add memory to Graphiti: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error adding memory to Graphiti: {str(e)}")
            return None
    
    async def search(self, agent_state: AgentState, actor: User, query: str, 
                     page: Optional[int] = 0, start: Optional[int] = 0) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search Graphiti graph for relevant memories.
        
        Args:
            agent_state: Current agent state
            actor: User performing the action
            query: Search query
            page: Page number for pagination (0-based)
            start: Starting index (alternative to page)
            
        Returns:
            Tuple of (results, total_count)
        """
        self._ensure_session()
        
        try:
            # Calculate pagination
            page_size = 10  # Match Letta's default
            if start is not None:
                offset = start
            else:
                offset = page * page_size
            
            # Use Graphiti's hybrid search (vector + graph)
            search_data = {
                "query": query,
                "max_nodes": page_size + offset,  # Get enough to paginate
                "group_ids": [agent_state.id],  # Filter by agent
                "entity": ""  # Search all entity types
            }
            
            async with self.session.post(
                f"{self.graphiti_url}/search/nodes",
                json=search_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    nodes = result.get("nodes", [])
                    
                    # Filter to this agent's memories
                    agent_memories = []
                    for node in nodes:
                        # Check if this node belongs to the agent
                        if (node.get("group_id") == agent_state.id or 
                            node.get("attributes", {}).get("agent_id") == agent_state.id or
                            agent_state.id in str(node.get("attributes", {}))):
                            
                            # Format as expected by Letta
                            memory_entry = {
                                "id": node.get("uuid"),
                                "text": node.get("summary", node.get("name", "")),
                                "created_at": node.get("created_at"),
                                "metadata": {
                                    "node_type": node.get("labels", ["Unknown"])[0],
                                    "attributes": node.get("attributes", {}),
                                    "graphiti_uuid": node.get("uuid")
                                }
                            }
                            agent_memories.append(memory_entry)
                    
                    # Apply pagination
                    total_count = len(agent_memories)
                    paginated_results = agent_memories[offset:offset + page_size]
                    
                    # Get additional context for each result
                    enriched_results = []
                    for memory in paginated_results:
                        # Get connected nodes for context
                        context = await self._get_memory_context(memory["id"])
                        memory["context"] = context
                        enriched_results.append(memory)
                    
                    logger.info(f"Graphiti search returned {len(enriched_results)} results for agent {agent_state.id}")
                    return enriched_results, total_count
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to search Graphiti: {response.status} - {error_text}")
                    return [], 0
                    
        except Exception as e:
            logger.error(f"Error searching Graphiti: {str(e)}")
            return [], 0
    
    async def _get_memory_context(self, node_id: str) -> str:
        """Get related entities and relationships for context"""
        try:
            async with self.session.get(
                f"{self.graphiti_url}/edges/by-node/{node_id}",
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Build context string from relationships
                    context_parts = []
                    
                    # Add source edges (this node points to others)
                    for edge in result.get("source_edges", [])[:3]:
                        context_parts.append(f"{edge.get('fact', 'Related to something')}")
                    
                    # Add target edges (others point to this node)
                    for edge in result.get("target_edges", [])[:2]:
                        context_parts.append(f"Referenced by: {edge.get('fact', 'something')}")
                    
                    return " | ".join(context_parts) if context_parts else "No additional context"
                else:
                    return "Context unavailable"
                    
        except Exception as e:
            logger.error(f"Error getting memory context: {str(e)}")
            return "Error retrieving context"
    
    async def get_summary(self, agent_state: AgentState) -> ArchivalMemorySummary:
        """Get summary statistics for archival memory"""
        try:
            # Query for count of agent's memories
            search_data = {
                "query": "",  # Empty query to get all
                "max_nodes": 1,  # We just need the count
                "group_ids": [agent_state.id]
            }
            
            async with self.session.post(
                f"{self.graphiti_url}/search/nodes",
                json=search_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    # For now, estimate size based on search
                    # In future, could add a count endpoint to Graphiti
                    return ArchivalMemorySummary(size=100)  # Placeholder
                else:
                    return ArchivalMemorySummary(size=0)
                    
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return ArchivalMemorySummary(size=0)


# Create tool definitions that use Graphiti
def create_graphiti_archival_tools(graphiti_memory: GraphitiArchivalMemory) -> List[Tool]:
    """Create the archival memory tools that agents can use"""
    
    async def archival_memory_insert(
        agent_state: AgentState,
        actor: User,
        content: str,
    ) -> Optional[str]:
        """
        Add content to archival memory (powered by Graphiti knowledge graph).
        
        Args:
            content: Text to write to archival memory
            
        Returns:
            Success message or None
        """
        return await graphiti_memory.insert(agent_state, actor, content)
    
    async def archival_memory_search(
        agent_state: AgentState,
        actor: User,
        query: str,
        page: Optional[int] = 0,
    ) -> Optional[str]:
        """
        Search archival memory using Graphiti's hybrid vector+graph search.
        
        Args:
            query: Search query
            page: Page number (0-indexed)
            
        Returns:
            Formatted search results
        """
        results, total = await graphiti_memory.search(agent_state, actor, query, page)
        
        if not results:
            return f"No results found for query: {query}"
        
        # Format results for agent
        formatted_results = []
        for i, result in enumerate(results):
            context = result.get("context", "")
            formatted_results.append(
                f"{i + 1 + page * 10}. {result['text']}\n"
                f"   [Created: {result.get('created_at', 'Unknown')}]\n"
                f"   [Context: {context}]"
            )
        
        result_text = "\n\n".join(formatted_results)
        
        # Add pagination info
        if total > len(results) + page * 10:
            result_text += f"\n\n[Page {page + 1}/{(total // 10) + 1}. Use page parameter to see more.]"
        
        return result_text
    
    # Create tool objects
    tools = [
        Tool(
            name="archival_memory_insert",
            description="Add content to archival memory (knowledge graph)",
            func=archival_memory_insert,
            parameters=[
                {"name": "content", "type": "string", "required": True, 
                 "description": "Text to write to archival memory"}
            ]
        ),
        Tool(
            name="archival_memory_search", 
            description="Search archival memory using vector similarity and graph relationships",
            func=archival_memory_search,
            parameters=[
                {"name": "query", "type": "string", "required": True,
                 "description": "Search query"},
                {"name": "page", "type": "integer", "required": False,
                 "description": "Page number (0-indexed)", "default": 0}
            ]
        )
    ]
    
    return tools