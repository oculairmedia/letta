# graphiti_tool_executor.py
"""
Modified CoreToolExecutor that uses Graphiti for archival memory.
Inherits from CoreToolExecutor but overrides archival memory functions.
"""

from typing import Optional, List, Dict, Any
import logging

from letta.services.tool_executor.core_tool_executor import LettaCoreToolExecutor
from letta.services.graphiti_archival_memory import GraphitiArchivalMemory
from letta.schemas.agent import AgentState
from letta.schemas.user import User
from letta.constants import JSON_ENSURE_ASCII
import json

logger = logging.getLogger(__name__)


class GraphitiToolExecutor(LettaCoreToolExecutor):
    """
    Core tool executor with Graphiti-backed archival memory.
    Overrides archival_memory_insert and archival_memory_search.
    """
    
    def __init__(self, agent_manager, user_manager, passage_manager, 
                 graphiti_url: str = "http://192.168.50.90:8001"):
        super().__init__(agent_manager, user_manager, passage_manager)
        self.graphiti_memory = GraphitiArchivalMemory(graphiti_url)
        logger.info(f"Initialized GraphitiToolExecutor with URL: {graphiti_url}")
    
    async def archival_memory_insert(
        self,
        agent_state: AgentState,
        actor: User,
        content: str,
    ) -> Optional[str]:
        """
        Add content to archival memory using Graphiti knowledge graph.
        
        Args:
            agent_state: The agent state
            actor: The user performing the action
            content: Content to insert into archival memory
            
        Returns:
            Success message or None
        """
        logger.debug(f"GraphitiToolExecutor: Inserting to archival memory for agent {agent_state.id}")
        
        try:
            result = await self.graphiti_memory.insert(agent_state, actor, content)
            
            if result:
                # Update agent's memory statistics
                # Note: In a full implementation, you might want to track this differently
                logger.info(f"Successfully inserted to Graphiti archival memory")
                return result
            else:
                logger.error("Failed to insert to Graphiti archival memory")
                return None
                
        except Exception as e:
            logger.error(f"Error in archival_memory_insert: {str(e)}")
            return None
    
    async def archival_memory_search(
        self,
        agent_state: AgentState,
        actor: User,
        query: str,
        page: Optional[int] = 0,
        start: Optional[int] = None,
    ) -> Optional[str]:
        """
        Search archival memory using Graphiti's hybrid vector+graph search.
        
        Args:
            agent_state: The agent state
            actor: The user performing the action
            query: Search query  
            page: Page number (0-indexed)
            start: Starting index (alternative to page)
            
        Returns:
            Formatted search results or None
        """
        logger.debug(f"GraphitiToolExecutor: Searching archival memory for agent {agent_state.id}")
        
        try:
            # Get results from Graphiti
            results, total_count = await self.graphiti_memory.search(
                agent_state, actor, query, page, start
            )
            
            if not results:
                return f"No results found for query: {query}"
            
            # Format results for agent consumption
            formatted_results = []
            
            for i, result in enumerate(results):
                # Extract text and context
                text = result.get("text", "")
                context = result.get("context", "")
                created_at = result.get("created_at", "Unknown")
                
                # Format each result
                formatted_results.append(
                    f"{i + 1 + (page * 10)}. {text}\n"
                    f"   [Created: {created_at}]"
                )
                
                # Add context if available
                if context and context != "No additional context":
                    formatted_results.append(f"   [Related: {context}]")
            
            # Join all results
            result_text = "\n\n".join(formatted_results)
            
            # Add pagination info if needed
            if total_count > len(results) + (page * 10):
                total_pages = (total_count // 10) + (1 if total_count % 10 > 0 else 0)
                result_text += f"\n\n[Page {page + 1}/{total_pages}. Use page={page + 1} to see more results.]"
            
            logger.info(f"Returning {len(results)} search results from Graphiti")
            return result_text
            
        except Exception as e:
            logger.error(f"Error in archival_memory_search: {str(e)}")
            return f"Error searching archival memory: {str(e)}"
    
    async def close(self):
        """Clean up resources"""
        await self.graphiti_memory.close()


def create_graphiti_tool_executor_factory(graphiti_url: str = "http://192.168.50.90:8001"):
    """
    Factory function to create GraphitiToolExecutor instances.
    This can be used to replace the default CoreToolExecutor in dependency injection.
    """
    def factory(agent_manager, user_manager, passage_manager):
        return GraphitiToolExecutor(
            agent_manager, 
            user_manager, 
            passage_manager,
            graphiti_url
        )
    
    return factory