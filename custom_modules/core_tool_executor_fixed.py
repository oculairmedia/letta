import math
import json
import aiohttp
from datetime import datetime
from typing import Any, Dict, Optional
import logging
import re

from letta.constants import (
    CORE_MEMORY_LINE_NUMBER_WARNING,
    MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX,
    READ_ONLY_BLOCK_EDIT_ERROR,
    RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE,
)
from letta.helpers.json_helpers import json_dumps
from letta.schemas.agent import AgentState
from letta.schemas.sandbox_config import SandboxConfig
from letta.schemas.tool import Tool
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.schemas.user import User
from letta.services.agent_manager import AgentManager
from letta.services.message_manager import MessageManager
from letta.services.passage_manager import PassageManager
from letta.services.tool_executor.tool_executor_base import ToolExecutor
from letta.utils import get_friendly_error_msg

logger = logging.getLogger(__name__)

# Graphiti configuration
GRAPHITI_ENABLED = True
GRAPHITI_URL = "http://192.168.50.90:8001"
GRAPHITI_CENTRALITY_URL = "http://192.168.50.90:8002"

def parse_temporal_query(query):
    """Parse temporal components from query."""
    temporal_params = {}
    
    # Remove temporal keywords and extract parameters
    patterns = [
        (r'\bafter\s+(\d{4}-\d{2}-\d{2})', 'start_date'),
        (r'\bbefore\s+(\d{4}-\d{2}-\d{2})', 'end_date'),
        (r'\bon\s+(\d{4}-\d{2}-\d{2})', 'specific_date'),
        (r'\bsince\s+(\d{4}-\d{2}-\d{2})', 'start_date'),
        (r'\buntil\s+(\d{4}-\d{2}-\d{2})', 'end_date'),
        (r'\blast\s+(\d+)\s+days?', 'days_back'),
        (r'\bpast\s+(\d+)\s+days?', 'days_back'),
        (r'\brecent\b', 'recent'),
        (r'\blatest\b', 'recent'),
    ]
    
    cleaned_query = query.lower()
    
    for pattern, param_name in patterns:
        match = re.search(pattern, cleaned_query)
        if match:
            if param_name in ['start_date', 'end_date', 'specific_date']:
                temporal_params[param_name] = match.group(1)
            elif param_name == 'days_back':
                temporal_params[param_name] = int(match.group(1))
            elif param_name == 'recent':
                temporal_params['days_back'] = 7  # Default to last 7 days
            
            # Remove the temporal phrase from query
            cleaned_query = re.sub(pattern, '', cleaned_query).strip()
    
    # Clean up multiple spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    
    return cleaned_query if cleaned_query else query, temporal_params


class LettaCoreToolExecutor(ToolExecutor):
    """Executor for LETTA core tools with direct implementation of functions."""

    async def execute(
        self,
        function_name: str,
        function_args: dict,
        tool: Tool,
        actor: User,
        agent_state: Optional[AgentState] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        sandbox_env_vars: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        # Map function names to method calls
        assert agent_state is not None, "Agent state is required for core tools"
        function_map = {
            "send_message": self.send_message,
            "conversation_search": self.conversation_search,
            "archival_memory_search": self.archival_memory_search,
            "archival_memory_insert": self.archival_memory_insert,
            "core_memory_append": self.core_memory_append,
            "core_memory_replace": self.core_memory_replace,
            "memory_replace": self.memory_replace,
            "memory_insert": self.memory_insert,
            "memory_rethink": self.memory_rethink,
            "memory_finish_edits": self.memory_finish_edits,
        }

        if function_name not in function_map:
            return ToolExecutionResult(
                status="error",
                func_return=f"Unknown function: {function_name}",
                agent_state=agent_state,
            )

        try:
            # Call the appropriate method
            method = function_map[function_name]
            result = await method(agent_state, actor, **function_args)
            return ToolExecutionResult(
                status="success",
                func_return=result,
                agent_state=agent_state,
            )
        except Exception as e:
            return ToolExecutionResult(
                status="error",
                func_return=get_friendly_error_msg(function_name, e.__class__.__name__, str(e)),
                agent_state=agent_state,
            )

    async def send_message(self, agent_state: AgentState, actor: User, message: str) -> str:
        """
        Sends a message to the human user.

        Args:
            message (str): Message contents. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        return message

    async def conversation_search(
        self, agent_state: AgentState, actor: User, query: str, page: Optional[int] = 0, start: Optional[int] = 0
    ) -> Optional[str]:
        """
        Search prior conversation history using semantic (embedding-based) search.

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

        try:
            # Get results using message manager
            all_results = await MessageManager().list_messages_for_agent_async(
                actor=actor,
                agent_id=agent_state.id,
                query_text=query,
                limit=count + start,  # Request enough results to handle offset
            )

            # Apply pagination
            end = min(count + start, len(all_results))
            paged_results = all_results[start:end]

            # Format results
            results_formatted = []
            for message in paged_results:
                if message.content and len(message.content) > 0:
                    # Find the first text content
                    for content_item in message.content:
                        if hasattr(content_item, 'text'):
                            results_formatted.append(content_item.text)
                            break
                    else:
                        # No text content found, try to get string representation
                        results_formatted.append(str(message.content[0]) if message.content else "")
                else:
                    results_formatted.append("")
            
            results_pref = f"Your search returned {len(paged_results)} results:"
            return f"{results_pref} {json_dumps(results_formatted)}"

        except Exception as e:
            raise e

    async def archival_memory_search(
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
        import sys
        print(f"[ARCHIVAL_SEARCH] Called with query={query}, page={page}, start={start}", file=sys.stderr, flush=True)
        
        if page is None or (isinstance(page, str) and page.lower().strip() == "none"):
            page = 0
        try:
            page = int(page)
        except:
            raise ValueError(f"'page' argument must be an integer")

        count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE

        print(f"[GRAPHITI DEBUG] Checking if enabled: GRAPHITI_ENABLED={GRAPHITI_ENABLED}", flush=True)
        # Parse temporal components from query
        parsed_query, temporal_params = parse_temporal_query(query)
        if temporal_params:
            print(f"[TEMPORAL] Detected temporal query: {temporal_params}", file=sys.stderr, flush=True)
            query = parsed_query  # Use the cleaned query

        if GRAPHITI_ENABLED:
            print(f"[ARCHIVAL_SEARCH] Graphiti enabled, searching...", file=sys.stderr, flush=True)
            # Use Graphiti for search
            try:
                async with aiohttp.ClientSession() as session:
                    search_data = {
                        "query": query,
                        "max_nodes": count * 2,  # Get extra for filtering
                        **temporal_params  # Add temporal parameters
                    }
                    print(f"[GRAPHITI DEBUG] Sending request to {GRAPHITI_URL}/search/nodes with data: {search_data}", file=sys.stderr, flush=True)
                    
                    async with session.post(
                        f"{GRAPHITI_URL}/search/nodes",
                        json=search_data,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            print(f"[GRAPHITI DEBUG] Response received: {len(response_data.get('nodes', []))} nodes", file=sys.stderr, flush=True)
                            
                            # Get centrality scores from the centrality service
                            centrality_scores = {}
                            try:
                                async with session.post(
                                    f"{GRAPHITI_CENTRALITY_URL}/centrality/all",
                                    json={"store_results": False},  # Don't store, just calculate
                                    headers={"Content-Type": "application/json"}
                                ) as cent_response:
                                    if cent_response.status == 200:
                                        centrality_data = await cent_response.json()
                                        # Combine all centrality scores
                                        for metric, scores in centrality_data.items():
                                            for node_id, score in scores.items():
                                                if node_id not in centrality_scores:
                                                    centrality_scores[node_id] = {}
                                                centrality_scores[node_id][metric] = score
                                        print(f"[CENTRALITY] Got centrality scores for {len(centrality_scores)} nodes", file=sys.stderr, flush=True)
                            except Exception as e:
                                print(f"[CENTRALITY] Failed to get centrality scores: {e}", file=sys.stderr, flush=True)
                            
                            # Process search results
                            nodes = response_data.get('nodes', [])
                            if not nodes:
                                print(f"[GRAPHITI DEBUG] No nodes found in response", file=sys.stderr, flush=True)
                                return "No results found."
                            
                            # Calculate combined centrality score and apply pagination
                            node_scores = []
                            for node in nodes:
                                node_uuid = node.get('uuid', '')
                                combined_score = 0
                                importance_level = ""
                                
                                if node_uuid in centrality_scores:
                                    scores = centrality_scores[node_uuid]
                                    # Weighted combination of centrality metrics
                                    pagerank = scores.get('pagerank', 0)
                                    degree = scores.get('degree', 0)
                                    betweenness = scores.get('betweenness', 0)
                                    
                                    # Normalize and combine (weights: PageRank=0.5, Degree=0.3, Betweenness=0.2)
                                    combined_score = (pagerank * 0.5) + (degree * 0.3) + (betweenness * 0.2)
                                    
                                    # Determine importance level
                                    if combined_score > 0.1:
                                        importance_level = "★★★ Highly Important"
                                    elif combined_score > 0.05:
                                        importance_level = "★★ Important"
                                    elif combined_score > 0.01:
                                        importance_level = "★ Relevant"
                                    else:
                                        importance_level = "• Standard"
                                
                                node_scores.append({
                                    'node': node,
                                    'score': combined_score,
                                    'importance': importance_level
                                })
                            
                            # Sort by combined centrality score (descending)
                            node_scores.sort(key=lambda x: x['score'], reverse=True)
                            
                            # Apply pagination
                            end = min(count + start, len(node_scores))
                            paged_results = node_scores[start:end]
                            
                            # Format results with importance indicators
                            results_formatted = []
                            for item in paged_results:
                                node = item['node']
                                importance = item['importance']
                                
                                # Use summary for rich content, fallback to name
                                content = node.get('summary', node.get('content', node.get('name', 'Unknown')))
                                name = node.get('name', 'Unknown')
                                created_at = node.get('created_at', '')
                                
                                # Create formatted entry with metadata
                                if created_at:
                                    timestamp = created_at.split('T')[0] if 'T' in created_at else created_at
                                    formatted_content = f"[{importance}] {name} ({timestamp}): {content}"
                                else:
                                    formatted_content = f"[{importance}] {name}: {content}"
                                
                                results_formatted.append(formatted_content)
                            
                            print(f"[GRAPHITI DEBUG] Formatted {len(results_formatted)} results", file=sys.stderr, flush=True)
                            
                            results_pref = f"Your search returned {len(paged_results)} results (ranked by importance):"
                            return f"{results_pref} {json_dumps(results_formatted)}"
                        else:
                            print(f"[GRAPHITI DEBUG] Request failed with status {response.status}", file=sys.stderr, flush=True)
                            # Fall back to original implementation
                            raise Exception("Graphiti search failed")
                            
            except Exception as e:
                print(f"[GRAPHITI DEBUG] Error in Graphiti search: {e}", file=sys.stderr, flush=True)
                # Fall back to original implementation
                pass

        # Original implementation as fallback
        try:
            # Get results using passage manager
            all_results = await AgentManager().list_agent_passages_async(
                actor=actor,
                agent_id=agent_state.id,
                query_text=query,
                limit=count + start,  # Request enough results to handle offset
            )

            # Apply pagination
            end = min(count + start, len(all_results))
            paged_results = all_results[start:end]

            # Format results
            results_formatted = []
            for passage in paged_results:
                if passage.text and len(passage.text) > 0:
                    results_formatted.append(passage.text)
                else:
                    results_formatted.append("")
            
            results_pref = f"Your search returned {len(paged_results)} results:"
            return f"{results_pref} {json_dumps(results_formatted)}" 

        except Exception as e:
            raise e

    async def archival_memory_insert(self, agent_state: AgentState, actor: User, content: str) -> Optional[str]:
        """
        Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

        Args:
            content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        if GRAPHITI_ENABLED:
            print(f"[ARCHIVAL_INSERT] Graphiti enabled, inserting...", file=sys.stderr, flush=True)
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
                            return None
                        else:
                            logger.error(f"Failed to add to Graphiti: {response.status}")
                            # Fall back to original implementation
                            raise Exception("Graphiti storage failed")
                            
            except Exception as e:
                logger.error(f"Error in Graphiti insert: {e}")
                # Fall back to original implementation
                pass

        # Original implementation as fallback
        try:
            await PassageManager().create_passage_async(
                actor=actor,
                agent_id=agent_state.id,
                text=content,
            )
            return None

        except Exception as e:
            raise e

    async def core_memory_append(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
        content: str,
    ) -> Optional[str]:
        """
        Append to the contents of core memory.

        Args:
            label (str): Section of the memory to be edited (persona or human).
            content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")
        current_value = str(agent_state.memory.get_block(label).value)
        new_value = current_value + "\n" + str(content)
        agent_state.memory.update_block_value(label=label, value=new_value)
        await AgentManager().update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
        return None

    async def core_memory_replace(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
        old_content: str,
        new_content: str,
    ) -> Optional[str]:
        """
        Replace the contents of core memory. To delete memories, use an empty string for new_content.

        Args:
            label (str): Section of the memory to be edited (persona or human).
            old_content (str): String to replace. Must be an exact match.
            new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")
        current_value = str(agent_state.memory.get_block(label).value)
        if old_content not in current_value:
            raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
        new_value = current_value.replace(str(old_content), str(new_content))
        agent_state.memory.update_block_value(label=label, value=new_value)
        await AgentManager().update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
        return None

    async def memory_replace(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
        old_str: str,
        new_str: Optional[str] = None,
    ) -> str:
        """
        The memory_replace command allows you to replace a specific string in a memory
        block with a new string. This is used for making precise edits.

        Args:
            label (str): Section of the memory to be edited, identified by its label.
            old_str (str): The text to replace (must match exactly, including whitespace
                and indentation). Do not include line number prefixes.
            new_str (Optional[str]): The new text to insert in place of the old text.
                Omit this argument to delete the old_str. Do not include line number prefixes.

        Returns:
            str: The success message
        """

        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(old_str)):
            raise ValueError(
                "old_str contains a line number prefix, which is not allowed. "
                "Do not include line numbers when calling memory tools (line "
                "numbers are for display purposes only)."
            )
        if CORE_MEMORY_LINE_NUMBER_WARNING in old_str:
            raise ValueError(
                "old_str contains a line number warning, which is not allowed. "
                "Do not include line number information when calling memory tools "
                "(line numbers are for display purposes only)."
            )
        if bool(MEMORY_TOOLS_LINE_NUMBER_PREFIX_REGEX.search(new_str)):
            raise ValueError(
                "new_str contains a line number prefix, which is not allowed. "
                "Do not include line numbers when calling memory tools (line "
                "numbers are for display purposes only)."
            )

        old_str = str(old_str).expandtabs()
        new_str = str(new_str).expandtabs()
        current_value = str(agent_state.memory.get_block(label).value).expandtabs()

        # Check if old_str is unique in the block
        occurences = current_value.count(old_str)
        if occurences == 0:
            raise ValueError(
                f"No replacement was performed, old_str `{old_str}` did not appear " f"verbatim in memory block with label `{label}`."
            )
        elif occurences > 1:
            content_value_lines = current_value.split("\n")
            lines = [idx + 1 for idx, line in enumerate(content_value_lines) if old_str in line]
            raise ValueError(
                f"No replacement was performed. Multiple occurrences of "
                f"old_str `{old_str}` in lines {lines}. Please ensure it is unique."
            )

        # Replace old_str with new_str
        new_value = current_value.replace(str(old_str), str(new_str))
        agent_state.memory.update_block_value(label=label, value=new_value)
        await AgentManager().update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
        return f"OK, replaced `{old_str}` with `{new_str}` in memory block with label `{label}`."

    async def memory_insert(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
        content: str,
    ) -> str:
        """
        The memory_insert command allows you to insert new content into a memory block.
        This is used for adding new information to existing memory blocks.

        Args:
            label (str): Section of the memory to be edited, identified by its label.
            content (str): Content to insert into the memory. All unicode (including emojis) are supported.

        Returns:
            str: The success message
        """

        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

        current_value = str(agent_state.memory.get_block(label).value)
        new_value = current_value + "\n" + str(content)
        agent_state.memory.update_block_value(label=label, value=new_value)
        await AgentManager().update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
        return f"OK, inserted `{content}` into memory block with label `{label}`."

    async def memory_rethink(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
        old_content: str,
        new_content: str,
    ) -> str:
        """
        The memory_rethink command allows you to revise existing content in a memory block.
        This is used for updating or correcting information.

        Args:
            label (str): Section of the memory to be edited, identified by its label.
            old_content (str): The existing content to be replaced.
            new_content (str): The new content to replace the old content with.

        Returns:
            str: The success message
        """

        if agent_state.memory.get_block(label).read_only:
            raise ValueError(f"{READ_ONLY_BLOCK_EDIT_ERROR}")

        current_value = str(agent_state.memory.get_block(label).value)
        if old_content not in current_value:
            raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
        new_value = current_value.replace(str(old_content), str(new_content))
        agent_state.memory.update_block_value(label=label, value=new_value)
        await AgentManager().update_memory_if_changed_async(agent_id=agent_state.id, new_memory=agent_state.memory, actor=actor)
        return f"OK, replaced `{old_content}` with `{new_content}` in memory block with label `{label}`."

    async def memory_finish_edits(
        self,
        agent_state: AgentState,
        actor: User,
        label: str,
    ) -> str:
        """
        The memory_finish_edits command indicates that you are done editing a memory block.
        This is used to signal completion of memory editing operations.

        Args:
            label (str): Section of the memory that was edited, identified by its label.

        Returns:
            str: The success message
        """

        return f"OK, finished editing memory block with label `{label}`."