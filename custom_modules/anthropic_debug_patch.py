"""
Patch for Anthropic client to add better error handling and debugging for tool processing.
This patch adds detailed logging to identify which tool is causing issues during count_tokens.
"""

import json
from typing import List

# Monkey patch for better debugging
def patch_anthropic_client():
    """Apply debugging patches to the Anthropic client"""
    try:
        # Import the module that needs patching
        import letta.llm_api.anthropic_client as anthropic_client
        from letta.log import get_logger
        
        logger = get_logger(__name__)
        
        # Store original functions
        original_count_tokens = anthropic_client.AnthropicClient.count_tokens
        original_convert_tools = anthropic_client.convert_tools_to_anthropic_format
        
        # Enhanced convert_tools function with debugging
        def convert_tools_to_anthropic_format_debug(tools):
            """Enhanced version with detailed error logging"""
            formatted_tools = []
            
            for i, tool in enumerate(tools):
                try:
                    # Log tool details before conversion
                    logger.debug(f"Converting tool #{i}: {tool.function.name if hasattr(tool, 'function') and hasattr(tool.function, 'name') else 'Unknown'}")
                    
                    # Check if tool has the expected structure
                    if not hasattr(tool, 'function'):
                        logger.error(f"Tool #{i} missing 'function' attribute: {tool}")
                        continue
                        
                    if not hasattr(tool.function, 'name'):
                        logger.error(f"Tool #{i} function missing 'name': {tool.function}")
                        continue
                    
                    # Log the parameters structure
                    if hasattr(tool.function, 'parameters'):
                        logger.debug(f"Tool #{i} ({tool.function.name}) parameters type: {type(tool.function.parameters)}")
                        if tool.function.parameters:
                            logger.debug(f"Tool #{i} ({tool.function.name}) parameters: {json.dumps(tool.function.parameters, indent=2)}")
                    
                    formatted_tool = {
                        "name": tool.function.name,
                        "description": tool.function.description if hasattr(tool.function, 'description') and tool.function.description else "",
                        "input_schema": tool.function.parameters if hasattr(tool.function, 'parameters') and tool.function.parameters else {"type": "object", "properties": {}, "required": []},
                    }
                    
                    # Validate the formatted tool
                    if not isinstance(formatted_tool["input_schema"], dict):
                        logger.error(f"Tool #{i} ({tool.function.name}) input_schema is not a dict: {type(formatted_tool['input_schema'])}")
                        formatted_tool["input_schema"] = {"type": "object", "properties": {}, "required": []}
                    
                    formatted_tools.append(formatted_tool)
                    
                except Exception as e:
                    logger.error(f"Error converting tool #{i}: {str(e)}", exc_info=True)
                    if hasattr(tool, 'function') and hasattr(tool.function, 'name'):
                        logger.error(f"Tool name: {tool.function.name}")
                    logger.error(f"Tool object: {tool}")
                    # Continue processing other tools
                    continue
            
            logger.info(f"Successfully converted {len(formatted_tools)} out of {len(tools)} tools")
            return formatted_tools
        
        # Enhanced count_tokens function
        async def count_tokens_debug(self, messages=None, model=None, tools=None):
            """Enhanced version with better error handling"""
            import anthropic
            import logging
            
            logging.getLogger("httpx").setLevel(logging.WARNING)
            
            client = anthropic.AsyncAnthropic()
            if messages and len(messages) == 0:
                messages = None
                
            anthropic_tools = None
            if tools and len(tools) > 0:
                logger.info(f"Processing {len(tools)} tools for token counting")
                try:
                    anthropic_tools = convert_tools_to_anthropic_format_debug(tools)
                    logger.info(f"Successfully converted {len(anthropic_tools)} tools")
                except Exception as e:
                    logger.error(f"Failed to convert tools: {str(e)}", exc_info=True)
                    # Try to identify the problematic tool
                    for i, tool in enumerate(tools):
                        try:
                            _ = convert_tools_to_anthropic_format_debug([tool])
                        except Exception as tool_error:
                            logger.error(f"Tool #{i} is causing the error: {tool_error}")
                    raise
            
            try:
                result = await client.beta.messages.count_tokens(
                    model=model or "claude-3-7-sonnet-20250219",
                    messages=messages or [{"role": "user", "content": "hi"}],
                    tools=anthropic_tools or [],
                )
            except Exception as e:
                logger.error(f"Error calling Anthropic count_tokens API: {str(e)}", exc_info=True)
                if anthropic_tools:
                    logger.error(f"Number of tools sent: {len(anthropic_tools)}")
                    # Log first few tools for debugging
                    for i, tool in enumerate(anthropic_tools[:3]):
                        logger.error(f"Tool #{i}: {json.dumps(tool, indent=2)}")
                raise
            
            token_count = result.input_tokens
            if messages is None:
                token_count -= 8
            return token_count
        
        # Apply the patches
        anthropic_client.convert_tools_to_anthropic_format = convert_tools_to_anthropic_format_debug
        anthropic_client.AnthropicClient.count_tokens = count_tokens_debug
        
        logger.info("Successfully applied Anthropic client debugging patches")
        return True
        
    except Exception as e:
        print(f"Failed to apply Anthropic patches: {e}")
        return False

if __name__ == "__main__":
    # Test the patch
    if patch_anthropic_client():
        print("Anthropic client patches applied successfully")
    else:
        print("Failed to apply Anthropic client patches")