#!/bin/bash

echo "Applying conversation_search fix to Letta container..."

# Copy the original file from container
docker cp letta-letta-1:/app/letta/services/tool_executor/core_tool_executor.py /tmp/core_tool_executor.py.orig

# Apply the fix directly using sed
docker exec letta-letta-1 sed -i \
    -e 's/list_agent_messages_async/list_messages_for_agent_async/g' \
    -e '/embedding_config=agent_state.embedding_config,/d' \
    -e '/embed_query=True,/d' \
    /app/letta/services/tool_executor/core_tool_executor.py

echo "Fix applied. Verifying changes..."

# Verify the changes
echo "Checking for list_agent_messages_async (should return nothing):"
docker exec letta-letta-1 grep "list_agent_messages_async" /app/letta/services/tool_executor/core_tool_executor.py || echo "✓ Old method name removed"

echo -e "\nChecking for list_messages_for_agent_async (should find it):"
docker exec letta-letta-1 grep "list_messages_for_agent_async" /app/letta/services/tool_executor/core_tool_executor.py && echo "✓ New method name in place"

echo -e "\nChecking for removed parameters (should return nothing):"
docker exec letta-letta-1 grep -E "(embedding_config=agent_state|embed_query=True)" /app/letta/services/tool_executor/core_tool_executor.py || echo "✓ Invalid parameters removed"

echo -e "\nFix applied successfully!"
echo "Note: You may need to restart the Letta service for changes to take effect."
echo "Run: cd /opt/stacks/letta && docker-compose restart letta"