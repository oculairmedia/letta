#!/usr/bin/env python3
"""
Test script to verify Graphiti archival memory integration is working.
"""

import asyncio
import aiohttp
import json
import uuid
from datetime import datetime

# Letta API configuration
LETTA_API_URL = "https://letta2.oculair.ca"
LETTA_PASSWORD = "lettaSecurePass123"  # From your config


async def create_test_agent():
    """Create a test agent"""
    async with aiohttp.ClientSession() as session:
        # Create agent with unique name
        agent_name = f"graphiti_test_{uuid.uuid4().hex[:8]}"
        
        agent_data = {
            "name": agent_name,
            "system": "You are a test agent for Graphiti archival memory. Use archival_memory_insert to store important information and archival_memory_search to retrieve it.",
            "tools": ["archival_memory_insert", "archival_memory_search"],
            "description": "Test agent for Graphiti integration"
        }
        
        headers = {
            "Authorization": f"Bearer {LETTA_PASSWORD}",
            "Content-Type": "application/json"
        }
        
        async with session.post(
            f"{LETTA_API_URL}/v1/agents",
            json=agent_data,
            headers=headers
        ) as response:
            if response.status == 200:
                agent = await response.json()
                print(f"✅ Created test agent: {agent['name']} (ID: {agent['id']})")
                return agent
            else:
                error = await response.text()
                print(f"❌ Failed to create agent: {error}")
                return None


async def send_message(agent_id: str, message: str):
    """Send a message to the agent"""
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {LETTA_PASSWORD}",
            "Content-Type": "application/json"
        }
        
        message_data = {
            "message": message,
            "return_message_object": True
        }
        
        async with session.post(
            f"{LETTA_API_URL}/v1/agents/{agent_id}/messages",
            json=message_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error = await response.text()
                print(f"❌ Failed to send message: {error}")
                return None


async def test_graphiti_integration():
    """Test the Graphiti archival memory integration"""
    print("🧪 Testing Graphiti Archival Memory Integration\n")
    
    # Create test agent
    agent = await create_test_agent()
    if not agent:
        return False
    
    agent_id = agent["id"]
    
    # Test 1: Insert memory
    print("\n📝 Test 1: Inserting memory into Graphiti")
    response = await send_message(
        agent_id,
        "Please store this in archival memory: 'Graphiti integration test successful at " + 
        datetime.now().isoformat() + ". The system can store and retrieve graph-based memories.'"
    )
    
    if response:
        # Look for tool calls in response
        for msg in response.get("messages", []):
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("function", {}).get("name") == "archival_memory_insert":
                        print("✅ Agent called archival_memory_insert")
                        print(f"   Content: {tool_call.get('function', {}).get('arguments', {})}")
    
    # Wait a moment for indexing
    await asyncio.sleep(2)
    
    # Test 2: Search memory
    print("\n🔍 Test 2: Searching Graphiti archival memory")
    response = await send_message(
        agent_id,
        "Search your archival memory for 'Graphiti integration test'"
    )
    
    if response:
        # Look for search results
        for msg in response.get("messages", []):
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("function", {}).get("name") == "archival_memory_search":
                        print("✅ Agent called archival_memory_search")
                        print(f"   Query: {tool_call.get('function', {}).get('arguments', {})}")
            
            # Check if results were returned
            if msg.get("role") == "assistant" and "test successful" in msg.get("text", "").lower():
                print("✅ Agent found the stored memory!")
    
    # Test 3: Verify in Graphiti directly
    print("\n🔗 Test 3: Verifying in Graphiti directly")
    async with aiohttp.ClientSession() as session:
        search_data = {
            "query": "Graphiti integration test",
            "max_nodes": 5,
            "group_ids": [agent_id]
        }
        
        async with session.post(
            "http://192.168.50.90:8001/search/nodes",
            json=search_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                nodes = data.get("nodes", [])
                if nodes:
                    print(f"✅ Found {len(nodes)} nodes in Graphiti")
                    for node in nodes:
                        print(f"   - {node.get('name', 'Unknown')}: {node.get('summary', '')[:100]}...")
                else:
                    print("❌ No nodes found in Graphiti")
    
    print("\n✨ Graphiti archival memory integration test complete!")
    return True


async def main():
    """Main test function"""
    success = await test_graphiti_integration()
    return success


if __name__ == "__main__":
    asyncio.run(main())