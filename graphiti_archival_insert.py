# Graphiti-enhanced archival_memory_insert function
# This replaces the original function in core_tool_executor.py

    async def archival_memory_insert(self, agent_state: AgentState, actor: User, content: str) -> Optional[str]:
        """
        Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

        Args:
            content (str): Content to write to the memory. All unicode (including emojis) are supported.

        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        # Try Graphiti first if enabled
        if os.getenv("LETTA_GRAPHITI_ENABLED", "true").lower() == "true":
            try:
                graphiti_url = os.getenv("GRAPHITI_URL", "http://192.168.50.90:8001")
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
                        f"{graphiti_url}/add-memory",
                        json=memory_data,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Successfully stored memory in Graphiti for agent {agent_state.id}")
                        else:
                            error_text = await response.text()
                            logger.error(f"Graphiti storage failed ({response.status}): {error_text}")
                            # Fall through to original implementation
                            raise Exception("Graphiti storage failed")
                            
            except Exception as e:
                logger.warning(f"Graphiti storage error: {e}, using default storage")
                # Fall through to original implementation
        
        # Original implementation (also used as fallback)
        await PassageManager().insert_passage_async(
            agent_state=agent_state,
            agent_id=agent_state.id,
            text=content,
            actor=actor,
        )
        await AgentManager().rebuild_system_prompt_async(agent_id=agent_state.id, actor=actor, force=True)
        return None