# Issue: Integrate webhook events into agent execution flow

**Type:** Enhancement  
**Priority:** High  
**Labels:** `enhancement`, `webhooks`, `agent-core`

## Overview
Add webhook event emission at key lifecycle points in the agent execution flow. This integrates the WebhookManager into the core agent loop, step tracking, tool management, and message handling.

## Scope
Insert `webhook_manager.publish_event()` calls at strategic integration points identified in the codebase analysis.

## Integration Points

### 1. Agent Step Completion
**File:** `letta/agents/letta_agent_v2.py`

**Location A:** After `_step_checkpoint_finish()` (line ~507)
```python
async def _step(self, ...):
    # ... existing step execution ...
    
    # Finish step checkpoint
    await self._step_checkpoint_finish(
        step_id=step_id,
        usage_stats=usage_stats,
        # ...
    )
    
    # NEW: Emit webhook event
    if self.webhook_manager:
        await self.webhook_manager.publish_event(
            agent_id=self.agent_state.id,
            event_type="agent.step.completed",
            payload={
                "step_id": step_id,
                "status": "completed",
                "usage": usage_stats.model_dump() if usage_stats else None,
                "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    
    return tool_calls, usage_stats
```

**Location B:** In error handling (line ~460-490)
```python
except Exception as e:
    # ... existing error handling ...
    
    # NEW: Emit step failure webhook
    if self.webhook_manager:
        await self.webhook_manager.publish_event(
            agent_id=self.agent_state.id,
            event_type="agent.step.failed",
            payload={
                "step_id": step_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    
    raise
```

### 2. Tool Attachment/Detachment
**File:** `letta/services/agent_manager.py`

**Location A:** After successful tool attachment (line ~2456)
```python
async def attach_tool_async(
    self,
    agent_id: str,
    tool_id: str,
    actor: PydanticUser,
) -> Tool:
    # ... existing attach logic ...
    
    await session.commit()
    
    # NEW: Emit webhook event
    if self.webhook_manager:
        await self.webhook_manager.publish_event(
            agent_id=agent_id,
            event_type="agent.tool.attached",
            payload={
                "tool_id": tool_id,
                "tool_name": tool.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    
    return tool
```

**Location B:** After successful tool detachment (line ~2657)
```python
async def detach_tool_async(
    self,
    agent_id: str,
    tool_id: str,
    actor: PydanticUser,
) -> None:
    # ... existing detach logic ...
    
    await session.commit()
    
    # NEW: Emit webhook event
    if self.webhook_manager:
        await self.webhook_manager.publish_event(
            agent_id=agent_id,
            event_type="agent.tool.detached",
            payload={
                "tool_id": tool_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
```

### 3. Message Sent (API Endpoint)
**File:** `letta/server/rest_api/routers/v1/agents.py`

**Location:** After message processing completes (line ~1356)
```python
@router.post("/{agent_id}/messages", ...)
async def send_message(
    agent_id: str,
    request: LettaRequest,
    ...
):
    # ... existing message handling ...
    
    response = await server.send_message_to_agent(
        agent_id=agent_id,
        user_id=actor.id,
        messages=request.messages,
        ...
    )
    
    # NEW: Emit webhook event
    if server.webhook_manager:
        await server.webhook_manager.publish_event(
            agent_id=agent_id,
            event_type="agent.message.sent",
            payload={
                "run_id": run.id if run else None,
                "message_count": len(request.messages),
                "usage": response.usage.model_dump() if response.usage else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    
    return response
```

### 4. Tool Execution Completion
**File:** `letta/services/tool_executor/tool_execution_manager.py`

**Location:** After tool execution (line ~158)
```python
async def execute_tool_async(
    self,
    agent_id: str,
    tool_name: str,
    tool_args: dict,
    ...
) -> ToolExecutionResult:
    # ... existing execution logic ...
    
    # Record metrics
    self.metrics_registry.record(...)
    
    # NEW: Emit webhook event (optional - may be too noisy)
    if self.webhook_manager and self.settings.webhook_emit_tool_execution:
        await self.webhook_manager.publish_event(
            agent_id=agent_id,
            event_type="tool.execution.completed",
            payload={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "status": "success" if result.success else "error",
                "execution_time_ms": execution_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    
    return result
```

### 5. Extend Existing Run/Job Callbacks
**File:** `letta/services/run_manager.py`

**Location:** Modify `update_run_by_id_async()` (line ~286)
```python
# Existing callback dispatch
if needs_callback and run_record.callback_url:
    await self._dispatch_callback_async(...)

# NEW: Also emit webhook event if agent has webhook configured
if needs_callback and self.webhook_manager:
    await self.webhook_manager.publish_event(
        agent_id=run_record.agent_id,
        event_type="agent.run.completed",
        payload={
            "run_id": run_record.id,
            "status": run_record.status,
            "completed_at": run_record.completed_at.isoformat(),
            "usage": run_record.usage.model_dump() if run_record.usage else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
```

**File:** `letta/services/job_manager.py`

**Location:** Modify `update_job_by_id_async()` (line ~136)
```python
# Existing callback dispatch
if needs_callback and job_record.callback_url:
    await self._dispatch_callback_async(...)

# NEW: Also emit webhook event if agent has webhook configured
if needs_callback and self.webhook_manager:
    # Note: Jobs may not have agent_id, handle accordingly
    if job_record.metadata and "agent_id" in job_record.metadata:
        await self.webhook_manager.publish_event(
            agent_id=job_record.metadata["agent_id"],
            event_type="agent.job.completed",
            payload={
                "job_id": job_record.id,
                "status": job_record.status,
                "completed_at": job_record.completed_at.isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
```

## Dependency Injection

### Add WebhookManager to Service Classes
Each service that emits webhooks needs WebhookManager injected:

**`letta/agents/letta_agent_v2.py`:**
```python
class LettaAgentV2:
    def __init__(
        self,
        # ... existing params ...
        webhook_manager: Optional[WebhookManager] = None,
    ):
        self.webhook_manager = webhook_manager
```

**`letta/services/agent_manager.py`:**
```python
class AgentManager:
    def __init__(
        self,
        # ... existing params ...
        webhook_manager: Optional[WebhookManager] = None,
    ):
        self.webhook_manager = webhook_manager
```

### Server Initialization
**`letta/server/server.py`:**
```python
class SyncServer:
    def __init__(self, ...):
        # ... existing init ...
        
        # Initialize webhook manager
        self.webhook_manager = WebhookManager(
            settings=self.settings,
            redis_client=self.redis_client,  # If using Redis queue
        )
        
        # Pass to managers
        self.agent_manager = AgentManager(
            # ...
            webhook_manager=self.webhook_manager,
        )
```

## Acceptance Criteria
- [ ] Webhook events emitted from all 5 integration points
- [ ] WebhookManager injected into relevant service classes
- [ ] Server initialization creates and shares WebhookManager instance
- [ ] Events only emitted if agent has `webhook_enabled=True`
- [ ] Events respect agent's `webhook_events` filter
- [ ] Error handling ensures webhook failures don't break agent execution
- [ ] Integration tests verify events are emitted at correct times
- [ ] Performance testing shows minimal overhead (<5ms per event)

## Error Handling Strategy
Webhook delivery must NEVER block or crash the agent:

```python
try:
    if self.webhook_manager:
        await self.webhook_manager.publish_event(...)
except Exception as e:
    # Log error but don't propagate
    logger.warning(f"Failed to publish webhook event: {e}", exc_info=True)
    # Optionally record metric for monitoring
    metrics.increment("webhook.publish.error")
```

## Performance Considerations
- Use async/await throughout to avoid blocking
- If Redis queue is enabled, `publish_event()` should be near-instant (just write to Redis)
- If Redis is disabled, consider fire-and-forget `asyncio.create_task()` for dispatch
- Add circuit breaker if webhook endpoint is consistently failing

## Files to Modify
- `letta/agents/letta_agent_v2.py`
- `letta/agents/letta_agent_v3.py` (similar changes)
- `letta/services/agent_manager.py`
- `letta/services/run_manager.py`
- `letta/services/job_manager.py`
- `letta/services/tool_executor/tool_execution_manager.py`
- `letta/server/rest_api/routers/v1/agents.py`
- `letta/server/server.py`

## Related Issues
- `01-webhook-module-foundation.md` - WebhookManager implementation
- `02-orm-model-webhook-fields.md` - Agent webhook config fields
- `06-redis-async-delivery.md` - Async delivery infrastructure

## Testing Strategy
- Unit tests: Mock WebhookManager and verify `publish_event()` called with correct args
- Integration tests: Real WebhookManager + mock HTTP server to verify delivery
- Performance tests: Measure overhead of webhook emission (should be <5ms with Redis)
