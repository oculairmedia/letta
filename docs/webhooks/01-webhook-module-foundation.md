# Issue: Add webhook module foundation for event-driven notifications

**Type:** Enhancement  
**Priority:** High  
**Labels:** `enhancement`, `webhooks`, `infrastructure`

## Overview
Add a webhook system to Letta that emits events on key lifecycle actions (agent steps, tool changes, run/job completion). This will replace polling-based integrations with push-based webhooks.

## Background
Letta already has a callback URL system for Jobs and Runs (`callback_url` field + `_dispatch_callback_async()` methods in `JobManager` and `RunManager`). This issue extends that pattern to a general-purpose webhook system.

## Scope
Create the foundational webhook infrastructure:
- **WebhookManager service** (`letta/services/webhook_manager.py`)
- **Event publishing API** - lightweight `publish_event(agent_id, event_type, payload)`
- **Delivery pipeline** - HTTP POST with retries, HMAC signing, timeout handling
- **Persistence** - track delivery attempts, status, errors

## Event Types (Initial)
- `agent.step.completed`
- `agent.step.failed`
- `agent.message.sent`
- `agent.tool.attached`
- `agent.tool.detached`
- `agent.run.completed` (extend existing callback)
- `agent.job.completed` (extend existing callback)
- `tool.execution.completed`

## Acceptance Criteria
- [ ] `WebhookManager` service created with `publish_event()` and `_dispatch_webhook_async()`
- [ ] Event payload schema defined (event_id, event_type, timestamp, agent_id, data)
- [ ] HMAC signature generation (`X-Letta-Signature` header)
- [ ] Retry logic with exponential backoff (configurable max retries)
- [ ] Timeout handling (default 5s)
- [ ] Delivery tracking (status code, timestamp, error message)
- [ ] Unit tests for webhook dispatch and retry logic

## Related Issues
- `02-orm-model-webhook-fields.md` - Add webhook fields to Agent model
- `03-webhook-settings-config.md` - Add webhook settings and configuration
- `04-agent-execution-integration.md` - Integrate webhooks into agent execution flow
- `05-webhook-crud-api.md` - Add webhook CRUD API endpoints
- `06-redis-async-delivery.md` - Add Redis queue for async webhook delivery

## Files to Create
- `letta/services/webhook_manager.py`
- `tests/test_webhook_manager.py`

## Implementation Notes

### WebhookManager Service Structure
```python
class WebhookManager:
    async def publish_event(
        self, 
        agent_id: str, 
        event_type: str, 
        payload: dict
    ) -> None:
        """
        Publish an event for webhook delivery.
        Checks if agent has webhook configured and event type enabled.
        """
        
    async def _dispatch_webhook_async(
        self,
        webhook_url: str,
        event: WebhookEvent,
        secret: Optional[str] = None
    ) -> WebhookDelivery:
        """
        Dispatch a single webhook with retry logic.
        Returns delivery status for persistence.
        """
        
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC-SHA256 signature for webhook verification"""
```

### Event Payload Schema
```python
{
    "event_id": "evt_abc123...",
    "event_type": "agent.step.completed",
    "timestamp": "2025-01-01T12:00:00Z",
    "agent_id": "agent-uuid",
    "organization_id": "org-uuid",
    "data": {
        # Event-specific payload
        "step_id": "step-uuid",
        "status": "completed",
        "usage": {...}
    }
}
```

### HTTP Headers
```
POST {webhook_url}
Content-Type: application/json
X-Letta-Event-Type: agent.step.completed
X-Letta-Event-Id: evt_abc123...
X-Letta-Signature: sha256=...
X-Letta-Timestamp: 1234567890
User-Agent: Letta-Webhooks/1.0
```

## References
- Existing callback pattern: `letta/services/run_manager.py:301-327` (`_dispatch_callback_async`)
- Existing callback pattern: `letta/services/job_manager.py:301-327`
- Redis client: `letta/data_sources/redis_client.py`
- Settings: `letta/settings.py`

## Testing Requirements
- Unit tests for signature generation
- Unit tests for retry logic (mock HTTP failures)
- Unit tests for event filtering
- Integration test for full webhook delivery flow
