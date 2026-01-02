# Letta Webhook System - Implementation Roadmap

## Overview
This directory contains detailed specifications for adding a comprehensive webhook system to Letta. The webhook system will emit events on key lifecycle actions (agent steps, tool changes, run/job completion) to replace polling-based integrations with push-based webhooks.

## Architecture Summary

### Key Discovery
Letta **already has webhook infrastructure** for Jobs and Runs:
- `callback_url` fields in Job and Run ORM models
- `_dispatch_callback_async()` methods in JobManager and RunManager
- Callback delivery tracking (status, timestamps, errors)

### Design Approach
**Extend the existing callback pattern** to create a full-featured webhook system:
- Reuse proven dispatch patterns from JobManager/RunManager
- Add webhook config to Agent model (following Job/Run pattern)
- Create WebhookManager service to centralize event publishing
- Leverage existing Redis infrastructure for async delivery
- Support multiple event types with per-agent filtering

## Implementation Issues

### Phase 1: Core Infrastructure
1. **[01-webhook-module-foundation.md](./01-webhook-module-foundation.md)**
   - Create WebhookManager service
   - Event publishing API and delivery pipeline
   - HMAC signature generation and retry logic
   - **Estimated effort:** 3-5 days

2. **[02-orm-model-webhook-fields.md](./02-orm-model-webhook-fields.md)**
   - Add webhook config fields to Agent model
   - Create WebhookDelivery tracking table
   - Database migration scripts
   - **Estimated effort:** 2-3 days

3. **[03-webhook-settings-config.md](./03-webhook-settings-config.md)**
   - Global webhook settings (timeout, retries, security)
   - WebhookURLValidator with SSRF protections
   - Environment variable support
   - **Estimated effort:** 2-3 days

### Phase 2: Integration
4. **[04-agent-execution-integration.md](./04-agent-execution-integration.md)**
   - Integrate webhook emission into agent execution flow
   - Hook into step completion, tool attach/detach, message handling
   - Dependency injection for WebhookManager
   - **Estimated effort:** 4-6 days

5. **[05-webhook-crud-api.md](./05-webhook-crud-api.md)**
   - REST API endpoints for webhook configuration
   - Test endpoint, delivery history, manual retry
   - **Estimated effort:** 3-4 days

### Phase 3: Production Readiness
6. **[06-redis-async-delivery.md](./06-redis-async-delivery.md)**
   - Redis stream-based async delivery queue
   - Background worker with consumer groups
   - Graceful shutdown and monitoring
   - **Estimated effort:** 4-6 days

## Event Types

### Supported Events
- `agent.step.completed` - Agent completes a reasoning step
- `agent.step.failed` - Agent step encounters an error
- `agent.message.sent` - Message sent to agent via API
- `agent.tool.attached` - Tool attached to agent
- `agent.tool.detached` - Tool detached from agent
- `agent.run.completed` - Run completes (extends existing callback)
- `agent.job.completed` - Job completes (extends existing callback)
- `tool.execution.completed` - Tool finishes execution (optional)

## Integration Points Found

### Agent Execution
- `letta/agents/letta_agent_v2.py:507` - After `_step_checkpoint_finish()`
- `letta/services/step_manager.py:241` - After `update_step_success_async()`

### Tool Management
- `letta/services/agent_manager.py:2456` - After `attach_tool_async()`
- `letta/services/agent_manager.py:2657` - After `detach_tool_async()`

### Run/Job Completion (Already Has Callbacks!)
- `letta/services/run_manager.py:286` - Existing `_dispatch_callback_async()`
- `letta/services/job_manager.py:136` - Existing `_dispatch_callback_async()`

### Tool Execution
- `letta/services/tool_executor/tool_execution_manager.py:158` - After tool completes

### Message Handling
- `letta/server/rest_api/routers/v1/agents.py:1356` - After `send_message()` completes

## Technical Requirements

### Dependencies
- Redis (already integrated in your fork - commit e8ef06e6a)
- httpx (for async HTTP requests)
- SQLAlchemy event listeners (already used in message.py, block.py)

### Security Considerations
- SSRF protection (block localhost/private IPs in production)
- HMAC signature verification
- Webhook secret encryption at rest
- Rate limiting per agent
- Payload size limits
- Timeout enforcement

### Performance Targets
- <5ms overhead per event emission (with Redis queue)
- Support 1000+ webhooks/sec
- At-least-once delivery guarantee
- Exponential backoff for retries

## Your Fork's Relevant Changes
From commit analysis of `letta-repo`:
- **e8ef06e6a** - Redis integration for caching and job queues (highly relevant!)
- **b42a87e87** - Polling instrumentation traces (will be replaced by webhooks)

## Recommended Implementation Order

### Quick Win (1-2 weeks)
1. Issue #1 - WebhookManager foundation
2. Issue #2 - Agent ORM fields + migration
3. Issue #4 - Basic integration (step completion, tool attach/detach)
4. Test with your existing proxy

### Full Feature (3-4 weeks)
5. Issue #3 - Settings + validation
6. Issue #5 - CRUD API endpoints
7. Issue #6 - Redis async delivery + worker

### Production Hardening (1-2 weeks)
8. Load testing and performance optimization
9. Monitoring, metrics, alerting
10. Documentation and migration guide

## Testing Strategy

### Unit Tests
- WebhookManager event publishing
- HMAC signature generation/verification
- URL validation (SSRF scenarios)
- Retry logic with mock HTTP failures

### Integration Tests
- End-to-end webhook delivery
- Agent execution triggers webhook
- Tool attach/detach emits events
- Redis queue → worker → HTTP delivery

### Performance Tests
- 1000+ webhooks/sec throughput
- Event emission overhead (<5ms)
- Worker scalability (multiple consumers)

## Migration Path

### For Your Proxy
1. **Phase 1:** Keep existing proxy, add webhook config to agents
2. **Phase 2:** Proxy subscribes to webhook events (instead of polling)
3. **Phase 3:** Reduce/eliminate polling, rely on webhooks
4. **Phase 4:** Proxy becomes pure webhook receiver (no Letta polling)

### Backward Compatibility
- Existing Job/Run callbacks continue to work
- New webhook system is opt-in per agent
- No breaking changes to existing APIs

## Files Reference

### Files to Create
- `letta/services/webhook_manager.py`
- `letta/services/webhook_worker.py`
- `letta/orm/webhook_delivery.py`
- `letta/schemas/webhook.py`
- `letta/utils/webhook_validation.py`
- `alembic/versions/XXXX_add_webhook_fields_to_agent.py`

### Files to Modify
- `letta/orm/agent.py` - Add webhook config fields
- `letta/schemas/agent.py` - Add webhook config to AgentState
- `letta/settings.py` - Add webhook settings
- `letta/agents/letta_agent_v2.py` - Emit step events
- `letta/services/agent_manager.py` - Emit tool events
- `letta/services/run_manager.py` - Extend existing callbacks
- `letta/services/job_manager.py` - Extend existing callbacks
- `letta/server/rest_api/routers/v1/agents.py` - Add webhook endpoints
- `letta/server/server.py` - Initialize WebhookManager + worker
- `letta/data_sources/redis_client.py` - Add stream operations

## Next Steps

1. Review these specs and provide feedback
2. Prioritize which issues to tackle first
3. Create actual GitHub issues (or start implementation directly)
4. Set up development environment with Redis
5. Begin with Issue #1 (WebhookManager foundation)

## Questions to Answer

Before starting implementation:
1. **Delivery guarantees:** At-least-once (retries) or best-effort?
2. **Webhook configuration:** Per-agent only, or also global/org-level?
3. **Auth/signing:** HMAC sufficient, or need JWT/OAuth support?
4. **Payload content:** Include full step data, or just metadata + IDs?
5. **Existing proxy:** Can you share its webhook expectations?

## Contact
For questions or clarifications, refer to the individual issue files in this directory.
