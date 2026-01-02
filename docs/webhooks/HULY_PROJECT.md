# Letta Core Development - Huly Project Tracking

**Project:** [LCORE - Letta Core Development](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE)  
**Repository:** `/opt/stacks/letta/letta-repo`  
**Created:** 2026-01-01

## Overview

This Huly project tracks the implementation of a comprehensive webhook system for Letta that emits events on key lifecycle actions (agent steps, tool changes, run/job completion) to replace polling-based integrations with push-based webhooks.

## Components

### 🔔 Webhook System
Event emission infrastructure for agent execution, tool operations, and job completion.

**Key Responsibilities:**
- WebhookManager service with `publish_event()` API
- HMAC-SHA256 signature generation
- Exponential backoff retry logic
- Redis stream-based async delivery
- SSRF protection

### ⚙️ Core Services
Platform services, configuration, and infrastructure.

**Key Responsibilities:**
- Global webhook settings
- SSRF protection and URL validation
- Server initialization
- Redis client integration

### 🌐 API
REST API endpoints for webhook management.

**Key Responsibilities:**
- Configure/update/delete webhooks
- Test webhook delivery
- View delivery history
- Manual retry failed deliveries

### 🗄️ Database
ORM models, migrations, and schema changes.

**Key Responsibilities:**
- Add webhook fields to Agent model
- Create WebhookDelivery tracking table
- Alembic migrations
- Webhook secret encryption

## Milestones

### Phase 1: Foundation (Target: Feb 1, 2025)
Core webhook infrastructure - WebhookManager service, database schema, and configuration.

**Issues:** LCORE-1, LCORE-2, LCORE-3

### Phase 2: Integration (Target: Feb 15, 2025)
Agent execution integration and REST API endpoints.

**Issues:** LCORE-4, LCORE-5

### Phase 3: Async Delivery (Target: Mar 1, 2025)
Redis queue for async webhook delivery with retry logic.

**Issues:** LCORE-6, LCORE-7, LCORE-8

## Issues

| ID | Title | Component | Phase | Priority | Status |
|----|-------|-----------|-------|----------|--------|
| [LCORE-1](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE-1) | Create WebhookManager service foundation | Webhook System | Phase 1 | High | Backlog |
| [LCORE-2](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE-2) | Add webhook fields to Agent ORM model | Database | Phase 1 | High | Backlog |
| [LCORE-3](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE-3) | Implement webhook settings and SSRF protection | Core Services | Phase 1 | High | Backlog |
| [LCORE-4](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE-4) | Integrate webhooks into agent execution flow | Webhook System | Phase 2 | High | Backlog |
| [LCORE-5](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE-5) | Create webhook CRUD API endpoints | API | Phase 2 | Medium | Backlog |
| [LCORE-6](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE-6) | Implement Redis-based async webhook delivery | Webhook System | Phase 3 | Medium | Backlog |
| [LCORE-7](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE-7) | Write webhook system tests | Webhook System | Phase 3 | Medium | Backlog |
| [LCORE-8](https://pm.oculair.ca/workbench/agentspace/tracker/LCORE-8) | Document webhook system | Webhook System | Phase 3 | Low | Backlog |

## Dependencies

```
LCORE-1 (WebhookManager)
├── LCORE-4 (Agent integration) - requires WebhookManager
└── LCORE-6 (Async delivery) - requires WebhookManager

LCORE-2 (ORM fields)
├── LCORE-4 (Agent integration) - requires webhook config fields
└── LCORE-5 (API endpoints) - requires webhook config fields

LCORE-3 (Settings) - standalone

LCORE-7 (Tests)
├── requires LCORE-1
├── requires LCORE-4
└── requires LCORE-6

LCORE-8 (Docs) - requires all implementation issues
```

## Implementation Sequence

### Quick Win (1-2 weeks)
Minimal viable webhook system for testing with existing proxy:
1. LCORE-1 - WebhookManager foundation
2. LCORE-2 - Agent ORM fields + migration
3. LCORE-4 - Basic integration (step completion, tool attach/detach)

### Full Feature (3-4 weeks)
Complete webhook system with all event types and API:
4. LCORE-3 - Settings + SSRF protection
5. LCORE-5 - CRUD API endpoints
6. LCORE-6 - Redis async delivery + worker

### Production Hardening (1-2 weeks)
Testing, documentation, and monitoring:
7. LCORE-7 - Comprehensive tests
8. LCORE-8 - User + developer documentation

## Event Types

| Event Type | Trigger Point | File Location |
|------------|---------------|---------------|
| `agent.step.completed` | After step checkpoint | `letta/agents/letta_agent_v2.py:507` |
| `agent.step.failed` | Step error | `letta/agents/letta_agent_v2.py` |
| `agent.message.sent` | After send_message | `letta/server/rest_api/routers/v1/agents.py:1356` |
| `agent.tool.attached` | After attach_tool_async | `letta/services/agent_manager.py:2456` |
| `agent.tool.detached` | After detach_tool_async | `letta/services/agent_manager.py:2657` |
| `agent.run.completed` | Run finishes | `letta/services/run_manager.py:286` |
| `agent.job.completed` | Job finishes | `letta/services/job_manager.py:136` |
| `tool.execution.completed` | Tool finishes | `letta/services/tool_executor/tool_execution_manager.py:158` |

## Files Reference

### Files to Create
- `letta/services/webhook_manager.py` - Core service (LCORE-1)
- `letta/services/webhook_worker.py` - Redis consumer (LCORE-6)
- `letta/orm/webhook_delivery.py` - Delivery tracking model (LCORE-2)
- `letta/schemas/webhook.py` - Event schemas (LCORE-1)
- `letta/utils/webhook_validation.py` - SSRF protection (LCORE-3)
- `alembic/versions/XXXX_add_webhook_fields.py` - Migration (LCORE-2)

### Files to Modify
- `letta/orm/agent.py` - Add webhook config fields (LCORE-2)
- `letta/schemas/agent.py` - Add webhook to AgentState (LCORE-2)
- `letta/settings.py` - Add webhook settings (LCORE-3)
- `letta/agents/letta_agent_v2.py` - Emit step events (LCORE-4)
- `letta/services/agent_manager.py` - Emit tool events (LCORE-4)
- `letta/services/run_manager.py` - Extend callbacks (LCORE-4)
- `letta/services/job_manager.py` - Extend callbacks (LCORE-4)
- `letta/server/rest_api/routers/v1/agents.py` - Add endpoints (LCORE-5)
- `letta/server/server.py` - Initialize WebhookManager (LCORE-4)
- `letta/data_sources/redis_client.py` - Add streams (LCORE-6)

## Technical Specifications

### Event Payload Schema
```json
{
  "event_id": "evt_abc123...",
  "event_type": "agent.step.completed",
  "timestamp": "2025-01-01T12:00:00Z",
  "agent_id": "agent-uuid",
  "organization_id": "org-uuid",
  "data": {
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
X-Letta-Signature: sha256=abc123...
X-Letta-Event: agent.step.completed
X-Letta-Delivery: delivery-uuid
User-Agent: Letta-Webhook/1.0
```

### Security Features
- HMAC-SHA256 signature verification
- SSRF protection (blocks localhost, private IPs, metadata endpoints)
- Webhook secret encryption at rest
- HTTPS enforcement in production
- Rate limiting per agent
- Timeout enforcement (default: 5s)

### Performance Targets
- <5ms overhead per event emission (with Redis queue)
- Support 1000+ webhooks/sec
- At-least-once delivery guarantee
- Exponential backoff for retries

## Related Documentation

- [Implementation Roadmap](./README.md)
- [01 - WebhookManager Foundation](./01-webhook-module-foundation.md) → LCORE-1
- [02 - ORM Model Changes](./02-orm-model-webhook-fields.md) → LCORE-2
- [03 - Settings & Configuration](./03-webhook-settings-config.md) → LCORE-3
- [04 - Agent Execution Integration](./04-agent-execution-integration.md) → LCORE-4
- [05 - Webhook CRUD API](./05-webhook-crud-api.md) → LCORE-5
- [06 - Redis Async Delivery](./06-redis-async-delivery.md) → LCORE-6

## Migration Path

### For Existing Polling Proxy
1. **Phase 1:** Keep existing proxy, add webhook config to agents
2. **Phase 2:** Proxy subscribes to webhook events (instead of polling)
3. **Phase 3:** Reduce/eliminate polling, rely on webhooks
4. **Phase 4:** Proxy becomes pure webhook receiver (no Letta polling)

### Backward Compatibility
- Existing Job/Run callbacks continue to work
- New webhook system is opt-in per agent
- No breaking changes to existing APIs

## Testing Strategy

### Unit Tests (LCORE-7)
- WebhookManager event publishing
- HMAC signature generation/verification
- URL validation (SSRF scenarios)
- Retry logic with mock HTTP failures

### Integration Tests (LCORE-7)
- End-to-end webhook delivery
- Agent execution triggers webhook
- Tool attach/detach emits events
- Redis queue → worker → HTTP delivery

### Performance Tests
- 1000+ webhooks/sec throughput
- Event emission overhead (<5ms)
- Worker scalability (multiple consumers)

## Open Questions

Before starting implementation:
1. **Delivery guarantees:** At-least-once (retries) or best-effort?
2. **Webhook configuration:** Per-agent only, or also global/org-level?
3. **Auth/signing:** HMAC sufficient, or need JWT/OAuth support?
4. **Payload content:** Include full step data, or just metadata + IDs?
5. **Existing proxy:** What are its webhook expectations?

## Contact

- **Project Manager:** Emmanuel (via Matrix)
- **Repository:** oculairmedia/letta (fork)
- **Upstream:** letta-ai/letta
