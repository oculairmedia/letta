# Letta Webhooks

Webhooks provide real-time event notifications from Letta agents. Instead of polling the API for updates, configure webhooks to receive HTTP POST requests when events occur.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Event Types](#event-types)
- [Configuration](#configuration)
- [Security](#security)
- [Delivery](#delivery)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

Webhooks emit events for key agent lifecycle actions:
- Agent execution (run completed)
- Tool lifecycle (created, updated, deleted)
- Agent tool operations (attached, detached)
- Memory updates
- Job and run completion

### Webhook Scopes

Letta supports two levels of webhooks:

| Scope | Configuration | Use Case |
|-------|---------------|----------|
| **Global** | Environment variables | Platform-level events (tool CRUD) |
| **Per-Agent** | Agent fields via API | Agent-specific events (runs, tool attach/detach) |

### Features

- **Global webhooks** - Platform-wide events delivered to a central endpoint
- **Per-agent configuration** - Each agent can have its own webhook URL and event subscriptions
- **Event filtering** - Subscribe to specific event types or all events
- **HMAC signatures** - Verify webhook authenticity with shared secrets
- **Delivery persistence** - Track delivery attempts, status, and errors
- **Automatic retries** - Failed deliveries are retried with exponential backoff
- **SSRF protection** - Built-in validation prevents requests to private networks

---

## Quick Start

### 1. Configure a Webhook

```bash
curl -X PUT "http://localhost:8283/v1/agents/{agent_id}/webhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-server.com/webhooks/letta",
    "secret": "your-webhook-secret",
    "events": ["agent.run.completed", "agent.step.completed"],
    "enabled": true
  }'
```

### 2. Receive Webhook Events

Your webhook endpoint will receive POST requests with this structure:

```json
{
  "id": "evt-550e8400-e29b-41d4-a716-446655440000",
  "event_type": "agent.run.completed",
  "timestamp": "2026-01-01T20:15:30.123456Z",
  "agent_id": "agent-123",
  "organization_id": "org-456",
  "data": {
    "step_id": "step-789",
    "status": "completed",
    "messages": [...]
  }
}
```

### 3. Verify the Signature

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    """Verify HMAC-SHA256 signature from X-Letta-Signature header."""
    expected = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

# In your webhook handler
payload = request.get_data(as_text=True)
signature = request.headers.get('X-Letta-Signature')
secret = 'your-webhook-secret'

if verify_webhook(payload, signature, secret):
    # Process the webhook
    pass
else:
    # Invalid signature
    return 401
```

---

## Event Types

### Global Events (Platform-Level)

These events are delivered to the global webhook URL (`LETTA_WEBHOOK_GLOBAL_URL`).

| Event Type | Trigger | Scope | Data Fields |
|------------|---------|-------|-------------|
| `tool.created` | Tool created via API | Global | `tool_id`, `tool_name`, `tool_type`, `description`, `json_schema`, `tags` |
| `tool.updated` | Tool updated via API | Global | `tool_id`, `tool_name`, `tool_type`, `description`, `json_schema`, `tags` |
| `tool.deleted` | Tool deleted via API | Global | `tool_id`, `tool_name` |

### Per-Agent Events

These events are delivered to the agent's configured webhook URL.

#### Agent Execution Events

| Event Type | Description | Data Fields |
|------------|-------------|-------------|
| `agent.run.started` | Agent run begins | `run_id`, `user_id` |
| `agent.run.completed` | Agent run finishes successfully | `run_id`, `step_count`, `messages`, `usage` |
| `agent.run.failed` | Agent run encounters error | `run_id`, `error`, `step_id` |
| `agent.step.completed` | Single agent step finishes | `step_id`, `messages`, `tool_calls`, `usage` |
| `agent.step.failed` | Single agent step fails | `step_id`, `error` |

#### Agent Tool Events

| Event Type | Description | Data Fields |
|------------|-------------|-------------|
| `agent.tool.attached` | Tool attached to agent | `agent_id`, `tool_id`, `tool_name` |
| `agent.tool.detached` | Tool detached from agent | `agent_id`, `tool_id`, `tool_name` |
| `tool.execution.completed` | Tool execution finishes | `tool_id`, `tool_name`, `arguments`, `result` |
| `tool.execution.failed` | Tool execution fails | `tool_id`, `tool_name`, `arguments`, `error` |

#### State Events

| Event Type | Description | Data Fields |
|------------|-------------|-------------|
| `agent.state.updated` | Agent configuration changed | `changes` (list of modified fields) |
| `agent.memory.updated` | Agent memory modified | `block_label`, `old_value`, `new_value` |
| `agent.message.sent` | Message sent to agent | `message_id`, `role`, `text` |

#### Job Events

| Event Type | Description | Data Fields |
|------------|-------------|-------------|
| `agent.job.completed` | Background job finishes | `job_id`, `status`, `result` |
| `agent.job.failed` | Background job fails | `job_id`, `error` |

---

## Payload Schemas

### Common Event Structure

All webhook events share this structure:

```json
{
  "id": "evt-{uuid}",
  "event_type": "event.type.here",
  "timestamp": "2026-01-02T20:15:30.123456Z",
  "agent_id": "agent-{uuid}",
  "tool_id": "tool-{uuid}",
  "organization_id": "org-{uuid}",
  "data": { ... }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique event identifier (format: `evt-{uuid}`) |
| `event_type` | string | The type of event (see Event Types above) |
| `timestamp` | string | ISO 8601 timestamp when event was created |
| `agent_id` | string \| null | Agent ID (null for global events like `tool.*`) |
| `tool_id` | string \| null | Tool ID for tool-related events |
| `organization_id` | string | Organization that owns the resource |
| `data` | object | Event-specific payload data |

### Tool Created/Updated Payload

```json
{
  "id": "evt-550e8400-e29b-41d4-a716-446655440000",
  "event_type": "tool.created",
  "timestamp": "2026-01-02T20:15:30.123456Z",
  "agent_id": null,
  "tool_id": "tool-123e4567-e89b-12d3-a456-426614174000",
  "organization_id": "org-456",
  "data": {
    "tool_id": "tool-123e4567-e89b-12d3-a456-426614174000",
    "tool_name": "web_search",
    "tool_type": "custom",
    "description": "Search the web for information",
    "json_schema": {
      "name": "web_search",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
      }
    },
    "tags": ["search", "web"]
  }
}
```

### Tool Deleted Payload

```json
{
  "id": "evt-550e8400-e29b-41d4-a716-446655440001",
  "event_type": "tool.deleted",
  "timestamp": "2026-01-02T20:15:30.123456Z",
  "agent_id": null,
  "tool_id": "tool-123e4567-e89b-12d3-a456-426614174000",
  "organization_id": "org-456",
  "data": {
    "tool_id": "tool-123e4567-e89b-12d3-a456-426614174000",
    "tool_name": "web_search"
  }
}
```

### Agent Tool Attached/Detached Payload

```json
{
  "id": "evt-550e8400-e29b-41d4-a716-446655440002",
  "event_type": "agent.tool.attached",
  "timestamp": "2026-01-02T20:15:30.123456Z",
  "agent_id": "agent-789",
  "tool_id": "tool-123e4567-e89b-12d3-a456-426614174000",
  "organization_id": "org-456",
  "data": {
    "agent_id": "agent-789",
    "tool_id": "tool-123e4567-e89b-12d3-a456-426614174000",
    "tool_name": "web_search"
  }
}
```

### Agent Run Completed Payload

```json
{
  "id": "evt-550e8400-e29b-41d4-a716-446655440003",
  "event_type": "agent.run.completed",
  "timestamp": "2026-01-02T20:15:30.123456Z",
  "agent_id": "agent-789",
  "tool_id": null,
  "organization_id": "org-456",
  "data": {
    "run_id": "run-abc123",
    "step_count": 3,
    "messages": [
      {
        "id": "message-001",
        "role": "assistant",
        "text": "I found the information you requested...",
        "tool_calls": null,
        "created_at": "2026-01-02T20:15:28.000000Z"
      }
    ],
    "usage": {
      "completion_tokens": 150,
      "prompt_tokens": 500,
      "total_tokens": 650
    }
  }
}
```

---

## Configuration

### Global Webhooks (Environment Variables)

Configure global webhooks for platform-level events (`tool.created`, `tool.updated`, `tool.deleted`):

```bash
# Required: URL to receive global webhook events
export LETTA_WEBHOOK_GLOBAL_URL="https://your-service.com/webhooks/tools"

# Optional: Secret for HMAC-SHA256 signature verification
export LETTA_WEBHOOK_GLOBAL_SECRET="your-secret-key"
```

**Docker Compose example:**

```yaml
services:
  letta:
    image: letta/letta:latest
    environment:
      - LETTA_WEBHOOK_GLOBAL_URL=https://your-service.com/webhooks/tools
      - LETTA_WEBHOOK_GLOBAL_SECRET=your-secret-key
```

Global webhooks are triggered automatically when tools are created, updated, or deleted via the API. No additional configuration is needed once the environment variables are set.

### Per-Agent Webhooks (API Endpoints)

#### Get Webhook Configuration

```bash
GET /v1/agents/{agent_id}/webhook
```

**Response:**
```json
{
  "url": "https://your-server.com/webhooks/letta",
  "events": ["agent.run.completed"],
  "enabled": true,
  "has_secret": true
}
```

**Note:** The actual secret is never returned by the API.

#### Update Webhook Configuration

```bash
PUT /v1/agents/{agent_id}/webhook
```

**Request Body:**
```json
{
  "url": "https://your-server.com/webhooks/letta",
  "secret": "new-secret",
  "events": ["agent.run.completed", "agent.step.completed"],
  "enabled": true
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | No | HTTPS URL to receive webhooks. Set to `null` to remove. |
| `secret` | string | No | Shared secret for HMAC signatures. Set to `null` to remove. |
| `events` | array | No | Event types to subscribe to. Empty array = all events. |
| `enabled` | boolean | No | Enable or disable webhook delivery. |

#### Delete Webhook Configuration

```bash
PUT /v1/agents/{agent_id}/webhook
```

**Request Body:**
```json
{
  "url": null,
  "secret": null,
  "enabled": false
}
```

---

## Security

### SSRF Protection

Webhook URLs are validated to prevent Server-Side Request Forgery (SSRF) attacks:

**Blocked:**
- `localhost`, `127.0.0.1`, `0.0.0.0`
- Private IP ranges: `10.x.x.x`, `172.16-31.x.x`, `192.168.x.x`
- Link-local addresses: `169.254.x.x`
- Cloud metadata endpoints: `169.254.169.254`

**Production mode** (when `LETTA_WEBHOOK_STRICT_VALIDATION=true`):
- Only HTTPS URLs allowed
- HTTP URLs are rejected

**Development mode** (default):
- HTTP URLs allowed for local testing
- SSRF protection still enforced

### HMAC Signature Verification

Every webhook request includes an `X-Letta-Signature` header with an HMAC-SHA256 signature of the payload.

**Algorithm:**
```
HMAC-SHA256(webhook_secret, request_body) → hexdigest
```

**Verification Steps:**
1. Extract raw request body (before parsing JSON)
2. Get `X-Letta-Signature` header
3. Compute HMAC-SHA256 of body using your secret
4. Compare signatures using constant-time comparison (`hmac.compare_digest`)

**Python Example:**
```python
import hmac
import hashlib

def verify_webhook(payload_bytes, signature, secret):
    expected = hmac.new(
        secret.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)
```

**Node.js Example:**
```javascript
const crypto = require('crypto');

function verifyWebhook(payload, signature, secret) {
  const expected = crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expected)
  );
}
```

### Headers Sent with Webhooks

Every webhook request includes these headers:

| Header | Description | Example |
|--------|-------------|---------|
| `Content-Type` | Always `application/json` | `application/json` |
| `User-Agent` | Identifies Letta webhook | `Letta-Webhooks/1.0` |
| `X-Letta-Event-Type` | The event type | `agent.run.completed` |
| `X-Letta-Event-Id` | Unique event identifier | `evt-550e8400-...` |
| `X-Letta-Timestamp` | Unix timestamp | `1735761330` |
| `X-Letta-Signature` | HMAC-SHA256 signature | `abc123...` |

---

## Delivery

### Delivery Guarantees

- **At-least-once delivery**: Failed deliveries are retried automatically
- **No ordering guarantee**: Events may arrive out of order
- **Idempotency**: Use `event.id` to deduplicate events

### Retry Behavior

Failed webhook deliveries are retried with exponential backoff:

| Attempt | Delay | Total Elapsed |
|---------|-------|---------------|
| 1 | Immediate | 0s |
| 2 | ~5 seconds | 5s |
| 3 | ~10 seconds | 15s |

**Retry Conditions:**
- HTTP status codes: 408, 429, 500, 502, 503, 504
- Network errors: timeout, connection refused, DNS failure

**No Retry:**
- HTTP 4xx errors (except 408, 429)
- HTTP 2xx success responses

**Max Attempts:** 3 (configurable in settings)

### Timeout

Default webhook timeout: **10 seconds**

Your endpoint should respond quickly. If processing takes longer than the timeout, return `200 OK` immediately and process asynchronously.

### Delivery Persistence

All webhook delivery attempts are persisted to the database with:

- Delivery status (`pending`, `delivered`, `failed`, `retrying`)
- Attempt count and timestamps
- HTTP status code and response time
- Error messages for failed attempts

**Note:** In the current implementation, delivery history is tracked but not yet exposed via API endpoints. Future versions will add endpoints to query delivery logs.

---

## Examples

### Flask Webhook Receiver

```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)
WEBHOOK_SECRET = "your-webhook-secret"

def verify_signature(payload, signature):
    expected = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

@app.route('/webhooks/letta', methods=['POST'])
def handle_webhook():
    # Verify signature
    payload = request.get_data(as_text=True)
    signature = request.headers.get('X-Letta-Signature', '')
    
    if not verify_signature(payload, signature):
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Parse event
    event = request.json
    event_type = event['event_type']
    
    # Handle different event types
    if event_type == 'agent.run.completed':
        agent_id = event['agent_id']
        messages = event['data']['messages']
        print(f"Agent {agent_id} completed run with {len(messages)} messages")
    
    elif event_type == 'agent.step.completed':
        step_id = event['data']['step_id']
        print(f"Step {step_id} completed")
    
    # Return 200 OK quickly
    return jsonify({'received': True}), 200

if __name__ == '__main__':
    app.run(port=5000)
```

### Express.js Webhook Receiver

```javascript
const express = require('express');
const crypto = require('crypto');

const app = express();
const WEBHOOK_SECRET = 'your-webhook-secret';

// Use raw body for signature verification
app.use(express.json({
  verify: (req, res, buf) => {
    req.rawBody = buf.toString('utf8');
  }
}));

function verifySignature(payload, signature) {
  const expected = crypto
    .createHmac('sha256', WEBHOOK_SECRET)
    .update(payload)
    .digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expected)
  );
}

app.post('/webhooks/letta', (req, res) => {
  // Verify signature
  const signature = req.headers['x-letta-signature'] || '';
  if (!verifySignature(req.rawBody, signature)) {
    return res.status(401).json({ error: 'Invalid signature' });
  }
  
  // Parse event
  const event = req.body;
  const eventType = event.event_type;
  
  // Handle different event types
  if (eventType === 'agent.run.completed') {
    const agentId = event.agent_id;
    const messages = event.data.messages;
    console.log(`Agent ${agentId} completed run with ${messages.length} messages`);
  }
  
  // Return 200 OK quickly
  res.json({ received: true });
});

app.listen(5000, () => {
  console.log('Webhook receiver listening on port 5000');
});
```

### Testing with curl

```bash
# Configure webhook
curl -X PUT "http://localhost:8283/v1/agents/agent-123/webhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://webhook.site/your-unique-url",
    "events": ["agent.run.completed"],
    "enabled": true
  }'

# Trigger an event by sending a message
curl -X POST "http://localhost:8283/v1/agents/agent-123/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "text": "Hello!"}]
  }'

# Check webhook was delivered at webhook.site
```

### Handling Multiple Event Types

```python
def handle_webhook_event(event):
    handlers = {
        # Per-agent events
        'agent.run.completed': handle_run_completed,
        'agent.step.completed': handle_step_completed,
        'agent.tool.attached': handle_tool_attached,
        'agent.tool.detached': handle_tool_detached,
        'agent.memory.updated': handle_memory_updated,
        # Global events (tool lifecycle)
        'tool.created': handle_tool_created,
        'tool.updated': handle_tool_updated,
        'tool.deleted': handle_tool_deleted,
    }
    
    handler = handlers.get(event['event_type'])
    if handler:
        handler(event)
    else:
        print(f"Unknown event type: {event['event_type']}")

def handle_run_completed(event):
    agent_id = event['agent_id']
    messages = event['data']['messages']
    print(f"Run completed for {agent_id}: {len(messages)} messages")

def handle_step_completed(event):
    step_id = event['data']['step_id']
    tool_calls = event['data'].get('tool_calls', [])
    print(f"Step {step_id} completed with {len(tool_calls)} tool calls")

def handle_tool_attached(event):
    tool_name = event['data']['tool_name']
    agent_id = event['agent_id']
    print(f"Tool '{tool_name}' attached to agent {agent_id}")

def handle_tool_detached(event):
    tool_name = event['data']['tool_name']
    agent_id = event['agent_id']
    print(f"Tool '{tool_name}' detached from agent {agent_id}")

def handle_memory_updated(event):
    block_label = event['data']['block_label']
    print(f"Memory block '{block_label}' updated")

# Global event handlers (for tool lifecycle)
def handle_tool_created(event):
    tool_id = event['tool_id']
    tool_name = event['data']['tool_name']
    print(f"Tool created: {tool_name} ({tool_id})")
    # Sync to vector database, update cache, etc.

def handle_tool_updated(event):
    tool_id = event['tool_id']
    tool_name = event['data']['tool_name']
    print(f"Tool updated: {tool_name} ({tool_id})")
    # Re-index in vector database, invalidate cache, etc.

def handle_tool_deleted(event):
    tool_id = event['tool_id']
    tool_name = event['data']['tool_name']
    print(f"Tool deleted: {tool_name} ({tool_id})")
    # Remove from vector database, clean up cache, etc.
```

### Tool Sync Service Example

A common use case is syncing tools to an external system (e.g., vector database for search). Instead of polling, use webhooks:

```python
from flask import Flask, request, jsonify
import httpx
import weaviate

app = Flask(__name__)
client = weaviate.Client("http://weaviate:8080")

@app.route('/webhooks/tools', methods=['POST'])
def handle_tool_webhook():
    event = request.json
    event_type = event['event_type']
    tool_id = event['tool_id']
    
    if event_type == 'tool.created':
        # Add tool to vector database
        tool_data = event['data']
        client.data_object.create(
            class_name="Tool",
            data_object={
                "tool_id": tool_id,
                "name": tool_data['tool_name'],
                "description": tool_data['description'],
                "schema": str(tool_data['json_schema']),
            }
        )
        print(f"Added tool {tool_id} to Weaviate")
    
    elif event_type == 'tool.updated':
        # Update tool in vector database
        tool_data = event['data']
        client.data_object.update(
            class_name="Tool",
            uuid=tool_id,
            data_object={
                "name": tool_data['tool_name'],
                "description": tool_data['description'],
                "schema": str(tool_data['json_schema']),
            }
        )
        print(f"Updated tool {tool_id} in Weaviate")
    
    elif event_type == 'tool.deleted':
        # Remove tool from vector database
        client.data_object.delete(class_name="Tool", uuid=tool_id)
        print(f"Deleted tool {tool_id} from Weaviate")
    
    return jsonify({'received': True}), 200

if __name__ == '__main__':
    app.run(port=5000)
```

---

## Troubleshooting

### Webhook Not Firing

**Check webhook configuration:**
```bash
curl "http://localhost:8283/v1/agents/{agent_id}/webhook"
```

Verify:
- `enabled` is `true`
- `url` is set
- `events` includes the event type (or is empty for all events)

**Check agent activity:**
- Ensure the agent is actually executing (send a message via API)
- Currently only `agent.run.completed` events are emitted during `step()` and `stream()`

### Webhooks Failing to Deliver

**Common causes:**

1. **SSRF protection blocking URL:**
   - Error: `"Webhook URL is not allowed"`
   - Solution: Use a public HTTPS endpoint, not localhost/private IPs

2. **Endpoint timeout:**
   - Error: `"Request timeout"`
   - Solution: Respond with 200 OK within 10 seconds

3. **Invalid SSL certificate:**
   - Error: `"SSL certificate verification failed"`
   - Solution: Use a valid SSL certificate

4. **Endpoint returning 4xx errors:**
   - Error: HTTP status in response
   - Solution: Fix your webhook handler to return 2xx

### Signature Verification Failing

**Common issues:**

1. **Wrong payload encoding:**
   - Use raw request body, not parsed JSON
   - Signature is computed on the exact bytes sent

2. **Wrong secret:**
   - Ensure you're using the correct secret
   - Secrets are never returned by the API

3. **Timing attack vulnerability:**
   - Use `hmac.compare_digest()` (Python) or `crypto.timingSafeEqual()` (Node.js)
   - Never use string equality (`==`)

### Enable Debug Logging

To see detailed webhook delivery logs, set environment variable:

```bash
LETTA_LOG_LEVEL=DEBUG
```

Check Docker logs:
```bash
docker logs letta-letta-1 --tail 100 -f
```

Look for log lines containing:
- `Publishing webhook event`
- `Dispatching webhook`
- `Webhook delivery succeeded`
- `Webhook delivery failed`

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LETTA_WEBHOOK_GLOBAL_URL` | `null` | URL for global webhook events (tool.created, etc.) |
| `LETTA_WEBHOOK_GLOBAL_SECRET` | `null` | Secret for global webhook HMAC signatures |
| `LETTA_WEBHOOK_TIMEOUT_SECONDS` | `30` | HTTP request timeout for webhook delivery |
| `LETTA_WEBHOOK_MAX_RETRIES` | `3` | Maximum delivery retry attempts |
| `LETTA_WEBHOOK_REQUIRE_HTTPS` | `false` | Enforce HTTPS-only URLs |
| `LETTA_WEBHOOK_BLOCKED_HOSTS` | `localhost,127.0.0.1,...` | Hosts blocked from receiving webhooks |
| `LETTA_WEBHOOK_ALLOWED_HOSTS` | `null` | If set, only these hosts can receive webhooks |

### Webhook Settings (letta/settings.py)

```python
class WebhookSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="letta_webhook_", extra="ignore")

    timeout_seconds: int = 30
    max_retries: int = 3
    require_https: bool = False
    blocked_hosts: list[str] = ["localhost", "127.0.0.1", "0.0.0.0", "::1", ...]
    allowed_hosts: list[str] | None = None
    
    # Global webhooks (platform-level events)
    global_url: str | None = None
    global_secret: str | None = None
```

---

## Migration from Polling

If you currently poll the Letta API for updates, webhooks provide a more efficient alternative:

### Before (Polling)
```python
import time
while True:
    response = requests.get(f"{LETTA_URL}/agents/{agent_id}/messages")
    messages = response.json()
    # Process new messages
    time.sleep(5)  # Poll every 5 seconds
```

### After (Webhooks)
```python
# Configure webhook once
requests.put(
    f"{LETTA_URL}/agents/{agent_id}/webhook",
    json={
        "url": "https://your-server.com/webhooks",
        "events": ["agent.run.completed"],
        "enabled": True
    }
)

# Receive push notifications
@app.route('/webhooks', methods=['POST'])
def handle_webhook():
    event = request.json
    # Process event immediately
    return {'received': True}, 200
```

**Benefits:**
- No polling overhead
- Real-time notifications
- Lower API load
- Reduced latency

---

## Limitations & Future Work

### Current Implementation Status

| Event Category | Status | Notes |
|----------------|--------|-------|
| `agent.run.completed` | ✅ Implemented | Emitted after agent run finishes |
| `tool.created/updated/deleted` | ✅ Implemented | Global events via `LETTA_WEBHOOK_GLOBAL_URL` |
| `agent.tool.attached/detached` | ✅ Implemented | Per-agent events |
| `agent.step.*` | 🔲 Planned | Step-level events not yet emitted |
| `agent.memory.updated` | 🔲 Planned | Memory change events not yet emitted |
| `tool.execution.*` | 🔲 Planned | Tool execution events not yet emitted |

### Current Limitations

1. **No delivery history API:** Delivery attempts are persisted but not yet exposed via API endpoints.

2. **No test endpoint:** No way to trigger a test webhook delivery to verify configuration.

3. **Step-level events not implemented:** `agent.step.completed` and `agent.step.failed` are defined but not yet emitted.

### Planned Features

- Step-level event emissions (`agent.step.completed`, `agent.step.failed`)
- Memory update events (`agent.memory.updated`)
- Tool execution events (`tool.execution.completed`, `tool.execution.failed`)
- API endpoints for querying webhook delivery history
- Test webhook endpoint for configuration verification
- Webhook delivery dashboard in web UI
- Support for multiple webhook URLs per agent
- Custom retry policies per webhook

---

## Support

For issues or questions:
- GitHub: [letta-ai/letta](https://github.com/letta-ai/letta)
- Documentation: [letta.com/docs](https://letta.com/docs)
