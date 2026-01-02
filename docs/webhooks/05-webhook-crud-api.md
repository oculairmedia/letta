# Issue: Add webhook CRUD API endpoints

**Type:** Enhancement  
**Priority:** Medium  
**Labels:** `enhancement`, `webhooks`, `api`

## Overview
Add REST API endpoints to manage webhook configurations on agents. This allows users to configure, update, test, and monitor webhooks through the API.

## Scope
Create new API endpoints in the agents router to manage webhook settings and view delivery history.

## API Endpoints

### 1. Get Webhook Configuration
**Endpoint:** `GET /v1/agents/{agent_id}/webhook`

**Description:** Retrieve the current webhook configuration for an agent.

**Response Schema:**
```python
class WebhookConfigResponse(LettaBase):
    webhook_url: Optional[HttpUrl]
    webhook_events: List[str]
    webhook_enabled: bool
    # Note: webhook_secret is never returned for security
```

**Response Example:**
```json
{
    "webhook_url": "https://example.com/webhooks/letta",
    "webhook_events": ["agent.step.completed", "agent.tool.attached"],
    "webhook_enabled": true
}
```

**Implementation:**
```python
@router.get("/{agent_id}/webhook", response_model=WebhookConfigResponse)
async def get_webhook_config(
    agent_id: str,
    server: SyncServer = Depends(get_letta_server),
    actor: PydanticUser = Depends(get_current_user),
):
    """Get webhook configuration for an agent"""
    agent = await server.agent_manager.get_agent_by_id_async(agent_id, actor)
    
    return WebhookConfigResponse(
        webhook_url=agent.webhook_url,
        webhook_events=agent.webhook_events or [],
        webhook_enabled=agent.webhook_enabled,
    )
```

---

### 2. Update Webhook Configuration
**Endpoint:** `PATCH /v1/agents/{agent_id}/webhook`

**Description:** Update or create webhook configuration for an agent.

**Request Schema:**
```python
class UpdateWebhookConfigRequest(LettaBase):
    webhook_url: Optional[HttpUrl] = None
    webhook_secret: Optional[str] = None
    webhook_events: Optional[List[str]] = None
    webhook_enabled: Optional[bool] = None
```

**Request Example:**
```json
{
    "webhook_url": "https://example.com/webhooks/letta",
    "webhook_secret": "whsec_abc123...",
    "webhook_events": [
        "agent.step.completed",
        "agent.tool.attached",
        "agent.tool.detached",
        "agent.message.sent"
    ],
    "webhook_enabled": true
}
```

**Response:** Updated `WebhookConfigResponse`

**Implementation:**
```python
@router.patch("/{agent_id}/webhook", response_model=WebhookConfigResponse)
async def update_webhook_config(
    agent_id: str,
    request: UpdateWebhookConfigRequest,
    server: SyncServer = Depends(get_letta_server),
    actor: PydanticUser = Depends(get_current_user),
):
    """Update webhook configuration for an agent"""
    
    # Validate webhook URL if provided
    if request.webhook_url:
        validator = WebhookURLValidator(
            allow_private_ips=server.settings.webhook_allow_private_ips,
            allowed_hosts=server.settings.webhook_allowed_hosts,
            blocked_hosts=server.settings.webhook_blocked_hosts,
        )
        is_valid, error = validator.validate(str(request.webhook_url))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
    
    # Validate event types if provided
    if request.webhook_events:
        valid_events = {
            "agent.step.completed",
            "agent.step.failed",
            "agent.message.sent",
            "agent.tool.attached",
            "agent.tool.detached",
            "agent.run.completed",
            "agent.job.completed",
            "tool.execution.completed",
        }
        invalid_events = set(request.webhook_events) - valid_events
        if invalid_events:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event types: {invalid_events}"
            )
    
    # Update agent
    updated_agent = await server.agent_manager.update_agent_webhook_config_async(
        agent_id=agent_id,
        webhook_url=request.webhook_url,
        webhook_secret=request.webhook_secret,
        webhook_events=request.webhook_events,
        webhook_enabled=request.webhook_enabled,
        actor=actor,
    )
    
    return WebhookConfigResponse(
        webhook_url=updated_agent.webhook_url,
        webhook_events=updated_agent.webhook_events or [],
        webhook_enabled=updated_agent.webhook_enabled,
    )
```

---

### 3. Delete Webhook Configuration
**Endpoint:** `DELETE /v1/agents/{agent_id}/webhook`

**Description:** Remove webhook configuration from an agent.

**Response:** 204 No Content

**Implementation:**
```python
@router.delete("/{agent_id}/webhook", status_code=204)
async def delete_webhook_config(
    agent_id: str,
    server: SyncServer = Depends(get_letta_server),
    actor: PydanticUser = Depends(get_current_user),
):
    """Delete webhook configuration for an agent"""
    await server.agent_manager.delete_agent_webhook_config_async(
        agent_id=agent_id,
        actor=actor,
    )
```

---

### 4. Test Webhook Endpoint
**Endpoint:** `POST /v1/agents/{agent_id}/webhook/test`

**Description:** Send a test webhook event to verify the endpoint is working.

**Request Schema:**
```python
class TestWebhookRequest(LettaBase):
    event_type: str = "test.webhook"
```

**Response Schema:**
```python
class TestWebhookResponse(LettaBase):
    success: bool
    status_code: Optional[int]
    error: Optional[str]
    response_time_ms: float
```

**Response Example:**
```json
{
    "success": true,
    "status_code": 200,
    "error": null,
    "response_time_ms": 145.3
}
```

**Implementation:**
```python
@router.post("/{agent_id}/webhook/test", response_model=TestWebhookResponse)
async def test_webhook(
    agent_id: str,
    request: TestWebhookRequest,
    server: SyncServer = Depends(get_letta_server),
    actor: PydanticUser = Depends(get_current_user),
):
    """Test webhook delivery to verify endpoint is working"""
    agent = await server.agent_manager.get_agent_by_id_async(agent_id, actor)
    
    if not agent.webhook_url:
        raise HTTPException(status_code=400, detail="Agent has no webhook configured")
    
    # Send test webhook
    import time
    start = time.time()
    
    delivery = await server.webhook_manager._dispatch_webhook_async(
        webhook_url=agent.webhook_url,
        event=WebhookEvent(
            event_id=f"test_{uuid.uuid4()}",
            event_type=request.event_type,
            timestamp=datetime.now(timezone.utc),
            agent_id=agent_id,
            organization_id=agent.organization_id,
            data={"test": True, "message": "This is a test webhook from Letta"},
        ),
        secret=agent.webhook_secret,
    )
    
    response_time = (time.time() - start) * 1000
    
    return TestWebhookResponse(
        success=delivery.status_code is not None and 200 <= delivery.status_code < 300,
        status_code=delivery.status_code,
        error=delivery.error_message,
        response_time_ms=response_time,
    )
```

---

### 5. List Webhook Deliveries
**Endpoint:** `GET /v1/agents/{agent_id}/webhook/deliveries`

**Description:** List recent webhook delivery attempts for debugging.

**Query Parameters:**
- `limit` (int, default=50, max=100): Number of deliveries to return
- `event_type` (str, optional): Filter by event type

**Response Schema:**
```python
class WebhookDeliveryListResponse(LettaBase):
    deliveries: List[WebhookDelivery]
    total: int
```

**Implementation:**
```python
@router.get("/{agent_id}/webhook/deliveries", response_model=WebhookDeliveryListResponse)
async def list_webhook_deliveries(
    agent_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    event_type: Optional[str] = None,
    server: SyncServer = Depends(get_letta_server),
    actor: PydanticUser = Depends(get_current_user),
):
    """List recent webhook delivery attempts for an agent"""
    deliveries = await server.webhook_manager.list_deliveries_async(
        agent_id=agent_id,
        limit=limit,
        event_type=event_type,
        actor=actor,
    )
    
    return WebhookDeliveryListResponse(
        deliveries=deliveries,
        total=len(deliveries),
    )
```

---

### 6. Retry Failed Webhook
**Endpoint:** `POST /v1/agents/{agent_id}/webhook/deliveries/{delivery_id}/retry`

**Description:** Manually retry a failed webhook delivery.

**Response:** Updated `WebhookDelivery`

**Implementation:**
```python
@router.post(
    "/{agent_id}/webhook/deliveries/{delivery_id}/retry",
    response_model=WebhookDelivery
)
async def retry_webhook_delivery(
    agent_id: str,
    delivery_id: str,
    server: SyncServer = Depends(get_letta_server),
    actor: PydanticUser = Depends(get_current_user),
):
    """Manually retry a failed webhook delivery"""
    delivery = await server.webhook_manager.retry_delivery_async(
        delivery_id=delivery_id,
        actor=actor,
    )
    return delivery
```

---

## AgentManager Methods to Add

**File:** `letta/services/agent_manager.py`

```python
async def update_agent_webhook_config_async(
    self,
    agent_id: str,
    webhook_url: Optional[str] = None,
    webhook_secret: Optional[str] = None,
    webhook_events: Optional[List[str]] = None,
    webhook_enabled: Optional[bool] = None,
    actor: PydanticUser,
) -> AgentModel:
    """Update webhook configuration for an agent"""
    async with self.session_maker() as session:
        agent = await self.get_agent_by_id_async(agent_id, actor)
        
        if webhook_url is not None:
            agent.webhook_url = webhook_url
        if webhook_secret is not None:
            agent.webhook_secret = webhook_secret
        if webhook_events is not None:
            agent.webhook_events = webhook_events
        if webhook_enabled is not None:
            agent.webhook_enabled = webhook_enabled
        
        await session.commit()
        return agent

async def delete_agent_webhook_config_async(
    self,
    agent_id: str,
    actor: PydanticUser,
) -> None:
    """Delete webhook configuration from an agent"""
    async with self.session_maker() as session:
        agent = await self.get_agent_by_id_async(agent_id, actor)
        
        agent.webhook_url = None
        agent.webhook_secret = None
        agent.webhook_events = []
        agent.webhook_enabled = False
        
        await session.commit()
```

## Acceptance Criteria
- [ ] All 6 endpoints implemented and tested
- [ ] Webhook URL validation integrated (SSRF protection)
- [ ] Event type validation ensures only valid events accepted
- [ ] Webhook secret never returned in API responses
- [ ] Test endpoint works and returns response time
- [ ] Delivery history endpoint paginated and filterable
- [ ] Retry endpoint allows manual retry of failed deliveries
- [ ] OpenAPI documentation generated for all endpoints
- [ ] Integration tests for all CRUD operations

## Security Considerations
- Webhook secrets must NEVER be returned in GET responses
- URL validation must block private IPs in production
- Rate limiting on test endpoint (max 10 tests/minute per agent)
- Delivery history should only show last 1000 deliveries (to prevent unbounded growth)

## Files to Modify/Create
- `letta/server/rest_api/routers/v1/agents.py` (add webhook routes)
- `letta/services/agent_manager.py` (add webhook config methods)
- `letta/services/webhook_manager.py` (add `list_deliveries_async`, `retry_delivery_async`)
- `tests/test_webhook_api.py` (create)

## Related Issues
- `01-webhook-module-foundation.md` - WebhookManager methods
- `02-orm-model-webhook-fields.md` - Agent webhook fields
- `03-webhook-settings-config.md` - URL validation utility

## OpenAPI Documentation
All endpoints should have comprehensive OpenAPI docs with:
- Description of purpose
- Request/response schemas
- Example payloads
- Error codes and meanings
