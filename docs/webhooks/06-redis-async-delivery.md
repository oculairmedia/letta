# Issue: Add Redis queue for async webhook delivery

**Type:** Enhancement  
**Priority:** High  
**Labels:** `enhancement`, `webhooks`, `redis`, `async`

## Overview
Implement a Redis-based queue system for asynchronous webhook delivery. This prevents webhook dispatch from blocking agent execution and provides reliable delivery with retries.

## Background
Your fork already has Redis integration (commit e8ef06e6a). This issue extends that infrastructure to support webhook event queuing.

## Architecture

### Event Flow
```
1. Agent execution completes step
2. webhook_manager.publish_event() → writes to Redis stream (non-blocking)
3. Background worker reads from Redis stream
4. Worker dispatches webhook HTTP request (with retries)
5. Worker updates delivery status in database
```

### Benefits
- Agent execution never blocked by webhook delivery
- Survives server restarts (events persisted in Redis)
- Natural rate limiting and backpressure
- Easy to scale (multiple workers)

## Implementation

### 1. Redis Stream Publisher

**File:** `letta/services/webhook_manager.py`

```python
class WebhookManager:
    def __init__(
        self,
        settings: Settings,
        redis_client: Optional[AsyncRedisClient] = None,
    ):
        self.settings = settings
        self.redis_client = redis_client
        self.stream_name = settings.webhook_redis_stream_name
        self.use_redis = settings.webhook_use_redis_queue and redis_client is not None
    
    async def publish_event(
        self,
        agent_id: str,
        event_type: str,
        payload: dict,
    ) -> None:
        """
        Publish webhook event for delivery.
        If Redis is enabled, writes to stream (async).
        Otherwise, dispatches immediately (fire-and-forget).
        """
        # Check if agent has webhook configured
        agent = await self._get_agent_webhook_config(agent_id)
        if not agent or not agent.webhook_enabled:
            return
        
        # Check if event type is enabled for this agent
        if agent.webhook_events and event_type not in agent.webhook_events:
            return
        
        # Create event
        event = WebhookEvent(
            event_id=f"evt_{uuid.uuid4().hex}",
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            agent_id=agent_id,
            organization_id=agent.organization_id,
            data=payload,
        )
        
        if self.use_redis:
            # Write to Redis stream (async, non-blocking)
            await self._publish_to_redis_stream(event, agent)
        else:
            # Fire-and-forget immediate dispatch
            asyncio.create_task(self._dispatch_webhook_async(
                webhook_url=agent.webhook_url,
                event=event,
                secret=agent.webhook_secret,
            ))
    
    async def _publish_to_redis_stream(
        self,
        event: WebhookEvent,
        agent: AgentModel,
    ) -> None:
        """Write webhook event to Redis stream"""
        message = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "agent_id": event.agent_id,
            "organization_id": event.organization_id,
            "webhook_url": agent.webhook_url,
            "webhook_secret": agent.webhook_secret,
            "payload": json.dumps(event.data),
        }
        
        await self.redis_client.xadd(
            stream_name=self.stream_name,
            fields=message,
            maxlen=10000,  # Keep last 10k events
        )
```

### 2. Redis Stream Consumer (Background Worker)

**File:** `letta/services/webhook_worker.py` (new file)

```python
import asyncio
import logging
from typing import Optional
from letta.data_sources.redis_client import AsyncRedisClient
from letta.services.webhook_manager import WebhookManager
from letta.settings import Settings

logger = logging.getLogger(__name__)

class WebhookWorker:
    """Background worker that consumes webhook events from Redis and delivers them"""
    
    def __init__(
        self,
        settings: Settings,
        redis_client: AsyncRedisClient,
        webhook_manager: WebhookManager,
    ):
        self.settings = settings
        self.redis_client = redis_client
        self.webhook_manager = webhook_manager
        self.stream_name = settings.webhook_redis_stream_name
        self.consumer_group = "webhook_workers"
        self.consumer_name = f"worker_{os.getpid()}"
        self.running = False
    
    async def start(self):
        """Start the worker (blocking)"""
        self.running = True
        
        # Create consumer group (idempotent)
        try:
            await self.redis_client.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id='0',
                mkstream=True,
            )
        except Exception as e:
            # Group already exists, that's fine
            logger.debug(f"Consumer group already exists: {e}")
        
        logger.info(f"Webhook worker {self.consumer_name} started")
        
        # Main processing loop
        while self.running:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(f"Error in webhook worker: {e}", exc_info=True)
                await asyncio.sleep(1)  # Back off on error
    
    async def stop(self):
        """Stop the worker"""
        self.running = False
    
    async def _process_batch(self):
        """Process a batch of webhook events"""
        # Read batch from stream
        batch = await self.redis_client.xreadgroup(
            groupname=self.consumer_group,
            consumername=self.consumer_name,
            streams={self.stream_name: '>'},
            count=self.settings.webhook_worker_batch_size,
            block=5000,  # Block for 5 seconds if no messages
        )
        
        if not batch:
            return
        
        # Process each event
        for stream_name, messages in batch:
            for message_id, fields in messages:
                try:
                    await self._process_event(message_id, fields)
                    
                    # ACK message (delivery successful or gave up)
                    await self.redis_client.xack(
                        name=self.stream_name,
                        groupname=self.consumer_group,
                        id=message_id,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to process webhook event {message_id}: {e}",
                        exc_info=True
                    )
                    # Don't ACK - will be retried by another worker
    
    async def _process_event(self, message_id: str, fields: dict):
        """Process a single webhook event"""
        event = WebhookEvent(
            event_id=fields["event_id"],
            event_type=fields["event_type"],
            timestamp=datetime.fromisoformat(fields["timestamp"]),
            agent_id=fields["agent_id"],
            organization_id=fields["organization_id"],
            data=json.loads(fields["payload"]),
        )
        
        # Dispatch webhook
        delivery = await self.webhook_manager._dispatch_webhook_async(
            webhook_url=fields["webhook_url"],
            event=event,
            secret=fields.get("webhook_secret"),
        )
        
        # Store delivery result
        await self.webhook_manager._store_delivery_result(delivery)
```

### 3. Worker Startup Integration

**File:** `letta/server/server.py`

```python
class SyncServer:
    def __init__(self, ...):
        # ... existing init ...
        
        # Initialize webhook system
        self.webhook_manager = WebhookManager(
            settings=self.settings,
            redis_client=self.redis_client,
        )
        
        # Start webhook worker if Redis enabled
        if self.settings.webhook_use_redis_queue and self.redis_client:
            self.webhook_worker = WebhookWorker(
                settings=self.settings,
                redis_client=self.redis_client,
                webhook_manager=self.webhook_manager,
            )
            # Start worker in background
            asyncio.create_task(self.webhook_worker.start())
```

**Graceful Shutdown:**
```python
async def shutdown(self):
    """Shutdown server and background workers"""
    if hasattr(self, 'webhook_worker'):
        await self.webhook_worker.stop()
    # ... existing shutdown ...
```

### 4. Redis Client Extensions

**File:** `letta/data_sources/redis_client.py`

Add stream operations if not already present:
```python
class AsyncRedisClient:
    # ... existing methods ...
    
    async def xadd(
        self,
        stream_name: str,
        fields: dict,
        maxlen: Optional[int] = None,
    ) -> str:
        """Add entry to Redis stream"""
        return await self.redis.xadd(
            name=stream_name,
            fields=fields,
            maxlen=maxlen,
            approximate=True if maxlen else False,
        )
    
    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict,
        count: Optional[int] = None,
        block: Optional[int] = None,
    ):
        """Read from Redis stream as consumer group"""
        return await self.redis.xreadgroup(
            groupname=groupname,
            consumername=consumername,
            streams=streams,
            count=count,
            block=block,
        )
    
    async def xack(self, name: str, groupname: str, id: str):
        """Acknowledge message processed"""
        return await self.redis.xack(name, groupname, id)
    
    async def xgroup_create(
        self,
        name: str,
        groupname: str,
        id: str = '0',
        mkstream: bool = False,
    ):
        """Create consumer group"""
        return await self.redis.xgroup_create(
            name=name,
            groupname=groupname,
            id=id,
            mkstream=mkstream,
        )
```

## Retry Logic

### Exponential Backoff in Worker
```python
async def _dispatch_webhook_async(
    self,
    webhook_url: str,
    event: WebhookEvent,
    secret: Optional[str] = None,
) -> WebhookDelivery:
    """Dispatch webhook with retry logic"""
    
    delivery = WebhookDelivery(
        id=f"del_{uuid.uuid4().hex}",
        agent_id=event.agent_id,
        event_id=event.event_id,
        event_type=event.event_type,
        webhook_url=webhook_url,
        attempt_count=0,
    )
    
    for attempt in range(self.settings.webhook_max_retries + 1):
        try:
            delivery.attempt_count = attempt + 1
            
            # Generate signature
            payload_json = event.model_dump_json()
            headers = {
                "Content-Type": "application/json",
                "X-Letta-Event-Type": event.event_type,
                "X-Letta-Event-Id": event.event_id,
                "X-Letta-Timestamp": str(int(event.timestamp.timestamp())),
                "User-Agent": "Letta-Webhooks/1.0",
            }
            
            if secret:
                signature = self._generate_signature(payload_json, secret)
                headers["X-Letta-Signature"] = f"sha256={signature}"
            
            # HTTP POST
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    content=payload_json,
                    headers=headers,
                    timeout=self.settings.webhook_timeout_seconds,
                )
            
            delivery.status_code = response.status_code
            
            # Success!
            if 200 <= response.status_code < 300:
                delivery.delivered_at = datetime.now(timezone.utc)
                return delivery
            
            # Retryable error
            delivery.error_message = f"HTTP {response.status_code}"
            
        except Exception as e:
            delivery.error_message = str(e)
        
        # Backoff before retry
        if attempt < self.settings.webhook_max_retries:
            backoff = self.settings.webhook_retry_backoff_base ** attempt
            await asyncio.sleep(backoff)
    
    # All retries exhausted
    return delivery
```

## Monitoring & Observability

### Metrics to Track
- `webhook.events.published` - Events written to Redis
- `webhook.events.processed` - Events processed by workers
- `webhook.delivery.success` - Successful deliveries
- `webhook.delivery.failure` - Failed deliveries after all retries
- `webhook.delivery.latency` - Time from publish to delivery
- `webhook.queue.depth` - Pending events in Redis stream

### Logging
- Log every webhook dispatch with event_id, agent_id, status
- Log retry attempts with backoff time
- Log permanent failures after max retries

## Acceptance Criteria
- [ ] Redis stream publisher implemented in WebhookManager
- [ ] Background worker consumes events and dispatches webhooks
- [ ] Worker handles retries with exponential backoff
- [ ] Worker ACKs messages only after successful delivery or max retries
- [ ] Graceful shutdown stops worker and processes pending events
- [ ] Redis client extended with stream operations
- [ ] Metrics collected for monitoring
- [ ] Integration tests verify end-to-end flow (Redis → worker → HTTP)
- [ ] Load test shows system can handle 1000+ webhooks/sec

## Deployment Considerations

### Multiple Workers
- Consumer groups allow multiple workers for horizontal scaling
- Each worker processes different events (load balanced by Redis)

### Redis Persistence
- Use Redis persistence (AOF or RDB) to survive Redis restarts
- Set `maxlen` to prevent unbounded stream growth

### Dead Letter Queue
- After max retries, move failed events to dead letter stream
- Allows manual inspection and replay

## Files to Create/Modify
- `letta/services/webhook_manager.py` (modify - add Redis publishing)
- `letta/services/webhook_worker.py` (create)
- `letta/data_sources/redis_client.py` (modify - add stream operations)
- `letta/server/server.py` (modify - start worker)
- `tests/test_webhook_worker.py` (create)
- `tests/test_webhook_redis.py` (create)

## Related Issues
- `01-webhook-module-foundation.md` - WebhookManager core
- `03-webhook-settings-config.md` - Redis queue settings
- `04-agent-execution-integration.md` - publish_event() calls

## References
- Existing Redis client: `letta/data_sources/redis_client.py`
- Redis stream manager: `letta/server/rest_api/redis_stream_manager.py`
- Redis settings: `letta/settings.py:258-259`
- Fork's Redis integration commit: e8ef06e6a
