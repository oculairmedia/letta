# Phase 2: Redis Job Queue Implementation for Letta

## Overview
Successfully implemented a comprehensive Redis Queue (RQ) system for Letta, providing asynchronous job processing for webhooks, LLM batch operations, and background tasks.

## Key Components Implemented

### 1. **Redis Job Queue Module** (`redis_job_queue.py`)
- **LettaJobQueue** class with priority-based queue management
- Support for CRITICAL, HIGH, DEFAULT, and LOW priority jobs
- Async version for non-blocking operations
- Job scheduling with RQ Scheduler
- Comprehensive error handling and retry logic

### 2. **Webhook Worker** (`webhook_worker.py`)
- Asynchronous webhook callback processing
- Automatic retry with exponential backoff
- HTTP timeout handling
- Response status tracking
- Integration with Letta job status updates

### 3. **LLM Batch Worker** (`llm_batch_worker.py`)
- Efficient batch processing for LLM operations
- Message grouping for optimal batching
- Progress tracking and reporting
- Concurrent processing support
- Error isolation per message group

### 4. **Job Manager Integration**
- Modified `job_manager.py` to use Redis Queue
- Automatic fallback to synchronous processing
- Environment variable control (`LETTA_USE_REDIS_QUEUE`)
- Maintains backward compatibility

### 5. **Worker Infrastructure**
- Dedicated worker container in Docker Compose
- Multiple workers for different priority levels
- RQ Scheduler for delayed/scheduled jobs
- Auto-scaling based on queue depth

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Letta Server  │────▶│ Redis Queue  │────▶│  RQ Workers     │
│                 │     │              │     │                 │
│ - Job Manager   │     │ - Critical   │     │ - 1x Critical   │
│ - Webhook APIs  │     │ - High       │     │ - 2x High       │
│ - Batch APIs    │     │ - Default    │     │ - 4x Default    │
│                 │     │ - Low        │     │ - Scheduler     │
└─────────────────┘     └──────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   Results    │
                        │              │
                        │ - Success    │
                        │ - Failed     │
                        │ - Retrying   │
                        └──────────────┘
```

## Usage Examples

### 1. Webhook Callbacks
When a Letta job completes, webhooks are now processed asynchronously:
```python
# Automatically handled by job_manager when job.callback_url is set
# Webhook is queued with retry logic
# Status code 202 (Accepted) returned immediately
```

### 2. LLM Batch Processing
```python
from letta.jobs.redis_job_queue import get_job_queue, JobPriority

job_queue = get_job_queue()
job_id = job_queue.enqueue_llm_batch_processing(
    batch_id="batch-123",
    llm_config={"model": "gpt-4", "temperature": 0.7},
    messages=[...],
    priority=JobPriority.HIGH,
)
```

### 3. Scheduled Jobs
```python
from datetime import datetime, timedelta

job_queue.schedule_job(
    func='letta.jobs.maintenance.cleanup_old_data',
    scheduled_time=datetime.utcnow() + timedelta(hours=1),
    interval=3600,  # Repeat every hour
    priority=JobPriority.LOW,
)
```

## Performance Benefits

1. **Non-blocking Operations**: Webhook callbacks no longer block request processing
2. **Parallel Processing**: Multiple workers handle jobs concurrently
3. **Retry Logic**: Failed webhooks automatically retry with backoff
4. **Queue Management**: Priority-based processing ensures critical jobs run first
5. **Resource Efficiency**: Workers scale independently from the main server

## Configuration

### Environment Variables
- `LETTA_REDIS_URL`: Redis connection URL (default: `redis://redis:6379/0`)
- `LETTA_USE_REDIS_QUEUE`: Enable/disable Redis Queue (default: `true`)
- `LETTA_WORKER_COUNT`: Number of default priority workers (default: `4`)
- `LETTA_WORKER_LOG_LEVEL`: Worker logging level (default: `INFO`)

### Docker Compose Configuration
```yaml
letta-workers:
  image: letta/letta:latest
  depends_on:
    redis:
      condition: service_healthy
  environment:
    - LETTA_REDIS_URL=redis://redis:6379/0
    - LETTA_WORKER_COUNT=4
  volumes:
    # Mount worker modules
  command: /app/start_workers.sh
```

## Monitoring

### Queue Statistics
```python
stats = job_queue.get_queue_stats()
# Returns pending, started, finished, failed counts per priority
```

### Redis CLI Monitoring
```bash
# Monitor job activity
docker exec letta-redis-1 valkey-cli MONITOR

# Check queue lengths
docker exec letta-redis-1 valkey-cli LLEN rq:queue:letta:high

# View job details
docker exec letta-redis-1 valkey-cli HGETALL rq:job:<job_id>
```

### Worker Logs
```bash
# View worker activity
docker logs letta-letta-workers-1 -f

# Check specific job processing
docker logs letta-letta-workers-1 | grep "webhook_"
```

## Testing

Run the comprehensive test suite:
```bash
docker exec letta-letta-1 python /tmp/test_redis_queue.py
```

This tests:
- Queue initialization
- Webhook callback enqueueing
- LLM batch job processing
- Job scheduling
- Worker availability
- End-to-end processing

## Future Enhancements

1. **Dynamic Worker Scaling**: Auto-scale workers based on queue depth
2. **Dead Letter Queue**: Handle permanently failed jobs
3. **Job Chaining**: Support complex workflows with dependencies
4. **Metrics Export**: Prometheus/Grafana integration
5. **Priority Tuning**: ML-based priority assignment

## Troubleshooting

### Workers Not Processing
1. Check worker container is running: `docker ps | grep letta-workers`
2. Verify Redis connection: `docker exec letta-letta-1 python -c "import redis; r = redis.from_url('redis://redis:6379/0'); print(r.ping())"`
3. Check worker logs: `docker logs letta-letta-workers-1`

### Jobs Stuck in Queue
1. Check queue status: `docker exec letta-letta-1 python -c "from letta.jobs.redis_job_queue import get_job_queue; print(get_job_queue().get_queue_stats())"`
2. Verify workers are connected: `docker exec letta-redis-1 valkey-cli CLIENT LIST`
3. Check for failed jobs: `docker exec letta-redis-1 valkey-cli LRANGE rq:failed:letta:high 0 -1`

### Performance Issues
1. Increase worker count: Set `LETTA_WORKER_COUNT=8`
2. Add more high-priority workers
3. Check Redis memory: `docker exec letta-redis-1 valkey-cli INFO memory`