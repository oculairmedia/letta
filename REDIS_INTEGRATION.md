# Redis Integration for Letta

This document describes the Redis integration work implemented for improved caching and job queue management in Letta.

## Overview

The Redis integration provides:
- **Agent State Caching**: Faster agent state retrieval and persistence
- **Message Caching**: Improved message handling performance
- **Job Queue Management**: Distributed job processing with Redis queues
- **Webhook Processing**: Asynchronous webhook handling
- **LLM Batch Processing**: Efficient batch processing of language model requests

## Components

### Core Redis Modules

Located in `redis_integration/`:

1. **`redis_cache.py`**: Core Redis caching implementation
2. **`redis_job_queue.py`**: Redis-based job queue system
3. **`agent_state_cache.py`**: Agent state caching layer
4. **`webhook_worker.py`**: Webhook processing worker
5. **`llm_batch_worker.py`**: LLM batch processing worker

### Support Files

- **`start_workers.sh`**: Script to start Redis workers
- **`test_redis_cache.py`**: Redis cache testing
- **`test_redis_queue.py`**: Redis queue testing
- **`redis_queue_documentation.md`**: Detailed queue documentation

## Deployment

### Docker Compose Setup

The Redis integration is configured in `docker-compose.redis.yml`:

```yaml
services:
  redis:
    image: valkey/valkey:8-alpine
    ports:
      - 6380:6379
    volumes:
      - ${HOME}/.letta/.persist/redis:/data
    command: valkey-server --save 60 1 --loglevel warning --maxmemory 2gb --maxmemory-policy allkeys-lru

  letta:
    environment:
      - LETTA_REDIS_URL=redis://redis:6379/0
    volumes:
      - ./redis_integration/redis_cache.py:/app/letta/cache/redis_cache.py

  letta-workers:
    image: letta/letta:latest
    environment:
      - LETTA_REDIS_URL=redis://redis:6379/0
      - LETTA_WORKER_COUNT=4
    volumes:
      - ./redis_integration/redis_job_queue.py:/app/letta/jobs/redis_job_queue.py
      - ./redis_integration/webhook_worker.py:/app/letta/jobs/webhook_worker.py
      - ./redis_integration/llm_batch_worker.py:/app/letta/jobs/llm_batch_worker.py
      - ./redis_integration/start_workers.sh:/app/start_workers.sh
    command: /app/start_workers.sh
```

### Environment Variables

Required environment variables:
- `LETTA_REDIS_URL`: Redis connection URL (default: `redis://localhost:6379/0`)
- `LETTA_WORKER_COUNT`: Number of worker processes (default: 4)
- `LETTA_WORKER_LOG_LEVEL`: Worker logging level (default: INFO)

## Features

### 1. Agent State Caching
- Caches agent states in Redis for faster retrieval
- Implements cache invalidation strategies
- Reduces database load for frequently accessed agents

### 2. Message Caching
- Caches recent messages for improved performance
- Implements TTL-based cache expiration
- Reduces message retrieval latency

### 3. Job Queue System
- Redis-based distributed job queue
- Supports job priorities and retries
- Handles webhook processing asynchronously
- Manages LLM batch processing

### 4. Worker System
- Multiple worker processes for parallel job processing
- Automatic job retry on failure
- Health monitoring and logging
- Graceful shutdown handling

## Performance Benefits

- **Reduced Database Load**: Caching frequently accessed data
- **Improved Response Times**: Faster agent state and message retrieval
- **Scalable Processing**: Distributed job processing across workers
- **Asynchronous Operations**: Non-blocking webhook and batch processing

## Testing

Run the test suite:
```bash
python redis_integration/test_redis_cache.py
python redis_integration/test_redis_queue.py
```

## Monitoring

Monitor Redis performance:
```bash
# Connect to Redis CLI
docker exec -it letta-redis-1 valkey-cli

# Monitor Redis operations
MONITOR

# Check memory usage
INFO memory
```

## Production Considerations

1. **Memory Management**: Configure appropriate `maxmemory` settings
2. **Persistence**: Enable Redis persistence for production
3. **Monitoring**: Implement Redis monitoring and alerting
4. **Backup**: Regular Redis data backups
5. **Scaling**: Consider Redis Cluster for high availability

## Implementation Status

✅ **Completed**:
- Redis caching infrastructure
- Job queue system
- Worker processes
- Docker integration
- Basic testing

🔄 **In Progress**:
- Performance optimization
- Advanced monitoring
- Production hardening

📋 **Planned**:
- Redis Cluster support
- Advanced caching strategies
- Comprehensive benchmarking