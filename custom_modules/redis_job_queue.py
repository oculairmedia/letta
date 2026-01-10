"""
Redis Queue (RQ) implementation for Letta background job processing.
Provides asynchronous job execution for webhooks, LLM batches, and other background tasks.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

import redis
from redis import asyncio as aioredis
from rq import Queue, Worker, get_current_job
from rq.decorators import job
from rq.job import Job as RQJob
from rq.registry import FinishedJobRegistry, FailedJobRegistry, StartedJobRegistry
from rq_scheduler import Scheduler

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels for queue management."""
    CRITICAL = "critical"
    HIGH = "high"
    DEFAULT = "default"
    LOW = "low"
    
    @property
    def queue_name(self):
        return f"letta:{self.value}"


class LettaJobQueue:
    """Redis Queue implementation for Letta background jobs."""
    
    def __init__(
        self,
        redis_url: str = None,
        default_timeout: int = 300,  # 5 minutes
        result_ttl: int = 3600,  # 1 hour
        failure_ttl: int = 86400,  # 24 hours
    ):
        """
        Initialize the job queue system.
        
        Args:
            redis_url: Redis connection URL
            default_timeout: Default job timeout in seconds
            result_ttl: How long to keep successful job results
            failure_ttl: How long to keep failed job information
        """
        self.redis_url = redis_url or os.getenv('LETTA_REDIS_URL', 'redis://localhost:6379/0')
        self.redis_conn = redis.from_url(self.redis_url)
        self.default_timeout = default_timeout
        self.result_ttl = result_ttl
        self.failure_ttl = failure_ttl
        
        # Initialize queues for different priorities
        self.queues = {
            priority: Queue(
                priority.queue_name,
                connection=self.redis_conn,
                default_timeout=self.default_timeout
            )
            for priority in JobPriority
        }
        
        # Default queue
        self.default_queue = self.queues[JobPriority.DEFAULT]
        
        # Scheduler for delayed jobs
        self.scheduler = Scheduler(connection=self.redis_conn)
    
    def enqueue_webhook_callback(
        self,
        job_id: str,
        callback_url: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.DEFAULT,
        retry_count: int = 3,
        retry_delay: int = 60,
    ) -> str:
        """
        Enqueue a webhook callback for asynchronous processing.
        
        Args:
            job_id: Letta job ID
            callback_url: URL to send the callback to
            payload: JSON payload to send
            priority: Job priority level
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            RQ job ID
        """
        queue = self.queues[priority]
        
        rq_job = queue.enqueue(
            'letta.jobs.webhook_worker.send_webhook_callback',
            args=(job_id, callback_url, payload),
            job_id=f"webhook_{job_id}_{uuid4().hex[:8]}",
            result_ttl=self.result_ttl,
            failure_ttl=self.failure_ttl,
            meta={
                'letta_job_id': job_id,
                'job_type': 'webhook_callback',
                'retry_count': retry_count,
                'retry_delay': retry_delay,
                'attempt': 0,
            }
        )
        
        logger.info(f"Enqueued webhook callback job {rq_job.id} for Letta job {job_id}")
        return rq_job.id
    
    def enqueue_llm_batch_processing(
        self,
        batch_id: str,
        llm_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        priority: JobPriority = JobPriority.HIGH,
    ) -> str:
        """
        Enqueue LLM batch processing job.
        
        Args:
            batch_id: Batch identifier
            llm_config: LLM configuration
            messages: List of messages to process
            priority: Job priority level
            
        Returns:
            RQ job ID
        """
        queue = self.queues[priority]
        
        rq_job = queue.enqueue(
            'letta.jobs.llm_batch_worker.process_llm_batch',
            args=(batch_id, llm_config, messages),
            job_id=f"llm_batch_{batch_id}",
            result_ttl=self.result_ttl * 2,  # Keep results longer for batch jobs
            failure_ttl=self.failure_ttl,
            job_timeout=1800,  # 30 minutes for batch processing
            meta={
                'batch_id': batch_id,
                'job_type': 'llm_batch',
                'message_count': len(messages),
            }
        )
        
        logger.info(f"Enqueued LLM batch job {rq_job.id} with {len(messages)} messages")
        return rq_job.id
    
    def enqueue_agent_step_processing(
        self,
        agent_id: str,
        step_id: str,
        step_data: Dict[str, Any],
        priority: JobPriority = JobPriority.HIGH,
        depends_on: Optional[str] = None,
    ) -> str:
        """
        Enqueue agent step processing with optional dependency.
        
        Args:
            agent_id: Agent identifier
            step_id: Step identifier
            step_data: Step processing data
            priority: Job priority level
            depends_on: RQ job ID to depend on
            
        Returns:
            RQ job ID
        """
        queue = self.queues[priority]
        
        kwargs = {
            'args': (agent_id, step_id, step_data),
            'job_id': f"agent_step_{agent_id}_{step_id}",
            'result_ttl': self.result_ttl,
            'failure_ttl': self.failure_ttl,
            'meta': {
                'agent_id': agent_id,
                'step_id': step_id,
                'job_type': 'agent_step',
            }
        }
        
        if depends_on:
            kwargs['depends_on'] = depends_on
        
        rq_job = queue.enqueue(
            'letta.jobs.agent_step_worker.process_agent_step',
            **kwargs
        )
        
        logger.info(f"Enqueued agent step job {rq_job.id} for agent {agent_id}")
        return rq_job.id
    
    def schedule_job(
        self,
        func: Union[str, Callable],
        args: tuple = (),
        kwargs: dict = None,
        scheduled_time: datetime = None,
        interval: Optional[int] = None,
        priority: JobPriority = JobPriority.DEFAULT,
    ) -> str:
        """
        Schedule a job for future execution.
        
        Args:
            func: Function to execute (string import path or callable)
            args: Function arguments
            kwargs: Function keyword arguments
            scheduled_time: When to execute (defaults to now)
            interval: Repeat interval in seconds (for recurring jobs)
            priority: Job priority level
            
        Returns:
            Scheduled job ID
        """
        queue = self.queues[priority]
        scheduled_time = scheduled_time or datetime.utcnow()
        
        job = self.scheduler.schedule(
            scheduled_time=scheduled_time,
            func=func,
            args=args,
            kwargs=kwargs or {},
            interval=interval,
            repeat=interval is not None,
            queue_name=queue.name,
        )
        
        logger.info(f"Scheduled job {job.id} for {scheduled_time}")
        return job.id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a job.
        
        Args:
            job_id: RQ job ID
            
        Returns:
            Job status information
        """
        try:
            job = RQJob.fetch(job_id, connection=self.redis_conn)
            
            return {
                'id': job.id,
                'status': job.get_status(),
                'created_at': job.created_at,
                'started_at': job.started_at,
                'ended_at': job.ended_at,
                'result': job.result,
                'exc_info': job.exc_info,
                'meta': job.meta,
            }
        except Exception as e:
            logger.error(f"Failed to fetch job {job_id}: {e}")
            return None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        
        Args:
            job_id: RQ job ID
            
        Returns:
            True if cancelled, False otherwise
        """
        try:
            job = RQJob.fetch(job_id, connection=self.redis_conn)
            job.cancel()
            logger.info(f"Cancelled job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all queues.
        
        Returns:
            Queue statistics by priority
        """
        stats = {}
        
        for priority, queue in self.queues.items():
            stats[priority.value] = {
                'pending': len(queue),
                'started': StartedJobRegistry(queue=queue).count,
                'finished': FinishedJobRegistry(queue=queue).count,
                'failed': FailedJobRegistry(queue=queue).count,
            }
        
        return stats
    
    def cleanup_old_jobs(self, older_than_days: int = 7):
        """
        Clean up old finished and failed jobs.
        
        Args:
            older_than_days: Remove jobs older than this many days
        """
        cutoff_time = datetime.utcnow() - timedelta(days=older_than_days)
        
        for priority, queue in self.queues.items():
            # Clean finished jobs
            finished_registry = FinishedJobRegistry(queue=queue)
            for job_id in finished_registry.get_job_ids():
                try:
                    job = RQJob.fetch(job_id, connection=self.redis_conn)
                    if job.ended_at and job.ended_at < cutoff_time:
                        job.delete()
                        logger.debug(f"Deleted old finished job {job_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up job {job_id}: {e}")
            
            # Clean failed jobs
            failed_registry = FailedJobRegistry(queue=queue)
            for job_id in failed_registry.get_job_ids():
                try:
                    job = RQJob.fetch(job_id, connection=self.redis_conn)
                    if job.ended_at and job.ended_at < cutoff_time:
                        job.delete()
                        logger.debug(f"Deleted old failed job {job_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up job {job_id}: {e}")


class LettaJobQueueAsync:
    """Async version of the job queue for use with async code."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('LETTA_REDIS_URL', 'redis://localhost:6379/0')
        self._redis_conn = None
    
    async def _get_connection(self):
        """Get or create async Redis connection."""
        if not self._redis_conn:
            self._redis_conn = await aioredis.from_url(self.redis_url)
        return self._redis_conn
    
    async def enqueue_webhook_callback_async(
        self,
        job_id: str,
        callback_url: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.DEFAULT,
    ) -> str:
        """Async version of enqueue_webhook_callback."""
        conn = await self._get_connection()
        
        # Create job data
        job_data = {
            'id': f"webhook_{job_id}_{uuid4().hex[:8]}",
            'func': 'letta.jobs.webhook_worker.send_webhook_callback',
            'args': (job_id, callback_url, payload),
            'created_at': datetime.utcnow().isoformat(),
            'priority': priority.value,
            'meta': {
                'letta_job_id': job_id,
                'job_type': 'webhook_callback',
            }
        }
        
        # Add to queue
        await conn.lpush(priority.queue_name, json.dumps(job_data))
        
        logger.info(f"Async enqueued webhook callback for job {job_id}")
        return job_data['id']
    
    async def close(self):
        """Close the async Redis connection."""
        if self._redis_conn:
            await self._redis_conn.close()


# Singleton instance for easy access
job_queue = None


def get_job_queue() -> LettaJobQueue:
    """Get the singleton job queue instance."""
    global job_queue
    if job_queue is None:
        job_queue = LettaJobQueue()
    return job_queue


def get_async_job_queue() -> LettaJobQueueAsync:
    """Get an async job queue instance."""
    return LettaJobQueueAsync()