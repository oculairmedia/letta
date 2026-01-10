#!/usr/bin/env python3
"""
Apply Redis Queue integration to Letta's job manager.
This patch modifies the job manager to use Redis Queue for background processing.
"""

import os
import shutil
from datetime import datetime

JOB_MANAGER_PATH = "/app/letta/services/job_manager.py"
BACKUP_PATH = f"/app/letta/services/job_manager.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def apply_redis_queue_patch():
    """Apply Redis Queue modifications to job_manager.py"""
    
    # Create backup
    if os.path.exists(JOB_MANAGER_PATH):
        shutil.copy2(JOB_MANAGER_PATH, BACKUP_PATH)
        print(f"Created backup: {BACKUP_PATH}")
    
    # Read the original file
    with open(JOB_MANAGER_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "from letta.jobs.redis_job_queue import" in content:
        print("Redis Queue patch already applied!")
        return
    
    # Add imports after the existing imports
    import_section = """from letta.utils import enforce_types

logger = get_logger(__name__)"""
    
    new_import_section = """from letta.utils import enforce_types

logger = get_logger(__name__)

# Redis Queue integration
try:
    from letta.jobs.redis_job_queue import get_job_queue, get_async_job_queue, JobPriority
    REDIS_QUEUE_AVAILABLE = True
except ImportError:
    logger.warning("Redis Queue not available, using synchronous processing")
    REDIS_QUEUE_AVAILABLE = False"""
    
    content = content.replace(import_section, new_import_section)
    
    # Modify _dispatch_callback to use Redis Queue
    old_dispatch = """    def _dispatch_callback(self, job: JobModel) -> None:
        \"\"\"
        POST a standard JSON payload to job.callback_url
        and record timestamp + HTTP status.
        \"\"\"

        payload = {
            "job_id": job.id,
            "status": job.status,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metadata": job.metadata_,
        }
        try:
            import httpx

            resp = httpx.post(job.callback_url, json=payload, timeout=5.0)
            job.callback_sent_at = get_utc_time().replace(tzinfo=None)
            job.callback_status_code = resp.status_code

        except Exception as e:
            error_message = f"Failed to dispatch callback for job {job.id} to {job.callback_url}: {str(e)}\""""
    
    new_dispatch = """    def _dispatch_callback(self, job: JobModel) -> None:
        \"\"\"
        POST a standard JSON payload to job.callback_url
        and record timestamp + HTTP status.
        \"\"\"

        payload = {
            "job_id": job.id,
            "status": job.status,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metadata": job.metadata_,
        }
        
        # Use Redis Queue if available
        if REDIS_QUEUE_AVAILABLE and os.getenv('LETTA_USE_REDIS_QUEUE', 'true').lower() == 'true':
            try:
                job_queue = get_job_queue()
                
                # Determine priority based on job type or metadata
                priority = JobPriority.HIGH if job.metadata_.get('priority') == 'high' else JobPriority.DEFAULT
                
                # Enqueue the webhook callback
                rq_job_id = job_queue.enqueue_webhook_callback(
                    job_id=str(job.id),
                    callback_url=job.callback_url,
                    payload=payload,
                    priority=priority,
                    retry_count=3,
                    retry_delay=60,
                )
                
                # Mark as queued
                job.callback_sent_at = get_utc_time().replace(tzinfo=None)
                job.callback_status_code = 202  # Accepted/Queued
                
                logger.info(f"Webhook callback queued for job {job.id} with RQ job {rq_job_id}")
                return
                
            except Exception as e:
                logger.error(f"Failed to queue webhook callback, falling back to sync: {e}")
        
        # Fallback to synchronous processing
        try:
            import httpx

            resp = httpx.post(job.callback_url, json=payload, timeout=5.0)
            job.callback_sent_at = get_utc_time().replace(tzinfo=None)
            job.callback_status_code = resp.status_code

        except Exception as e:
            error_message = f"Failed to dispatch callback for job {job.id} to {job.callback_url}: {str(e)}\""""
    
    content = content.replace(old_dispatch, new_dispatch)
    
    # Modify async version similarly
    old_async_dispatch = """    async def _dispatch_callback_async(self, job: JobModel) -> None:
        \"\"\"
        POST a standard JSON payload to job.callback_url and record timestamp + HTTP status asynchronously.
        \"\"\"
        payload = {
            "job_id": job.id,
            "status": job.status,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metadata": job.metadata_,
        }

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(job.callback_url, json=payload, timeout=5.0)
                # Ensure timestamp is timezone-naive for DB compatibility
                job.callback_sent_at = get_utc_time().replace(tzinfo=None)
                job.callback_status_code = resp.status_code
        except Exception as e:
            error_message = f"Failed to dispatch callback for job {job.id} to {job.callback_url}: {str(e)}\""""
    
    new_async_dispatch = """    async def _dispatch_callback_async(self, job: JobModel) -> None:
        \"\"\"
        POST a standard JSON payload to job.callback_url and record timestamp + HTTP status asynchronously.
        \"\"\"
        payload = {
            "job_id": job.id,
            "status": job.status,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "metadata": job.metadata_,
        }
        
        # Use Redis Queue if available
        if REDIS_QUEUE_AVAILABLE and os.getenv('LETTA_USE_REDIS_QUEUE', 'true').lower() == 'true':
            try:
                async_queue = get_async_job_queue()
                
                # Determine priority
                priority = JobPriority.HIGH if job.metadata_.get('priority') == 'high' else JobPriority.DEFAULT
                
                # Enqueue the webhook callback
                rq_job_id = await async_queue.enqueue_webhook_callback_async(
                    job_id=str(job.id),
                    callback_url=job.callback_url,
                    payload=payload,
                    priority=priority,
                )
                
                # Mark as queued
                job.callback_sent_at = get_utc_time().replace(tzinfo=None)
                job.callback_status_code = 202  # Accepted/Queued
                
                logger.info(f"Webhook callback async queued for job {job.id} with RQ job {rq_job_id}")
                
                await async_queue.close()
                return
                
            except Exception as e:
                logger.error(f"Failed to queue webhook callback async, falling back to sync: {e}")

        # Fallback to synchronous processing
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(job.callback_url, json=payload, timeout=5.0)
                # Ensure timestamp is timezone-naive for DB compatibility
                job.callback_sent_at = get_utc_time().replace(tzinfo=None)
                job.callback_status_code = resp.status_code
        except Exception as e:
            error_message = f"Failed to dispatch callback for job {job.id} to {job.callback_url}: {str(e)}\""""
    
    content = content.replace(old_async_dispatch, new_async_dispatch)
    
    # Write the patched file
    with open(JOB_MANAGER_PATH, 'w') as f:
        f.write(content)
    
    print("Redis Queue patch applied successfully!")
    print("Webhook callbacks will now be processed asynchronously via Redis Queue.")
    print("Set LETTA_USE_REDIS_QUEUE=false to disable Redis Queue processing.")


if __name__ == "__main__":
    apply_redis_queue_patch()