"""
Webhook worker for processing webhook callbacks asynchronously.
"""

import logging
import time
from typing import Any, Dict

import httpx
from rq import get_current_job

from letta.helpers.datetime_helpers import get_utc_time

logger = logging.getLogger(__name__)


def send_webhook_callback(
    job_id: str,
    callback_url: str,
    payload: Dict[str, Any],
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Send webhook callback with retry logic.
    
    Args:
        job_id: Letta job ID
        callback_url: URL to send the callback to
        payload: JSON payload to send
        timeout: Request timeout in seconds
        
    Returns:
        Result dictionary with status information
    """
    current_job = get_current_job()
    attempt = current_job.meta.get('attempt', 0) + 1
    max_retries = current_job.meta.get('retry_count', 3)
    
    logger.info(f"Sending webhook callback for job {job_id} to {callback_url} (attempt {attempt}/{max_retries})")
    
    start_time = time.time()
    
    try:
        # Send the webhook
        with httpx.Client() as client:
            response = client.post(
                callback_url,
                json=payload,
                timeout=timeout,
                headers={
                    'User-Agent': 'Letta-Webhook/1.0',
                    'X-Letta-Job-Id': job_id,
                    'X-Letta-Attempt': str(attempt),
                }
            )
        
        duration = time.time() - start_time
        
        result = {
            'job_id': job_id,
            'callback_url': callback_url,
            'status_code': response.status_code,
            'duration': duration,
            'attempt': attempt,
            'success': 200 <= response.status_code < 300,
            'sent_at': get_utc_time().isoformat(),
        }
        
        # Log response details
        if result['success']:
            logger.info(f"Webhook callback successful for job {job_id}: {response.status_code}")
        else:
            logger.warning(f"Webhook callback failed for job {job_id}: {response.status_code}")
            result['response_body'] = response.text[:500]  # First 500 chars of error response
        
        # Update job metadata
        current_job.meta['last_attempt'] = attempt
        current_job.meta['last_status_code'] = response.status_code
        current_job.save_meta()
        
        # Retry logic for failed requests
        if not result['success'] and attempt < max_retries:
            retry_delay = current_job.meta.get('retry_delay', 60) * attempt
            raise RetryableWebhookError(
                f"Webhook returned {response.status_code}, will retry in {retry_delay}s",
                retry_delay=retry_delay
            )
        
        return result
        
    except httpx.TimeoutException as e:
        duration = time.time() - start_time
        logger.error(f"Webhook timeout for job {job_id} after {duration:.2f}s: {e}")
        
        if attempt < max_retries:
            retry_delay = current_job.meta.get('retry_delay', 60) * attempt
            raise RetryableWebhookError(
                f"Webhook timeout, will retry in {retry_delay}s",
                retry_delay=retry_delay
            )
        
        return {
            'job_id': job_id,
            'callback_url': callback_url,
            'error': 'timeout',
            'duration': duration,
            'attempt': attempt,
            'success': False,
            'sent_at': get_utc_time().isoformat(),
        }
        
    except httpx.RequestError as e:
        duration = time.time() - start_time
        logger.error(f"Webhook request error for job {job_id}: {e}")
        
        if attempt < max_retries:
            retry_delay = current_job.meta.get('retry_delay', 60) * attempt
            raise RetryableWebhookError(
                f"Request error: {e}, will retry in {retry_delay}s",
                retry_delay=retry_delay
            )
        
        return {
            'job_id': job_id,
            'callback_url': callback_url,
            'error': str(e),
            'duration': duration,
            'attempt': attempt,
            'success': False,
            'sent_at': get_utc_time().isoformat(),
        }
    
    except Exception as e:
        logger.exception(f"Unexpected error sending webhook for job {job_id}")
        return {
            'job_id': job_id,
            'callback_url': callback_url,
            'error': str(e),
            'attempt': attempt,
            'success': False,
            'sent_at': get_utc_time().isoformat(),
        }


class RetryableWebhookError(Exception):
    """Exception that triggers a retry with delay."""
    
    def __init__(self, message: str, retry_delay: int):
        super().__init__(message)
        self.retry_delay = retry_delay


def update_job_with_webhook_result(job_id: str, result: Dict[str, Any]):
    """
    Update the Letta job with webhook result information.
    
    This function should be called after webhook processing to update
    the job's callback status in the database.
    """
    # Import here to avoid circular imports
    from letta.server.db import db_registry
    from letta.services.job_manager import JobManager
    from letta.schemas.job import JobUpdate
    
    try:
        with db_registry.session() as session:
            job_manager = JobManager()
            
            # Prepare update data
            update_data = {
                'callback_sent_at': result.get('sent_at'),
                'callback_status_code': result.get('status_code'),
            }
            
            if result.get('error'):
                update_data['callback_error'] = result['error']
            
            # TODO: This needs proper actor context
            # For now, we'll need to modify the job manager to support system updates
            logger.info(f"Webhook result recorded for job {job_id}: {result}")
            
    except Exception as e:
        logger.error(f"Failed to update job {job_id} with webhook result: {e}")