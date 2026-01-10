#!/usr/bin/env python3
"""Test Redis Queue integration with Letta"""

import sys
import os
import time
import json
from uuid import uuid4

sys.path.append('/app')

# Set environment to use Redis Queue
os.environ['LETTA_USE_REDIS_QUEUE'] = 'true'

from letta.jobs.redis_job_queue import get_job_queue, JobPriority


def test_redis_queue():
    """Test Redis Queue functionality"""
    
    print("Testing Redis Queue integration...")
    
    try:
        # Initialize queue
        job_queue = get_job_queue()
        print("✓ Job queue initialized successfully")
        
        # Test 1: Enqueue a webhook callback
        test_job_id = str(uuid4())
        test_payload = {
            "job_id": test_job_id,
            "status": "completed",
            "completed_at": "2025-06-22T17:30:00+00:00",
            "metadata": {"test": True}
        }
        
        rq_job_id = job_queue.enqueue_webhook_callback(
            job_id=test_job_id,
            callback_url="https://httpbin.org/post",
            payload=test_payload,
            priority=JobPriority.HIGH,
        )
        
        print(f"✓ Enqueued webhook callback: {rq_job_id}")
        
        # Test 2: Check job status
        time.sleep(1)
        status = job_queue.get_job_status(rq_job_id)
        if status:
            print(f"✓ Job status: {status['status']}")
        
        # Test 3: Check queue statistics
        stats = job_queue.get_queue_stats()
        print("\n📊 Queue Statistics:")
        for priority, counts in stats.items():
            print(f"  {priority}: {counts}")
        
        # Test 4: Enqueue a batch processing job
        batch_id = str(uuid4())
        batch_job_id = job_queue.enqueue_llm_batch_processing(
            batch_id=batch_id,
            llm_config={"model": "gpt-4", "temperature": 0.7},
            messages=[
                {"id": "msg1", "content": "Test message 1"},
                {"id": "msg2", "content": "Test message 2"},
            ],
            priority=JobPriority.DEFAULT,
        )
        
        print(f"\n✓ Enqueued LLM batch job: {batch_job_id}")
        
        # Test 5: Schedule a future job
        from datetime import datetime, timedelta
        
        future_time = datetime.utcnow() + timedelta(minutes=5)
        scheduled_job_id = job_queue.schedule_job(
            func='letta.jobs.webhook_worker.send_webhook_callback',
            args=(test_job_id, "https://httpbin.org/post", {"scheduled": True}),
            scheduled_time=future_time,
            priority=JobPriority.LOW,
        )
        
        print(f"✓ Scheduled future job: {scheduled_job_id} for {future_time}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_worker_availability():
    """Check if workers are processing jobs"""
    
    print("\n🔧 Checking worker availability...")
    
    try:
        job_queue = get_job_queue()
        
        # Create a simple test job
        test_job_id = job_queue.enqueue_webhook_callback(
            job_id="worker-test",
            callback_url="https://httpbin.org/status/200",
            payload={"test": "worker"},
            priority=JobPriority.CRITICAL,
        )
        
        print(f"Created test job: {test_job_id}")
        
        # Wait for processing
        max_wait = 10
        for i in range(max_wait):
            time.sleep(1)
            status = job_queue.get_job_status(test_job_id)
            
            if status:
                print(f"Job status after {i+1}s: {status['status']}")
                
                if status['status'] == 'finished':
                    print("✓ Workers are processing jobs successfully!")
                    if status.get('result'):
                        print(f"Result: {json.dumps(status['result'], indent=2)}")
                    return True
                elif status['status'] == 'failed':
                    print(f"✗ Job failed: {status.get('exc_info')}")
                    return False
        
        print("⚠️ Job not processed within timeout - workers may not be running")
        return False
        
    except Exception as e:
        print(f"Error checking workers: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    queue_ok = test_redis_queue()
    workers_ok = test_worker_availability()
    
    sys.exit(0 if (queue_ok and workers_ok) else 1)