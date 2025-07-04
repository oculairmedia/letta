#!/bin/bash
# Start Redis Queue workers for Letta background job processing

# Configuration
REDIS_URL="${LETTA_REDIS_URL:-redis://redis:6379/0}"
WORKER_COUNT="${LETTA_WORKER_COUNT:-4}"
LOG_LEVEL="${LETTA_WORKER_LOG_LEVEL:-INFO}"

echo "Starting Letta Redis Queue workers..."
echo "Redis URL: $REDIS_URL"
echo "Worker count: $WORKER_COUNT"
echo "Log level: $LOG_LEVEL"

# Install RQ if not already installed
if ! python -c "import rq" 2>/dev/null; then
    echo "Installing RQ dependencies..."
    pip install rq rq-scheduler
fi

# Ensure the jobs module is in the Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Create jobs directory structure if needed
mkdir -p /app/letta/jobs
touch /app/letta/jobs/__init__.py

# Copy worker modules to the jobs directory
cp /app/letta/jobs/webhook_worker.py /app/letta/jobs/ 2>/dev/null || true
cp /app/letta/jobs/llm_batch_worker.py /app/letta/jobs/ 2>/dev/null || true
cp /app/letta/jobs/redis_job_queue.py /app/letta/jobs/ 2>/dev/null || true

# Start workers for different priority queues
echo "Starting critical priority worker..."
rq worker letta:critical --url "$REDIS_URL" --logging_level "$LOG_LEVEL" &

echo "Starting high priority workers..."
for i in $(seq 1 2); do
    rq worker letta:high --url "$REDIS_URL" --logging_level "$LOG_LEVEL" &
done

echo "Starting default priority workers..."
for i in $(seq 1 $WORKER_COUNT); do
    rq worker letta:default letta:low --url "$REDIS_URL" --logging_level "$LOG_LEVEL" &
done

# Start the scheduler for delayed/scheduled jobs
echo "Starting RQ scheduler..."
rq-scheduler --url "$REDIS_URL" --logging_level "$LOG_LEVEL" &

# Keep the script running
echo "Workers started. Press Ctrl+C to stop."
wait