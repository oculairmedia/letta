#!/bin/bash
# Startup script for Letta with caching enabled

echo "Starting Letta with Redis caching..."

# Apply cache patches
python /app/apply_cache_patch.py

# Start the original Letta server
exec "$@"