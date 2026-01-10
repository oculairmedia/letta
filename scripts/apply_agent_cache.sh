#!/bin/bash
# Apply agent state caching patches to Letta

set -e

echo "=== Applying Agent State Caching to Letta ==="

# 1. Copy the agent state cache module
echo "1. Copying agent_state_cache.py..."
docker cp /opt/stacks/letta/custom_modules/agent_state_cache.py letta-letta-1:/app/letta/cache/agent_state_cache.py

# 2. Apply patches
echo "2. Applying patches..."
for patch in /opt/stacks/letta/patches/002_agent_state_cache.patch \
            /opt/stacks/letta/patches/003_message_cache.patch \
            /opt/stacks/letta/patches/004_agent_session_management.patch; do
    if [ -f "$patch" ]; then
        echo "   Applying $(basename $patch)..."
        docker exec letta-letta-1 patch -p1 < "$patch" || echo "   Patch may have already been applied"
    fi
done

# 3. Set environment variable
echo "3. Adding LETTA_USE_AGENT_CACHE to docker-compose..."
if ! grep -q "LETTA_USE_AGENT_CACHE" /opt/stacks/letta/compose.yaml; then
    sed -i '/LETTA_USE_REDIS_QUEUE/a\      - LETTA_USE_AGENT_CACHE=${LETTA_USE_AGENT_CACHE:-true}' /opt/stacks/letta/compose.yaml
    echo "   Environment variable added"
else
    echo "   Environment variable already exists"
fi

# 4. Restart Letta
echo "4. Restarting Letta to apply changes..."
cd /opt/stacks/letta
docker compose restart letta

echo ""
echo "=== Agent State Caching Applied Successfully ==="
echo ""
echo "Features enabled:"
echo "✓ Agent state caching (1 hour TTL)"
echo "✓ Memory block caching"
echo "✓ Recent message caching (100 messages, 30 min TTL)"
echo "✓ Active session management (15 min TTL)"
echo ""
echo "To disable caching, set LETTA_USE_AGENT_CACHE=false"
echo ""
echo "Monitor cache performance:"
echo "docker exec letta-letta-1 python -c \"from letta.cache.agent_state_cache import get_agent_cache; print(get_agent_cache().get_stats())\""