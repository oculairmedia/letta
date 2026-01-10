#!/bin/bash
# Simple cache performance test

echo "=== Letta Agent Cache Performance Test ==="
echo "Time: $(date)"
echo ""

# Get first agent ID
echo "Getting agent list..."
AGENT_ID=$(curl -s -H "Authorization: Bearer lettaSecurePass123" \
                -H "X-BARE-PASSWORD: password lettaSecurePass123" \
                https://letta2.oculair.ca/v1/agents | \
                python3 -c "import sys, json; data=json.load(sys.stdin); print(data[0]['id'] if data else '')")

if [ -z "$AGENT_ID" ]; then
    echo "No agents found"
    exit 1
fi

echo "Testing with agent: $AGENT_ID"
echo ""

# Test 1: Cold cache
echo "1. First load (cold cache):"
START=$(date +%s.%N)
curl -s -H "Authorization: Bearer lettaSecurePass123" \
     -H "X-BARE-PASSWORD: password lettaSecurePass123" \
     https://letta2.oculair.ca/v1/agents/$AGENT_ID > /dev/null
END=$(date +%s.%N)
COLD_TIME=$(echo "$END - $START" | bc)
echo "   Load time: ${COLD_TIME}s"

# Test 2: Warm cache
echo ""
echo "2. Second load (warm cache):"
START=$(date +%s.%N)
curl -s -H "Authorization: Bearer lettaSecurePass123" \
     -H "X-BARE-PASSWORD: password lettaSecurePass123" \
     https://letta2.oculair.ca/v1/agents/$AGENT_ID > /dev/null
END=$(date +%s.%N)
WARM_TIME=$(echo "$END - $START" | bc)
echo "   Load time: ${WARM_TIME}s"
SPEEDUP=$(echo "scale=1; $COLD_TIME / $WARM_TIME" | bc)
echo "   Speedup: ${SPEEDUP}x faster"

# Test 3: Messages
echo ""
echo "3. Recent messages:"
START=$(date +%s.%N)
MSG_COUNT=$(curl -s -H "Authorization: Bearer lettaSecurePass123" \
                 -H "X-BARE-PASSWORD: password lettaSecurePass123" \
                 https://letta2.oculair.ca/v1/agents/$AGENT_ID/messages?limit=20 | \
                 python3 -c "import sys, json; print(len(json.load(sys.stdin)))")
END=$(date +%s.%N)
MSG_TIME=$(echo "$END - $START" | bc)
echo "   Load time: ${MSG_TIME}s"
echo "   Messages retrieved: $MSG_COUNT"

# Check logs for cache hits
echo ""
echo "4. Cache activity in logs:"
docker logs letta-letta-1 --tail 100 2>&1 | grep -i "cache" | tail -5 || echo "   No cache logs found yet"

echo ""
echo "=== Test Complete ==="