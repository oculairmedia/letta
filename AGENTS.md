# Letta Project Agent Instructions

## Fixing Corrupted Agent Message State

Letta agents can get into a corrupted state where the message history has orphaned `tool_result` blocks without matching `tool_use` calls. This causes the Anthropic API to reject requests with errors like:

```
invalid_request_error: unexpected `tool_use_id` found in `tool_result` blocks
```

### Quick Fix

Use the fix script to repair corrupted agents:

```bash
/opt/stacks/letta/scripts/fix-agent-messages.sh <agent-id>
```

### Options

```bash
# Preview what would be done without making changes
./scripts/fix-agent-messages.sh --dry-run <agent-id>

# Verbose output for debugging
./scripts/fix-agent-messages.sh --verbose <agent-id>
```

### What It Fixes

1. **Orphaned tool_result** - Consecutive tool messages without assistant messages between them
2. **Tool messages without matching tool_use** - Tool responses missing corresponding tool calls
3. **Missing system message** - Empty `message_ids` or deleted system message

### How It Works

1. Detects corruption patterns in the agent's message history
2. Finds a healthy agent in the same org to use as a template
3. Deletes all corrupted messages from the agent
4. Creates a new system message by copying content from the template
5. Updates the agent's `message_ids` to point to the new system message

### Manual Database Fix (if script fails)

If the script doesn't work, you can manually fix via the database:

```bash
# Connect to the database
docker exec -it letta-postgres-1 psql -U letta -d letta

# Check agent state
SELECT message_ids FROM agents WHERE id = 'agent-xxx';

# Delete corrupted messages
DELETE FROM messages WHERE agent_id = 'agent-xxx';

# Find a healthy agent's system message to use as template
SELECT id, content FROM messages WHERE role = 'system' AND content IS NOT NULL LIMIT 1;

# Create new system message (replace template-msg-id)
INSERT INTO messages (id, organization_id, agent_id, role, text, tool_calls, content, created_at)
SELECT 'message-<new-uuid>', organization_id, 'agent-xxx', 'system', '', '[]', 
       (SELECT content FROM messages WHERE id = 'template-msg-id'), NOW()
FROM agents WHERE id = 'agent-xxx';

# Update agent to use new system message
UPDATE agents SET message_ids = '["message-<new-uuid>"]' WHERE id = 'agent-xxx';
```

### Environment Variables

The fix script supports these environment variables:

- `LETTA_POSTGRES_CONTAINER` - Postgres container name (default: `letta-postgres-1`)
- `LETTA_POSTGRES_USER` - Database user (default: `letta`)
- `LETTA_POSTGRES_DB` - Database name (default: `letta`)
- `LETTA_API_URL` - Letta API URL (default: `http://localhost:8283`)

## OpenCode Configuration

### Config File Locations

OpenCode uses a hierarchical configuration system with the following locations (in order of precedence):

1. **Global config:** `~/.config/opencode/opencode.json`
   - Use for themes, providers, keybinds, and globally available MCP servers
   
2. **Project config:** `opencode.json` in the project root (or nearest Git directory)
   - Use for project-specific settings like models, MCP servers, or modes

3. **Custom path:** Set via `OPENCODE_CONFIG` environment variable

### Config Merging Behavior

Configuration files are **merged together, not replaced**. Settings from later configs override earlier ones only for conflicting keys. Non-conflicting settings from all configs are preserved.

For example:
- Global config sets `theme: "opencode"` and `autoupdate: true`
- Project config sets `model: "anthropic/claude-sonnet-4-5"`
- Final config includes all three settings

This means you cannot simply omit a key to disable something from the global config - you must explicitly override it.

### MCP Server Configuration

When configuring MCP servers in a project's `opencode.json`, you must include the full server definition with `type` and `url` fields even when disabling servers that are enabled globally.

### Disabling Global MCP Servers

To disable MCP servers that are enabled in `~/.config/opencode/opencode.json`, you must replicate the full server config with `enabled: false`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "server-to-disable": {
      "type": "sse",
      "url": "http://example.com/mcp",
      "enabled": false
    }
  }
}
```

**Important:** Simply setting `"server-name": { "enabled": false }` without `type` and `url` will cause validation errors:
```
Invalid input mcp.server-name.type
```

### Example: Letta Project Config

This project enables `letta-mcp` and disables `huly-mcp`, `vibe-task`, and `vibe-system`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "letta-mcp": {
      "type": "sse",
      "url": "http://192.168.50.90:3001/mcp",
      "enabled": true
    },
    "huly-mcp": {
      "type": "sse",
      "url": "http://192.168.50.90:3002/mcp",
      "enabled": false
    },
    "vibe-task": {
      "type": "sse",
      "url": "http://192.168.50.90:3004/mcp",
      "enabled": false
    },
    "vibe-system": {
      "type": "sse",
      "url": "http://192.168.50.90:3005/mcp",
      "enabled": false
    }
  }
}
```
