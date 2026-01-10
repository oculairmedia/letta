# Letta Deployment Configuration

This branch contains the production deployment configuration for Letta.

## Structure

- `compose.yaml` - Docker Compose configuration with Redis, PostgreSQL, and Letta services
- `llmux_shim/` - OpenAI-compatible shim for routing to multiple LLM providers
- `custom_modules/` - Custom Python modules for webhook workers, Redis integration
- `scripts/` - Deployment and maintenance scripts
- `AGENTS.md` - Agent management documentation
- `README.md` - Data protection and backup information

## LLM Provider Routing

The llmux shim automatically routes requests based on model names:

- **Claude models** (`claude-*`) → Port 8082 (Claude Max Proxy)
- **Gemini models** (`gpt-gemini-*`, `gemini-*`) → Port 8089 (IT-BAER Proxy)
- **GLM models** (`glm-*`) → Z.AI API (https://api.z.ai)

## Quick Start

```bash
# Start the stack
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f letta

# Restart after configuration changes
docker compose restart
```

## Configuration

Required environment variables in `.env`:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `ZAI_API_KEY`
- `TOGETHER_API_KEY`
- `COMPOSIO_API_KEY`
- `TAVILY_API_KEY`

## Backup

Run `./scripts/backup.sh` for manual PostgreSQL backups.
