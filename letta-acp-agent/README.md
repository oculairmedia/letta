# Letta ACP Agent

> 🚀 **Agent Client Protocol wrapper for Letta** - Bringing stateful AI agents with persistent memory to any ACP-compatible code editor

## Overview

**Letta ACP Agent** is a bridge that exposes [Letta](https://letta.com)'s stateful AI agents through the [Agent Client Protocol (ACP)](https://agentclientprotocol.com/). This enables you to use Letta's powerful memory-enabled agents in any ACP-compatible code editor like Zed, Cursor, or VS Code.

### What Makes This Special?

- **Persistent Memory**: Letta agents maintain context across sessions with self-editing memory blocks
- **Stateful Conversations**: Full conversation history and evolving agent knowledge
- **Standard Protocol**: Works with any ACP-compatible editor
- **MCP Integration**: Support for Model Context Protocol tools
- **Production-Ready**: Built on Letta's proven agent infrastructure

## Features

✅ Full ACP protocol support (v1)
✅ Session creation and loading
✅ Streaming responses
✅ Persistent memory blocks
✅ Tool call integration
✅ Permission system for sensitive operations
✅ MCP server support
✅ Conversation history replay

## Prerequisites

- Node.js 18 or higher
- Access to Letta API server (self-hosted or [Letta Cloud](https://app.letta.com/))
- An ACP-compatible code editor (Zed, Cursor, etc.)

## Installation

### From Source

```bash
# Clone the repository
cd letta-acp-agent

# Install dependencies
npm install

# Build
npm run build
```

### Quick Start with npm (future)

```bash
npm install -g letta-acp-agent
```

## Configuration

Configure the agent using environment variables:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
LETTA_API_URL=http://localhost:8283
LETTA_API_KEY=your-api-key
LETTA_DEFAULT_MODEL=openai/gpt-4.1
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LETTA_API_URL` | Letta API server URL | `http://localhost:8283` |
| `LETTA_API_KEY` | API key for authentication | (optional) |
| `LETTA_DEFAULT_MODEL` | Default LLM model | `openai/gpt-4.1` |

## Usage

### Running the Agent

```bash
# With default configuration
npm start

# With custom configuration
LETTA_API_URL=https://api.letta.com \
LETTA_API_KEY=your-key \
npm start
```

### Integrating with Code Editors

#### Zed

Add to your Zed configuration (`.config/zed/settings.json`):

```json
{
  "agents": {
    "letta": {
      "command": "/path/to/letta-acp-agent/dist/index.js",
      "env": {
        "LETTA_API_URL": "http://localhost:8283",
        "LETTA_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### Cursor / VS Code

Cursor and VS Code with ACP extensions will have similar configuration. Check the editor-specific documentation for ACP agent setup.

## Architecture

```
┌─────────────────────────────────┐
│   Code Editor (Zed, Cursor)    │
└────────────┬────────────────────┘
             │ ACP Protocol
             │ (JSON-RPC over stdio)
┌────────────┴────────────────────┐
│   Letta ACP Wrapper             │
│   - Protocol handler            │
│   - Message streaming           │
│   - Tool call mapping           │
│   - Session management          │
└────────────┬────────────────────┘
             │ HTTP/REST API
┌────────────┴────────────────────┐
│   Letta API Server              │
│   - Agent state                 │
│   - Memory blocks               │
│   - Message history             │
└─────────────────────────────────┘
```

## How It Works

1. **Initialization**: Code editor spawns the ACP agent process
2. **Protocol Negotiation**: Agent and client negotiate capabilities
3. **Session Creation**: New Letta agent created with workspace context
4. **Prompt Processing**: User messages converted to Letta API calls
5. **Response Streaming**: Letta responses streamed back via ACP notifications
6. **Memory Persistence**: Agent state and memory saved automatically

## Development

### Project Structure

```
letta-acp-agent/
├── src/
│   ├── index.ts              # Entry point
│   ├── agent.ts              # Main ACP agent implementation
│   ├── session.ts            # Session management
│   ├── converters/
│   │   ├── content.ts        # Content format conversion
│   │   └── tools.ts          # Tool call mapping
│   └── types/
│       └── index.ts          # TypeScript types
├── examples/
│   └── zed-config.json       # Example editor config
├── package.json
├── tsconfig.json
└── README.md
```

### Building

```bash
# Build TypeScript
npm run build

# Build and watch for changes
npm run dev

# Type check without building
npm run typecheck
```

### Testing

```bash
# Run tests
npm test

# Run tests once
npm run test:once
```

## Supported Features

### ACP Methods

| Method | Status | Description |
|--------|--------|-------------|
| `initialize` | ✅ | Protocol negotiation |
| `session/new` | ✅ | Create new session |
| `session/load` | ✅ | Load existing session |
| `session/prompt` | ✅ | Process user prompt |
| `session/cancel` | ✅ | Cancel ongoing operation |
| `authenticate` | ✅ | Authentication (pass-through) |

### ACP Capabilities

| Capability | Supported | Notes |
|------------|-----------|-------|
| Load Session | ✅ | Full conversation history replay |
| Embedded Context | ✅ | File and resource content |
| Images | ❌ | Not yet supported by Letta |
| Audio | ❌ | Not yet supported by Letta |
| MCP HTTP | ✅ | Via Letta's MCP integration |
| MCP SSE | ❌ | Not supported |

### Content Types

- ✅ Text content
- ✅ Resource (file) content
- ✅ Resource links
- ✅ Diffs (for file edits)
- ❌ Images (pending Letta support)
- ❌ Audio (pending Letta support)

## Troubleshooting

### Agent not connecting

1. Check Letta API is running: `curl http://localhost:8283/health`
2. Verify API key is correct
3. Check stderr logs for error messages

### Session not loading

- Ensure session ID is valid
- Check Letta database has the agent
- Verify API permissions

### Permission errors

- Confirm `LETTA_API_KEY` is set correctly
- Check Letta server authentication settings

## Limitations

- Image and audio content not yet supported (waiting on Letta)
- MCP integration is simplified (full support coming)
- Tool permission system is basic (will be enhanced)

## Roadmap

### v0.2.0
- [ ] Enhanced tool call handling
- [ ] Full MCP server integration
- [ ] Improved error handling
- [ ] Comprehensive tests

### v0.3.0
- [ ] Image/audio support (when Letta adds it)
- [ ] Advanced permission policies
- [ ] Performance optimizations
- [ ] Documentation site

### v1.0.0
- [ ] npm package release
- [ ] Production hardening
- [ ] Extended editor support
- [ ] Migration tools

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Resources

- [Letta Documentation](https://docs.letta.com/)
- [Agent Client Protocol](https://agentclientprotocol.com/)
- [ACP TypeScript SDK](https://github.com/agentclientprotocol/typescript-sdk)
- [Letta GitHub](https://github.com/letta-ai/letta)

## License

MIT

## Support

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)
- Letta Discord: [Join here](https://discord.gg/letta)

---

Built with ❤️ by the Letta community
