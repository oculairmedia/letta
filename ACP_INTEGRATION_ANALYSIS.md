# Agent Client Protocol (ACP) Integration Analysis for Letta

## Executive Summary

This document outlines the strategy for integrating Letta with the Agent Client Protocol (ACP), enabling Letta's stateful AI agents to work with any ACP-compatible code editor (Zed, Cursor, VS Code, etc.).

## Architecture Overview

### Current State

**Letta:**
- Python-based API server with TypeScript/Node.js client SDK
- Stateful agents with persistent memory blocks
- Self-editing memory system
- MCP tool support
- HTTP/REST API for agent interaction
- Streaming responses
- Built-in tools (web_search, run_code, filesystem operations)

**ACP:**
- JSON-RPC 2.0 protocol over stdio
- Bidirectional client-server communication
- 14 core methods for initialization, session management, file I/O, terminal execution
- Streaming architecture with real-time notifications
- Official SDKs: TypeScript, Python, Rust, Kotlin
- Tool call permission system
- Content types: text, images, audio, resources, diffs, terminals

### Integration Strategy: Letta as ACP Agent

**Recommended Approach:** Create an ACP wrapper that exposes Letta agents through the ACP protocol.

**Benefits:**
1. Letta's stateful agents become available to entire ACP ecosystem
2. Any ACP-compatible code editor can use Letta's unique memory system
3. Leverages existing Letta infrastructure (API server, memory, skills)
4. Maintains Letta's self-editing memory capabilities
5. Enables broader ecosystem adoption

## Protocol Mapping

### 1. Initialization Phase

**ACP Method:** `initialize`

**Mapping:**
```typescript
initialize(params: InitializeRequest) -> InitializeResponse {
  return {
    protocolVersion: 1,
    agentCapabilities: {
      loadSession: true,  // Letta supports persistent sessions
      promptCapabilities: {
        embeddedContext: true,  // Letta can handle file context
        image: false,  // Not currently supported
        audio: false   // Not currently supported
      },
      mcpCapabilities: {
        http: true,   // Letta has MCP support
        sse: false
      }
    },
    agentInfo: {
      name: "letta",
      title: "Letta Agent",
      version: "<letta-version>"
    }
  }
}
```

### 2. Session Management

**ACP Methods:** `session/new`, `session/load`

**Mapping to Letta:**

```typescript
// session/new maps to Letta's agent creation
async newSession(params: NewSessionRequest) {
  const lettaAgent = await lettaClient.agents.create({
    model: "openai/gpt-4.1",
    memoryBlocks: [
      { label: "persona", value: "I am a helpful coding assistant" },
      { label: "workspace", value: `Working directory: ${params.cwd}` }
    ],
    tools: mapMcpServersToLettaTools(params.mcpServers)
  });

  return {
    sessionId: lettaAgent.id
  };
}

// session/load maps to loading existing Letta agent
async loadSession(params: LoadSessionRequest) {
  const lettaAgent = await lettaClient.agents.retrieve(params.sessionId);

  // Stream conversation history to client
  const messages = await lettaClient.agents.messages.list(params.sessionId);
  for (const msg of messages) {
    await connection.sessionUpdate({
      sessionId: params.sessionId,
      update: convertLettaMessageToACPUpdate(msg)
    });
  }

  return {};
}
```

### 3. Prompt Processing

**ACP Method:** `session/prompt`

**Mapping to Letta:**

```typescript
async prompt(params: PromptRequest) {
  const sessionId = params.sessionId;

  // Convert ACP content to Letta message format
  const lettaMessage = convertACPContentToLettaMessage(params.prompt);

  // Create streaming response
  const stream = await lettaClient.agents.messages.createStream({
    agentId: sessionId,
    messages: [lettaMessage],
    streamTokens: true
  });

  // Stream updates to client
  for await (const chunk of stream) {
    const acpUpdate = convertLettaChunkToACPUpdate(chunk);
    await connection.sessionUpdate({
      sessionId,
      update: acpUpdate
    });
  }

  return { stopReason: "end_turn" };
}
```

### 4. Client Capabilities Mapping

**File System Operations:**

| ACP Method | Letta Implementation |
|------------|---------------------|
| `fs/read_text_file` | Use Letta's filesystem tools or call client method |
| `fs/write_text_file` | Use Letta's filesystem tools or call client method |

**Terminal Operations:**

| ACP Method | Letta Implementation |
|------------|---------------------|
| `terminal/create` | Delegate to client (code editor handles terminals) |
| `terminal/output` | Delegate to client |
| `terminal/wait_for_exit` | Delegate to client |
| `terminal/kill` | Delegate to client |
| `terminal/release` | Delegate to client |

### 5. Tool Call Mapping

**Letta Tool → ACP Tool Call:**

```typescript
function convertLettaToolCallToACP(lettaTool: LettaToolCall): ToolCallUpdate {
  return {
    toolCallId: lettaTool.id,
    title: lettaTool.name,
    kind: mapLettaToolKindToACP(lettaTool.name),
    status: mapLettaToolStatusToACP(lettaTool.status),
    rawInput: lettaTool.arguments,
    rawOutput: lettaTool.result
  };
}

function mapLettaToolKindToACP(toolName: string): ToolKind {
  const kindMap = {
    'open_file': 'read',
    'grep_file': 'search',
    'search_file': 'search',
    'run_code': 'execute',
    'web_search': 'fetch',
    'edit_file': 'edit',
    'delete_file': 'delete',
    'move_file': 'move'
  };
  return kindMap[toolName] || 'other';
}
```

### 6. Memory Block Integration

**Letta's Unique Feature:** Self-editing memory blocks

**ACP Integration Strategy:**
- Use Letta's memory blocks as persistent context
- Memory edits are represented as internal tool calls
- Client can view memory state through session updates

```typescript
// When Letta edits its memory, send as tool call update
async function handleMemoryEdit(memoryEdit: LettaMemoryEdit) {
  await connection.sessionUpdate({
    sessionId: currentSessionId,
    update: {
      sessionUpdate: 'tool_call',
      toolCallId: `memory_${Date.now()}`,
      title: `Updated memory: ${memoryEdit.blockLabel}`,
      kind: 'think',  // Memory editing is internal reasoning
      status: 'completed',
      content: [{
        type: 'diff',
        path: `memory://${memoryEdit.blockLabel}`,
        oldText: memoryEdit.oldValue,
        newText: memoryEdit.newValue
      }]
    }
  });
}
```

## Implementation Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Code Editor (Client)                     │
│                  (Zed, Cursor, VS Code, etc.)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ ACP Protocol
                         │ (JSON-RPC over stdio)
                         │
┌────────────────────────┴────────────────────────────────────┐
│              Letta ACP Wrapper (TypeScript)                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         AgentSideConnection (ACP SDK)                  │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  • Protocol handler (initialize, session/*, etc.)      │ │
│  │  • Message streaming & transformation                  │ │
│  │  • Tool call mapping                                   │ │
│  │  • Permission management                               │ │
│  └─────────────────────┬──────────────────────────────────┘ │
└────────────────────────┼────────────────────────────────────┘
                         │
                         │ HTTP/REST API
                         │ (Letta Client SDK)
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   Letta API Server (Python)                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  • Agent state management                              │ │
│  │  • Memory blocks (persona, workspace, custom)          │ │
│  │  • Message history                                     │ │
│  │  • LLM integration                                     │ │
│  │  • MCP tool integration                                │ │
│  │  • Streaming responses                                 │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
letta-acp-agent/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts                    # Entry point (stdio handler)
│   ├── agent.ts                    # Main LettaACPAgent class
│   ├── session.ts                  # Session management
│   ├── converters/
│   │   ├── content.ts              # ACP ↔ Letta content conversion
│   │   ├── tools.ts                # Tool call mapping
│   │   └── messages.ts             # Message format conversion
│   ├── letta/
│   │   ├── client.ts               # Letta API client wrapper
│   │   ├── streaming.ts            # Stream handling
│   │   └── memory.ts               # Memory block management
│   └── utils/
│       ├── logger.ts               # Logging utilities
│       └── errors.ts               # Error handling
├── examples/
│   └── basic-agent.ts              # Example usage
└── README.md
```

## Key Implementation Details

### 1. Message Streaming

**Challenge:** Convert Letta's streaming format to ACP session updates

**Solution:**
```typescript
async function streamLettaResponseToACP(
  lettaStream: AsyncIterable<LettaChunk>,
  sessionId: string,
  connection: AgentSideConnection
) {
  for await (const chunk of lettaStream) {
    switch (chunk.type) {
      case 'message_chunk':
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: 'agent_message_chunk',
            content: {
              type: 'text',
              text: chunk.content
            }
          }
        });
        break;

      case 'tool_call':
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: 'tool_call',
            toolCallId: chunk.toolCallId,
            title: chunk.toolName,
            kind: mapToolKind(chunk.toolName),
            status: 'pending',
            rawInput: chunk.arguments
          }
        });
        break;

      case 'tool_result':
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: 'tool_call_update',
            toolCallId: chunk.toolCallId,
            status: 'completed',
            content: convertToolResultToContent(chunk.result),
            rawOutput: chunk.result
          }
        });
        break;
    }
  }
}
```

### 2. Permission Requests

**Letta Integration:**
- Some tools should require user permission (file writes, command execution)
- ACP provides `session/request_permission` method
- Map sensitive Letta operations to permission requests

```typescript
async function shouldRequestPermission(toolName: string): boolean {
  const sensitiveTools = [
    'edit_file',
    'delete_file',
    'run_code',
    'web_search'  // Optional: some users may want to approve web requests
  ];
  return sensitiveTools.includes(toolName);
}

async function executeToolWithPermission(
  toolCall: LettaToolCall,
  sessionId: string,
  connection: AgentSideConnection
): Promise<ToolResult> {
  if (await shouldRequestPermission(toolCall.name)) {
    const permissionResponse = await connection.requestPermission({
      sessionId,
      toolCall: convertToACPToolCall(toolCall),
      options: [
        { optionId: 'allow', name: 'Allow', kind: 'allow_once' },
        { optionId: 'reject', name: 'Reject', kind: 'reject_once' },
        { optionId: 'allow_always', name: 'Always allow', kind: 'allow_always' }
      ]
    });

    if (permissionResponse.outcome.outcome === 'cancelled') {
      throw new Error('Operation cancelled');
    }

    if (permissionResponse.outcome.outcome === 'selected' &&
        permissionResponse.outcome.optionId === 'reject') {
      throw new Error('Operation rejected by user');
    }
  }

  return await executeLettaTool(toolCall);
}
```

### 3. MCP Server Integration

**Challenge:** ACP clients provide MCP server configurations, Letta has its own MCP integration

**Solution:** Bridge ACP MCP configs to Letta's MCP system

```typescript
async function setupMCPServers(
  mcpServers: McpServerConfig[],
  lettaAgentId: string
) {
  for (const mcpServer of mcpServers) {
    // Add MCP server to Letta
    const tools = await lettaClient.tools.listMcpToolsByServer(mcpServer.name);

    for (const tool of tools) {
      const lettaTool = await lettaClient.tools.addMcpTool(
        mcpServer.name,
        tool.name
      );

      // Attach to agent
      await lettaClient.agents.tool.attach({
        agentId: lettaAgentId,
        toolId: lettaTool.id
      });
    }
  }
}
```

### 4. Working Directory Context

**ACP Requirement:** `cwd` parameter in session creation

**Letta Integration:**
- Add `cwd` to agent's workspace memory block
- Use for file path resolution
- Constrain file operations to workspace

```typescript
async function createLettaAgentWithWorkspace(cwd: string) {
  return await lettaClient.agents.create({
    model: "openai/gpt-4.1",
    memoryBlocks: [
      {
        label: "persona",
        value: "I am a helpful coding assistant with access to the workspace."
      },
      {
        label: "workspace",
        value: `Working directory: ${cwd}\n` +
               `I should constrain file operations to this directory.`
      }
    ]
  });
}
```

## Session State Management

```typescript
interface LettaACPSession {
  sessionId: string;
  lettaAgentId: string;
  cwd: string;
  mcpServers: McpServerConfig[];
  pendingPrompt: AbortController | null;
  memoryBlocks: Map<string, MemoryBlock>;
}

class SessionManager {
  private sessions: Map<string, LettaACPSession> = new Map();

  async createSession(cwd: string, mcpServers: McpServerConfig[]): Promise<string> {
    const lettaAgent = await createLettaAgentWithWorkspace(cwd);
    await setupMCPServers(mcpServers, lettaAgent.id);

    const session: LettaACPSession = {
      sessionId: lettaAgent.id,
      lettaAgentId: lettaAgent.id,
      cwd,
      mcpServers,
      pendingPrompt: null,
      memoryBlocks: new Map()
    };

    this.sessions.set(lettaAgent.id, session);
    return lettaAgent.id;
  }

  async loadSession(sessionId: string, cwd: string, mcpServers: McpServerConfig[]): Promise<void> {
    const lettaAgent = await lettaClient.agents.retrieve(sessionId);

    const session: LettaACPSession = {
      sessionId,
      lettaAgentId: sessionId,
      cwd,
      mcpServers,
      pendingPrompt: null,
      memoryBlocks: new Map()
    };

    await setupMCPServers(mcpServers, sessionId);
    this.sessions.set(sessionId, session);
  }

  getSession(sessionId: string): LettaACPSession | undefined {
    return this.sessions.get(sessionId);
  }
}
```

## Error Handling

```typescript
class ACPErrorHandler {
  static handleLettaError(error: any): never {
    if (error.response?.status === 401) {
      throw RequestError.authRequired({}, 'Letta API authentication failed');
    }

    if (error.response?.status === 404) {
      throw RequestError.resourceNotFound(error.message);
    }

    throw RequestError.internalError({
      message: error.message,
      stack: error.stack
    });
  }

  static async withCancellation<T>(
    operation: Promise<T>,
    abortSignal: AbortSignal
  ): Promise<T> {
    return Promise.race([
      operation,
      new Promise<never>((_, reject) => {
        abortSignal.addEventListener('abort', () => {
          reject(new Error('Operation cancelled'));
        });
      })
    ]);
  }
}
```

## Testing Strategy

### Unit Tests
- Content conversion (ACP ↔ Letta)
- Tool call mapping
- Message format transformation
- Error handling

### Integration Tests
- Full prompt cycle
- Session creation and loading
- MCP server integration
- Permission requests
- Cancellation handling

### End-to-End Tests
- Connect real code editor (Zed)
- Execute multi-turn conversations
- Test file operations
- Test terminal operations
- Test memory persistence

## Deployment

### Running the ACP Agent

```bash
# Install dependencies
npm install @agentclientprotocol/sdk @letta-ai/letta-client

# Build
npm run build

# Run agent (stdio mode for ACP)
node dist/index.js

# Or with environment variables
LETTA_API_URL=http://localhost:8283 \
LETTA_API_KEY=your-api-key \
node dist/index.js
```

### Code Editor Configuration

**Zed Example:**
```json
{
  "agents": {
    "letta": {
      "command": "/path/to/letta-acp-agent/dist/index.js",
      "args": [],
      "env": {
        "LETTA_API_URL": "http://localhost:8283",
        "LETTA_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Future Enhancements

### Phase 1: Core Implementation (Current)
- ✅ Basic ACP protocol implementation
- ✅ Session management
- ✅ Message streaming
- ✅ Tool call mapping
- ✅ File system integration

### Phase 2: Advanced Features
- Image/Audio content support (when Letta adds support)
- Agent plan mapping to ACP's plan structure
- Enhanced memory visualization
- Custom permission policies
- Multi-agent coordination via ACP

### Phase 3: Ecosystem Integration
- Publish as npm package
- Add to ACP agent registry
- Create VS Code extension
- Create Cursor integration
- Documentation and tutorials

## Success Metrics

1. **Protocol Compliance:** Pass all ACP protocol tests
2. **Performance:** < 100ms overhead vs direct Letta API calls
3. **Reliability:** Handle 1000+ messages without memory leaks
4. **Compatibility:** Work with Zed, Cursor, and VS Code
5. **User Experience:** Smooth streaming, clear tool call visualization

## Conclusion

Integrating Letta with ACP will:
- Make Letta's stateful agents available to the entire ACP ecosystem
- Leverage Letta's unique self-editing memory in any code editor
- Enable broader adoption of Letta's agent technology
- Create new possibilities for multi-agent workflows
- Maintain backward compatibility with existing Letta infrastructure

The implementation is straightforward using the ACP TypeScript SDK, with clear mappings between Letta's API and ACP's protocol methods.
