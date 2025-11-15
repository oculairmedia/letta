# Letta ACP Integration - Implementation Plan

## Overview

This document provides a step-by-step implementation plan for wrapping Letta with the Agent Client Protocol (ACP), enabling Letta agents to work with any ACP-compatible code editor.

## Prerequisites

- Node.js 18+ and npm/yarn
- TypeScript 5.x
- Access to Letta API (self-hosted or cloud)
- Familiarity with ACP protocol (see ACP_INTEGRATION_ANALYSIS.md)

## Phase 1: Project Setup

### Step 1.1: Create Project Directory

```bash
cd /home/user/letta
mkdir letta-acp-agent
cd letta-acp-agent
```

### Step 1.2: Initialize npm Project

```bash
npm init -y
```

### Step 1.3: Install Dependencies

```bash
# Core dependencies
npm install @agentclientprotocol/sdk @letta-ai/letta-client zod

# Development dependencies
npm install -D typescript @types/node tsx vitest
```

### Step 1.4: Configure TypeScript

Create `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### Step 1.5: Update package.json

```json
{
  "name": "letta-acp-agent",
  "version": "0.1.0",
  "description": "ACP wrapper for Letta agents",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "bin": {
    "letta-acp-agent": "./dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsx watch src/index.ts",
    "test": "vitest",
    "test:once": "vitest run",
    "start": "node dist/index.js"
  },
  "keywords": ["letta", "acp", "agent", "ai", "code-editor"],
  "author": "",
  "license": "MIT"
}
```

## Phase 2: Core Implementation

### Step 2.1: Create Project Structure

```bash
mkdir -p src/{converters,letta,utils}
mkdir -p src/types
mkdir -p examples
mkdir -p test
```

### Step 2.2: Implement Type Definitions

Create `src/types/index.ts`:

```typescript
import type * as acp from '@agentclientprotocol/sdk';
import type { LettaClient } from '@letta-ai/letta-client';

export interface LettaACPSession {
  sessionId: string;
  lettaAgentId: string;
  cwd: string;
  mcpServers: acp.McpServerConfig[];
  pendingPrompt: AbortController | null;
  memoryBlocks: Map<string, MemoryBlock>;
}

export interface MemoryBlock {
  label: string;
  value: string;
  description?: string;
}

export interface LettaConfig {
  apiUrl: string;
  apiKey?: string;
  defaultModel: string;
}

export interface ToolMapping {
  lettaToolName: string;
  acpKind: acp.ToolKind;
  requiresPermission: boolean;
}
```

### Step 2.3: Implement Session Manager

Create `src/session.ts`:

```typescript
import type * as acp from '@agentclientprotocol/sdk';
import type { LettaClient } from '@letta-ai/letta-client';
import type { LettaACPSession } from './types/index.js';

export class SessionManager {
  private sessions: Map<string, LettaACPSession> = new Map();

  constructor(private lettaClient: LettaClient) {}

  async createSession(
    cwd: string,
    mcpServers: acp.McpServerConfig[]
  ): Promise<string> {
    // Implementation here
    throw new Error('Not implemented');
  }

  async loadSession(
    sessionId: string,
    cwd: string,
    mcpServers: acp.McpServerConfig[]
  ): Promise<void> {
    // Implementation here
    throw new Error('Not implemented');
  }

  getSession(sessionId: string): LettaACPSession | undefined {
    return this.sessions.get(sessionId);
  }

  deleteSession(sessionId: string): void {
    this.sessions.delete(sessionId);
  }
}
```

### Step 2.4: Implement Content Converters

Create `src/converters/content.ts`:

```typescript
import type * as acp from '@agentclientprotocol/sdk';

export function convertACPContentToLettaMessage(
  content: acp.ContentBlock[]
): any {
  // Convert ACP content blocks to Letta message format
  throw new Error('Not implemented');
}

export function convertLettaMessageToACPContent(
  lettaMessage: any
): acp.ContentBlock[] {
  // Convert Letta message to ACP content blocks
  throw new Error('Not implemented');
}
```

### Step 2.5: Implement Tool Converters

Create `src/converters/tools.ts`:

```typescript
import type * as acp from '@agentclientprotocol/sdk';
import type { ToolMapping } from '../types/index.js';

export const TOOL_MAPPINGS: ToolMapping[] = [
  { lettaToolName: 'open_file', acpKind: 'read', requiresPermission: false },
  { lettaToolName: 'grep_file', acpKind: 'search', requiresPermission: false },
  { lettaToolName: 'search_file', acpKind: 'search', requiresPermission: false },
  { lettaToolName: 'run_code', acpKind: 'execute', requiresPermission: true },
  { lettaToolName: 'web_search', acpKind: 'fetch', requiresPermission: true },
  { lettaToolName: 'edit_file', acpKind: 'edit', requiresPermission: true },
  { lettaToolName: 'delete_file', acpKind: 'delete', requiresPermission: true },
  { lettaToolName: 'move_file', acpKind: 'move', requiresPermission: true }
];

export function mapLettaToolKindToACP(toolName: string): acp.ToolKind {
  const mapping = TOOL_MAPPINGS.find(m => m.lettaToolName === toolName);
  return mapping?.acpKind || 'other';
}

export function shouldRequestPermission(toolName: string): boolean {
  const mapping = TOOL_MAPPINGS.find(m => m.lettaToolName === toolName);
  return mapping?.requiresPermission ?? false;
}

export function convertLettaToolCallToACP(lettaTool: any): acp.ToolCallUpdate {
  // Convert Letta tool call to ACP format
  throw new Error('Not implemented');
}
```

### Step 2.6: Implement Main Agent Class

Create `src/agent.ts`:

```typescript
#!/usr/bin/env node

import * as acp from '@agentclientprotocol/sdk';
import { LettaClient } from '@letta-ai/letta-client';
import { SessionManager } from './session.js';
import type { LettaConfig } from './types/index.js';

export class LettaACPAgent implements acp.Agent {
  private connection: acp.AgentSideConnection;
  private lettaClient: LettaClient;
  private sessionManager: SessionManager;
  private config: LettaConfig;

  constructor(
    connection: acp.AgentSideConnection,
    config: LettaConfig
  ) {
    this.connection = connection;
    this.config = config;
    this.lettaClient = new LettaClient({
      baseUrl: config.apiUrl,
      token: config.apiKey
    });
    this.sessionManager = new SessionManager(this.lettaClient);
  }

  async initialize(
    params: acp.InitializeRequest
  ): Promise<acp.InitializeResponse> {
    return {
      protocolVersion: acp.PROTOCOL_VERSION,
      agentCapabilities: {
        loadSession: true,
        promptCapabilities: {
          embeddedContext: true,
          image: false,
          audio: false
        },
        mcpCapabilities: {
          http: true,
          sse: false
        }
      },
      agentInfo: {
        name: 'letta',
        title: 'Letta Agent',
        version: '0.1.0'
      }
    };
  }

  async newSession(
    params: acp.NewSessionRequest
  ): Promise<acp.NewSessionResponse> {
    const sessionId = await this.sessionManager.createSession(
      params.cwd,
      params.mcpServers || []
    );

    return { sessionId };
  }

  async loadSession(
    params: acp.LoadSessionRequest
  ): Promise<acp.LoadSessionResponse> {
    await this.sessionManager.loadSession(
      params.sessionId,
      params.cwd,
      params.mcpServers || []
    );

    // Stream conversation history
    // TODO: Implement history streaming

    return {};
  }

  async authenticate(
    params: acp.AuthenticateRequest
  ): Promise<acp.AuthenticateResponse | void> {
    // No authentication needed for now
    return {};
  }

  async prompt(
    params: acp.PromptRequest
  ): Promise<acp.PromptResponse> {
    // TODO: Implement prompt processing
    throw new Error('Not implemented');
  }

  async cancel(params: acp.CancelNotification): Promise<void> {
    const session = this.sessionManager.getSession(params.sessionId);
    session?.pendingPrompt?.abort();
  }
}
```

### Step 2.7: Implement Entry Point

Create `src/index.ts`:

```typescript
#!/usr/bin/env node

import * as acp from '@agentclientprotocol/sdk';
import { Readable, Writable } from 'node:stream';
import { LettaACPAgent } from './agent.js';
import type { LettaConfig } from './types/index.js';

function getConfig(): LettaConfig {
  return {
    apiUrl: process.env.LETTA_API_URL || 'http://localhost:8283',
    apiKey: process.env.LETTA_API_KEY,
    defaultModel: process.env.LETTA_DEFAULT_MODEL || 'openai/gpt-4.1'
  };
}

function main() {
  const config = getConfig();

  // Create stdio streams for ACP communication
  const input = Writable.toWeb(process.stdout);
  const output = Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>;

  // Create ACP stream
  const stream = acp.ndJsonStream(input, output);

  // Create and start agent
  new acp.AgentSideConnection(
    (conn) => new LettaACPAgent(conn, config),
    stream
  );
}

main();
```

## Phase 3: Detailed Implementation

### Step 3.1: Implement Session Creation

Update `src/session.ts`:

```typescript
async createSession(
  cwd: string,
  mcpServers: acp.McpServerConfig[]
): Promise<string> {
  // Create Letta agent
  const lettaAgent = await this.lettaClient.agents.create({
    model: 'openai/gpt-4.1',
    memoryBlocks: [
      {
        label: 'persona',
        value: 'I am a helpful AI coding assistant with persistent memory.'
      },
      {
        label: 'workspace',
        value: `Working directory: ${cwd}\n` +
               `I should focus on this workspace for file operations.`
      }
    ]
  });

  // Setup MCP servers if provided
  if (mcpServers.length > 0) {
    await this.setupMCPServers(mcpServers, lettaAgent.id);
  }

  // Create session record
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

private async setupMCPServers(
  mcpServers: acp.McpServerConfig[],
  agentId: string
): Promise<void> {
  for (const server of mcpServers) {
    // TODO: Integrate with Letta's MCP system
    console.log(`Setting up MCP server: ${server.name}`);
  }
}
```

### Step 3.2: Implement Prompt Processing

Update `src/agent.ts` prompt method:

```typescript
async prompt(
  params: acp.PromptRequest
): Promise<acp.PromptResponse> {
  const session = this.sessionManager.getSession(params.sessionId);
  if (!session) {
    throw acp.RequestError.resourceNotFound(params.sessionId);
  }

  // Cancel any pending prompt
  session.pendingPrompt?.abort();
  session.pendingPrompt = new AbortController();

  try {
    // Convert ACP content to Letta message
    const lettaMessage = convertACPContentToLettaMessage(params.prompt);

    // Create streaming response
    const stream = await this.lettaClient.agents.messages.createStream({
      agentId: session.lettaAgentId,
      messages: [lettaMessage],
      streamTokens: true
    });

    // Stream updates to client
    await this.streamLettaResponse(
      stream,
      params.sessionId,
      session.pendingPrompt.signal
    );

    return { stopReason: 'end_turn' };
  } catch (error) {
    if (session.pendingPrompt.signal.aborted) {
      return { stopReason: 'cancelled' };
    }
    throw error;
  } finally {
    session.pendingPrompt = null;
  }
}

private async streamLettaResponse(
  stream: any,
  sessionId: string,
  abortSignal: AbortSignal
): Promise<void> {
  for await (const chunk of stream) {
    if (abortSignal.aborted) {
      break;
    }

    // Convert chunk to ACP update
    const update = this.convertLettaChunkToACPUpdate(chunk);
    if (update) {
      await this.connection.sessionUpdate({
        sessionId,
        update
      });
    }
  }
}

private convertLettaChunkToACPUpdate(chunk: any): acp.SessionUpdate | null {
  // TODO: Implement conversion logic
  return null;
}
```

### Step 3.3: Implement Content Converters

Update `src/converters/content.ts`:

```typescript
export function convertACPContentToLettaMessage(
  content: acp.ContentBlock[]
): any {
  const textParts: string[] = [];
  const resources: any[] = [];

  for (const block of content) {
    if (block.type === 'text') {
      textParts.push(block.text);
    } else if (block.type === 'resource') {
      resources.push({
        uri: block.resource.uri,
        mimeType: block.resource.mimeType,
        text: block.resource.text
      });
    }
  }

  return {
    role: 'user',
    content: textParts.join('\n'),
    resources: resources.length > 0 ? resources : undefined
  };
}

export function convertLettaMessageToACPContent(
  lettaMessage: any
): acp.ContentBlock[] {
  const blocks: acp.ContentBlock[] = [];

  if (lettaMessage.content) {
    blocks.push({
      type: 'text',
      text: lettaMessage.content
    });
  }

  return blocks;
}
```

## Phase 4: Testing

### Step 4.1: Create Test Structure

```bash
mkdir -p test/{unit,integration}
```

### Step 4.2: Write Unit Tests

Create `test/unit/converters.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { convertACPContentToLettaMessage } from '../../src/converters/content.js';

describe('Content Converters', () => {
  it('should convert text content', () => {
    const acpContent = [
      { type: 'text' as const, text: 'Hello world' }
    ];

    const result = convertACPContentToLettaMessage(acpContent);
    expect(result.content).toBe('Hello world');
    expect(result.role).toBe('user');
  });

  // Add more tests
});
```

### Step 4.3: Write Integration Tests

Create `test/integration/agent.test.ts`:

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as acp from '@agentclientprotocol/sdk';
// Import agent and test helpers

describe('LettaACPAgent Integration', () => {
  // Test full workflow
});
```

## Phase 5: Documentation & Examples

### Step 5.1: Create Example

Create `examples/basic-usage.ts`:

```typescript
#!/usr/bin/env node

import * as acp from '@agentclientprotocol/sdk';
import { Readable, Writable } from 'node:stream';
import { LettaACPAgent } from '../src/agent.js';

// Example demonstrating basic usage
async function main() {
  // Setup would go here
}

main().catch(console.error);
```

### Step 5.2: Create README

Create `README.md`:

```markdown
# Letta ACP Agent

ACP wrapper for Letta agents, enabling Letta's stateful AI agents in any ACP-compatible code editor.

## Features

- Full ACP protocol support
- Persistent memory with Letta's memory blocks
- MCP tool integration
- Streaming responses
- Permission system for sensitive operations

## Installation

\`\`\`bash
npm install letta-acp-agent
\`\`\`

## Usage

\`\`\`bash
# With Letta Cloud
LETTA_API_KEY=your-key letta-acp-agent

# With self-hosted Letta
LETTA_API_URL=http://localhost:8283 letta-acp-agent
\`\`\`

## Configuration

See documentation for details.
```

## Phase 6: Build & Deploy

### Step 6.1: Build Project

```bash
npm run build
```

### Step 6.2: Test with Code Editor

Configure in your code editor (e.g., Zed):

```json
{
  "agents": {
    "letta": {
      "command": "/path/to/letta-acp-agent/dist/index.js",
      "env": {
        "LETTA_API_URL": "http://localhost:8283"
      }
    }
  }
}
```

### Step 6.3: Package for Distribution

```bash
# Ensure build is clean
npm run build

# Create tarball
npm pack

# Or publish to npm
npm publish
```

## Implementation Checklist

### Core Protocol
- [ ] Initialize method
- [ ] Session creation (session/new)
- [ ] Session loading (session/load)
- [ ] Prompt processing (session/prompt)
- [ ] Cancellation (session/cancel)
- [ ] Session updates

### Content Handling
- [ ] Text content conversion
- [ ] Resource content conversion
- [ ] Message format conversion
- [ ] Tool call conversion

### Tool Integration
- [ ] Tool kind mapping
- [ ] Permission requests
- [ ] Tool execution
- [ ] Tool result formatting
- [ ] Diff generation for file edits

### Session Management
- [ ] Session creation
- [ ] Session state tracking
- [ ] Session cleanup
- [ ] History streaming

### MCP Integration
- [ ] MCP server setup
- [ ] Tool discovery
- [ ] Tool attachment

### Error Handling
- [ ] Request errors
- [ ] Letta API errors
- [ ] Cancellation errors
- [ ] Resource not found

### Testing
- [ ] Unit tests for converters
- [ ] Unit tests for tool mapping
- [ ] Integration tests
- [ ] End-to-end tests with code editor

### Documentation
- [ ] README with usage instructions
- [ ] API documentation
- [ ] Configuration guide
- [ ] Examples
- [ ] Troubleshooting guide

## Next Steps

1. Start with Phase 1: Project Setup
2. Implement core protocol methods (Phase 2)
3. Add detailed implementations (Phase 3)
4. Write tests (Phase 4)
5. Create documentation (Phase 5)
6. Test with real code editors (Phase 6)

## Resources

- [ACP Protocol Documentation](https://agentclientprotocol.com/)
- [ACP TypeScript SDK](https://github.com/agentclientprotocol/typescript-sdk)
- [Letta Documentation](https://docs.letta.com/)
- [Letta TypeScript Client](https://github.com/letta-ai/letta-node)

## Support

For issues or questions:
- Create an issue in the repository
- Check the troubleshooting guide
- Review ACP and Letta documentation
