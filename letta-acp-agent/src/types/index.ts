import type * as acp from '@agentclientprotocol/sdk';

export interface LettaACPSession {
  sessionId: string;
  lettaAgentId: string;
  cwd: string;
  mcpServers: acp.McpServer[];
  pendingPrompt: AbortController | null;
  memoryBlocks: Map<string, MemoryBlock>;
}

export interface MemoryBlock {
  id: string;
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

export interface LettaMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface LettaStreamChunk {
  type: 'message_chunk' | 'tool_call' | 'tool_result' | 'memory_edit' | 'error';
  content?: string;
  toolCallId?: string;
  toolName?: string;
  arguments?: Record<string, unknown>;
  result?: unknown;
  memoryBlock?: string;
  oldValue?: string;
  newValue?: string;
  error?: string;
}

export interface LettaAgent {
  id: string;
  name: string;
  model: string;
  memoryBlocks: MemoryBlock[];
}

export interface LettaToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  result?: unknown;
}
