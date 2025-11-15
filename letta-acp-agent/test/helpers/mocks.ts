import type * as acp from '@agentclientprotocol/sdk';

/**
 * Mock Letta client for testing
 */
export class MockLettaClient {
  agents = {
    create: async (config: any) => ({
      id: 'test-agent-123',
      name: 'Test Agent',
      model: config.model,
      memoryBlocks: config.memoryBlocks || [],
    }),
    retrieve: async (id: string) => ({
      id,
      name: 'Test Agent',
      model: 'openai/gpt-4.1',
      memoryBlocks: [],
    }),
    messages: {
      create: async (agentId: string, params: any) => ({
        messages: [
          {
            role: 'assistant',
            content: 'This is a test response',
          },
        ],
      }),
      list: async (agentId: string, params?: any) => [],
      createStream: async (params: any) => {
        // Return async iterator for streaming
        return (async function* () {
          yield {
            type: 'message',
            content: 'Test streaming response',
          };
        })();
      },
    },
    tool: {
      attach: async (params: any) => ({}),
    },
  };

  tools = {
    listMcpToolsByServer: async (serverName: string) => [],
    addMcpTool: async (serverName: string, toolName: string) => ({
      id: 'test-tool-123',
      name: toolName,
    }),
  };
}

/**
 * Create mock ACP connection
 */
export function createMockConnection(): Partial<acp.AgentSideConnection> {
  return {
    sessionUpdate: async (params: acp.SessionNotification) => {
      // Mock implementation - just log
      console.log('[Mock] Session update:', params);
    },
    requestPermission: async (params: acp.RequestPermissionRequest) => ({
      outcome: {
        outcome: 'selected',
        optionId: 'allow',
      },
    }),
  };
}

/**
 * Sample ACP content blocks for testing
 */
export const sampleACPContent: acp.ContentBlock[] = [
  {
    type: 'text',
    text: 'Hello, this is a test message',
  },
  {
    type: 'resource',
    resource: {
      uri: 'file:///test/path/file.txt',
      mimeType: 'text/plain',
      text: 'This is file content',
    },
  },
];

/**
 * Sample Letta message for testing
 */
export const sampleLettaMessage = {
  role: 'user' as const,
  content: 'Hello, this is a test message',
};

/**
 * Sample MCP server config
 */
export const sampleMcpServer: acp.McpServer = {
  type: 'http',
  name: 'test-server',
  url: 'http://localhost:3000',
  headers: [
    {
      name: 'Authorization',
      value: 'Bearer test-token',
    },
  ],
};
