import type * as acp from '@agentclientprotocol/sdk';
import type { LettaClient } from '@letta-ai/letta-client';
import type { LettaACPSession, LettaConfig } from './types/index.js';

/**
 * SessionManager handles creation, loading, and lifecycle of Letta ACP sessions
 */
export class SessionManager {
  private sessions: Map<string, LettaACPSession> = new Map();

  constructor(
    private lettaClient: LettaClient,
    private config: LettaConfig
  ) {}

  /**
   * Create a new Letta agent session
   */
  async createSession(
    cwd: string,
    mcpServers: acp.McpServer[]
  ): Promise<string> {
    console.error(`[SessionManager] Creating new session for cwd: ${cwd}`);

    // Create Letta agent with workspace context
    const lettaAgent = await this.lettaClient.agents.create({
      model: this.config.defaultModel,
      memoryBlocks: [
        {
          label: 'persona',
          value:
            'I am a helpful AI coding assistant with persistent memory. ' +
            'I can help with code review, debugging, refactoring, and implementation. ' +
            'I have access to files and can execute commands.',
        },
        {
          label: 'workspace',
          value:
            `Working directory: ${cwd}\n` +
            `I should focus file operations within this workspace. ` +
            `I will ask for permission before making destructive changes.`,
        },
      ],
      tools: ['web_search', 'run_code'],
    });

    console.error(`[SessionManager] Created Letta agent: ${lettaAgent.id}`);

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
      memoryBlocks: new Map(),
    };

    this.sessions.set(lettaAgent.id, session);

    console.error(`[SessionManager] Session created: ${lettaAgent.id}`);
    return lettaAgent.id;
  }

  /**
   * Load an existing Letta agent session
   */
  async loadSession(
    sessionId: string,
    cwd: string,
    mcpServers: acp.McpServer[]
  ): Promise<void> {
    console.error(`[SessionManager] Loading session: ${sessionId}`);

    // Verify agent exists
    try {
      const lettaAgent = await this.lettaClient.agents.retrieve(sessionId);
      console.error(`[SessionManager] Found existing agent: ${lettaAgent.id}`);
    } catch (error) {
      console.error(`[SessionManager] Failed to load agent: ${error}`);
      throw new Error(`Session ${sessionId} not found`);
    }

    // Setup MCP servers
    if (mcpServers.length > 0) {
      await this.setupMCPServers(mcpServers, sessionId);
    }

    // Create session record
    const session: LettaACPSession = {
      sessionId,
      lettaAgentId: sessionId,
      cwd,
      mcpServers,
      pendingPrompt: null,
      memoryBlocks: new Map(),
    };

    this.sessions.set(sessionId, session);
    console.error(`[SessionManager] Session loaded: ${sessionId}`);
  }

  /**
   * Get an existing session
   */
  getSession(sessionId: string): LettaACPSession | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * Delete a session
   */
  deleteSession(sessionId: string): void {
    console.error(`[SessionManager] Deleting session: ${sessionId}`);
    this.sessions.delete(sessionId);
  }

  /**
   * Get conversation history for a session
   */
  async getSessionHistory(sessionId: string): Promise<any[]> {
    try {
      const messages = await this.lettaClient.agents.messages.list(sessionId, {
        limit: 100,
      });
      return messages;
    } catch (error) {
      console.error(`[SessionManager] Failed to get history: ${error}`);
      return [];
    }
  }

  /**
   * Setup MCP servers for an agent
   */
  private async setupMCPServers(
    mcpServers: acp.McpServer[],
    agentId: string
  ): Promise<void> {
    console.error(
      `[SessionManager] Setting up ${mcpServers.length} MCP servers for agent ${agentId}`
    );

    for (const server of mcpServers) {
      try {
        console.error(`[SessionManager] Setting up MCP server: ${server.name}`);

        // In a full implementation, we would:
        // 1. List tools from the MCP server using Letta's MCP client
        // 2. Add each tool to Letta's tool registry
        // 3. Attach tools to the agent

        // For now, log that we would set this up
        // This requires Letta's MCP integration to be fully available
        console.error(
          `[SessionManager] MCP server ${server.name} would be integrated here`
        );
      } catch (error) {
        console.error(
          `[SessionManager] Failed to setup MCP server ${server.name}: ${error}`
        );
      }
    }
  }
}
