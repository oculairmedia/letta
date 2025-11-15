import * as acp from '@agentclientprotocol/sdk';
import { LettaClient } from '@letta-ai/letta-client';
import { SessionManager } from './session.js';
import { convertACPContentToLettaMessage, convertLettaMessageToACPContent } from './converters/content.js';
import type { LettaConfig } from './types/index.js';

/**
 * LettaACPAgent implements the ACP Agent interface for Letta
 */
export class LettaACPAgent implements acp.Agent {
  private connection: acp.AgentSideConnection;
  private lettaClient: LettaClient;
  private sessionManager: SessionManager;

  constructor(connection: acp.AgentSideConnection, config: LettaConfig) {
    this.connection = connection;

    console.error('[LettaACPAgent] Initializing agent...');
    console.error(`[LettaACPAgent] Letta API URL: ${config.apiUrl}`);
    console.error(`[LettaACPAgent] Default model: ${config.defaultModel}`);

    // Initialize Letta client
    this.lettaClient = new LettaClient({
      baseUrl: config.apiUrl,
      token: config.apiKey,
    });

    this.sessionManager = new SessionManager(this.lettaClient, config);

    console.error('[LettaACPAgent] Agent initialized');
  }

  /**
   * Initialize - negotiate protocol version and capabilities
   */
  async initialize(params: acp.InitializeRequest): Promise<acp.InitializeResponse> {
    console.error('[LettaACPAgent] Handling initialize request');
    console.error(`[LettaACPAgent] Client protocol version: ${params.protocolVersion}`);
    console.error(`[LettaACPAgent] Client capabilities:`, params.clientCapabilities);

    return {
      protocolVersion: acp.PROTOCOL_VERSION,
      agentCapabilities: {
        loadSession: true,
        promptCapabilities: {
          embeddedContext: true,
          image: false,
          audio: false,
        },
        mcpCapabilities: {
          http: true,
          sse: false,
        },
      },
      agentInfo: {
        name: 'letta',
        title: 'Letta Agent',
        version: '0.1.0',
      },
    };
  }

  /**
   * Create a new session
   */
  async newSession(params: acp.NewSessionRequest): Promise<acp.NewSessionResponse> {
    console.error('[LettaACPAgent] Creating new session');
    console.error(`[LettaACPAgent] Working directory: ${params.cwd}`);
    console.error(`[LettaACPAgent] MCP servers: ${params.mcpServers?.length || 0}`);

    const sessionId = await this.sessionManager.createSession(
      params.cwd,
      params.mcpServers || []
    );

    console.error(`[LettaACPAgent] New session created: ${sessionId}`);

    return { sessionId };
  }

  /**
   * Load an existing session
   */
  async loadSession(params: acp.LoadSessionRequest): Promise<acp.LoadSessionResponse> {
    console.error(`[LettaACPAgent] Loading session: ${params.sessionId}`);

    await this.sessionManager.loadSession(
      params.sessionId,
      params.cwd,
      params.mcpServers || []
    );

    // Stream conversation history to client
    const history = await this.sessionManager.getSessionHistory(params.sessionId);

    console.error(`[LettaACPAgent] Streaming ${history.length} messages from history`);

    for (const message of history) {
      try {
        const content = convertLettaMessageToACPContent({
          role: message.role || 'assistant',
          content: message.content || '',
        });

        await this.connection.sessionUpdate({
          sessionId: params.sessionId,
          update: {
            sessionUpdate:
              message.role === 'user' ? 'user_message_chunk' : 'agent_message_chunk',
            content: content[0] || { type: 'text', text: '' },
          },
        });
      } catch (error) {
        console.error(`[LettaACPAgent] Error streaming message: ${error}`);
      }
    }

    console.error('[LettaACPAgent] Session loaded');
    return {};
  }

  /**
   * Authenticate - not needed for Letta
   */
  async authenticate(_params: acp.AuthenticateRequest): Promise<acp.AuthenticateResponse | void> {
    console.error('[LettaACPAgent] Authentication not required');
    return {};
  }

  /**
   * Process a prompt
   */
  async prompt(params: acp.PromptRequest): Promise<acp.PromptResponse> {
    console.error(`[LettaACPAgent] Processing prompt for session ${params.sessionId}`);

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
      console.error(`[LettaACPAgent] Sending message to Letta: ${lettaMessage.content.substring(0, 100)}...`);

      // Send message to Letta agent
      const response = await this.lettaClient.agents.messages.create(session.lettaAgentId, {
        messages: [
          {
            role: lettaMessage.role,
            content: lettaMessage.content,
          },
        ],
      });

      console.error(`[LettaACPAgent] Received response with ${response.messages.length} messages`);

      // Stream response messages to client
      for (const message of response.messages) {
        if (session.pendingPrompt.signal.aborted) {
          console.error('[LettaACPAgent] Prompt cancelled by client');
          return { stopReason: 'cancelled' };
        }

        // Send message chunk
        if ('content' in message && message.content) {
          const contentText = typeof message.content === 'string'
            ? message.content
            : JSON.stringify(message.content);

          await this.connection.sessionUpdate({
            sessionId: params.sessionId,
            update: {
              sessionUpdate: 'agent_message_chunk',
              content: {
                type: 'text',
                text: contentText,
              },
            },
          });
        }

        // Handle tool calls if present
        if ('tool_calls' in message && message.tool_calls && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
          for (const toolCall of message.tool_calls) {
            await this.handleToolCall(toolCall, params.sessionId);
          }
        }
      }

      console.error('[LettaACPAgent] Prompt completed');
      return { stopReason: 'end_turn' };
    } catch (error) {
      if (session.pendingPrompt.signal.aborted) {
        console.error('[LettaACPAgent] Prompt cancelled');
        return { stopReason: 'cancelled' };
      }

      console.error(`[LettaACPAgent] Error processing prompt: ${error}`);
      throw error;
    } finally {
      session.pendingPrompt = null;
    }
  }

  /**
   * Handle a tool call from Letta
   */
  private async handleToolCall(toolCall: any, sessionId: string): Promise<void> {
    console.error(`[LettaACPAgent] Handling tool call: ${toolCall.function?.name}`);

    const toolCallId = toolCall.id || `tool_${Date.now()}`;

    // Send tool call notification
    await this.connection.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: 'tool_call',
        toolCallId,
        title: toolCall.function?.name || 'Tool call',
        kind: 'other',
        status: 'pending',
        rawInput: toolCall.function?.arguments || {},
      },
    });

    // Update to in_progress
    await this.connection.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: 'tool_call_update',
        toolCallId,
        status: 'in_progress',
      },
    });

    // Tool execution would happen here in Letta
    // For now, we just mark as completed
    await this.connection.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: 'tool_call_update',
        toolCallId,
        status: 'completed',
        content: [
          {
            type: 'content',
            content: {
              type: 'text',
              text: 'Tool executed',
            },
          },
        ],
      },
    });
  }

  /**
   * Cancel an ongoing prompt
   */
  async cancel(params: acp.CancelNotification): Promise<void> {
    console.error(`[LettaACPAgent] Cancelling prompt for session ${params.sessionId}`);

    const session = this.sessionManager.getSession(params.sessionId);
    if (session?.pendingPrompt) {
      session.pendingPrompt.abort();
      console.error('[LettaACPAgent] Prompt cancelled');
    }
  }
}
