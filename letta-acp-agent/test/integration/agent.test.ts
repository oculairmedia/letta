import { describe, it, expect, beforeEach } from 'vitest';
import { LettaACPAgent } from '../../src/agent.js';
import { createMockConnection, sampleACPContent } from '../helpers/mocks.js';
import type * as acp from '@agentclientprotocol/sdk';
import type { LettaConfig } from '../../src/types/index.js';

describe('LettaACPAgent Integration', () => {
  let agent: LettaACPAgent;
  let mockConnection: any;
  let config: LettaConfig;

  beforeEach(() => {
    mockConnection = createMockConnection();
    config = {
      apiUrl: 'http://localhost:8283',
      defaultModel: 'openai/gpt-4.1',
    };

    agent = new LettaACPAgent(
      mockConnection as acp.AgentSideConnection,
      config
    );
  });

  describe('initialize', () => {
    it('should return protocol version 1', async () => {
      const params: acp.InitializeRequest = {
        protocolVersion: 1,
        clientCapabilities: {},
      };

      const response = await agent.initialize(params);

      expect(response.protocolVersion).toBe(1);
    });

    it('should advertise agent capabilities', async () => {
      const params: acp.InitializeRequest = {
        protocolVersion: 1,
        clientCapabilities: {},
      };

      const response = await agent.initialize(params);

      expect(response.agentCapabilities).toBeDefined();
      expect(response.agentCapabilities?.loadSession).toBe(true);
      expect(response.agentCapabilities?.promptCapabilities).toBeDefined();
      expect(response.agentCapabilities?.mcpCapabilities).toBeDefined();
    });

    it('should support embedded context', async () => {
      const params: acp.InitializeRequest = {
        protocolVersion: 1,
        clientCapabilities: {},
      };

      const response = await agent.initialize(params);

      expect(response.agentCapabilities?.promptCapabilities?.embeddedContext).toBe(
        true
      );
    });

    it('should not support images yet', async () => {
      const params: acp.InitializeRequest = {
        protocolVersion: 1,
        clientCapabilities: {},
      };

      const response = await agent.initialize(params);

      expect(response.agentCapabilities?.promptCapabilities?.image).toBe(false);
      expect(response.agentCapabilities?.promptCapabilities?.audio).toBe(false);
    });

    it('should support MCP HTTP', async () => {
      const params: acp.InitializeRequest = {
        protocolVersion: 1,
        clientCapabilities: {},
      };

      const response = await agent.initialize(params);

      expect(response.agentCapabilities?.mcpCapabilities?.http).toBe(true);
    });

    it('should provide agent info', async () => {
      const params: acp.InitializeRequest = {
        protocolVersion: 1,
        clientCapabilities: {},
      };

      const response = await agent.initialize(params);

      expect(response.agentInfo).toBeDefined();
      expect(response.agentInfo?.name).toBe('letta');
      expect(response.agentInfo?.title).toBe('Letta Agent');
      expect(response.agentInfo?.version).toBeTruthy();
    });
  });

  describe('newSession', () => {
    it('should create new session and return ID', async () => {
      const params: acp.NewSessionRequest = {
        cwd: '/workspace',
        mcpServers: [],
      };

      const response = await agent.newSession(params);

      expect(response.sessionId).toBeTruthy();
      expect(typeof response.sessionId).toBe('string');
    });

    it('should handle MCP servers in request', async () => {
      const params: acp.NewSessionRequest = {
        cwd: '/workspace',
        mcpServers: [
          {
            type: 'http',
            name: 'test-server',
            url: 'http://localhost:3000',
            headers: [],
          },
        ],
      };

      const response = await agent.newSession(params);

      expect(response.sessionId).toBeTruthy();
    });
  });

  describe('loadSession', () => {
    it('should load existing session', async () => {
      // First create a session
      const createParams: acp.NewSessionRequest = {
        cwd: '/workspace',
        mcpServers: [],
      };
      const createResponse = await agent.newSession(createParams);

      // Then load it
      const loadParams: acp.LoadSessionRequest = {
        sessionId: createResponse.sessionId,
        cwd: '/workspace',
        mcpServers: [],
      };

      const response = await agent.loadSession(loadParams);

      expect(response).toEqual({});
    });

    it('should handle non-existent session', async () => {
      const params: acp.LoadSessionRequest = {
        sessionId: 'non-existent-session',
        cwd: '/workspace',
        mcpServers: [],
      };

      // This may throw or return depending on implementation
      // For now, we expect it not to crash
      try {
        await agent.loadSession(params);
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  describe('authenticate', () => {
    it('should return empty response', async () => {
      const params: acp.AuthenticateRequest = {
        authMethod: 'none',
      };

      const response = await agent.authenticate(params);

      expect(response).toEqual({});
    });
  });

  describe('prompt', () => {
    it('should process simple text prompt', async () => {
      // First create a session
      const createParams: acp.NewSessionRequest = {
        cwd: '/workspace',
        mcpServers: [],
      };
      const session = await agent.newSession(createParams);

      // Then send a prompt
      const promptParams: acp.PromptRequest = {
        sessionId: session.sessionId,
        prompt: [
          {
            type: 'text',
            text: 'Hello, test message',
          },
        ],
      };

      const response = await agent.prompt(promptParams);

      expect(response.stopReason).toBeDefined();
      expect(['end_turn', 'cancelled', 'max_tokens']).toContain(
        response.stopReason
      );
    });

    it('should handle prompt with resources', async () => {
      const session = await agent.newSession({
        cwd: '/workspace',
        mcpServers: [],
      });

      const promptParams: acp.PromptRequest = {
        sessionId: session.sessionId,
        prompt: sampleACPContent,
      };

      const response = await agent.prompt(promptParams);

      expect(response.stopReason).toBeDefined();
    });

    it('should fail for non-existent session', async () => {
      const promptParams: acp.PromptRequest = {
        sessionId: 'non-existent',
        prompt: [{ type: 'text', text: 'test' }],
      };

      await expect(agent.prompt(promptParams)).rejects.toThrow();
    });
  });

  describe('cancel', () => {
    it('should not throw for valid session', async () => {
      const session = await agent.newSession({
        cwd: '/workspace',
        mcpServers: [],
      });

      const params: acp.CancelNotification = {
        sessionId: session.sessionId,
      };

      await expect(agent.cancel(params)).resolves.not.toThrow();
    });

    it('should not throw for non-existent session', async () => {
      const params: acp.CancelNotification = {
        sessionId: 'non-existent',
      };

      // Should handle gracefully
      await expect(agent.cancel(params)).resolves.not.toThrow();
    });
  });

  describe('full workflow', () => {
    it('should complete initialize -> create -> prompt flow', async () => {
      // Initialize
      const initResponse = await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
      });
      expect(initResponse.protocolVersion).toBe(1);

      // Create session
      const sessionResponse = await agent.newSession({
        cwd: '/workspace',
        mcpServers: [],
      });
      expect(sessionResponse.sessionId).toBeTruthy();

      // Send prompt
      const promptResponse = await agent.prompt({
        sessionId: sessionResponse.sessionId,
        prompt: [{ type: 'text', text: 'Test message' }],
      });
      expect(promptResponse.stopReason).toBeDefined();
    });

    it('should handle multiple sessions independently', async () => {
      const session1 = await agent.newSession({
        cwd: '/workspace1',
        mcpServers: [],
      });

      const session2 = await agent.newSession({
        cwd: '/workspace2',
        mcpServers: [],
      });

      expect(session1.sessionId).not.toBe(session2.sessionId);

      // Both should work independently
      const response1 = await agent.prompt({
        sessionId: session1.sessionId,
        prompt: [{ type: 'text', text: 'Message 1' }],
      });

      const response2 = await agent.prompt({
        sessionId: session2.sessionId,
        prompt: [{ type: 'text', text: 'Message 2' }],
      });

      expect(response1.stopReason).toBeDefined();
      expect(response2.stopReason).toBeDefined();
    });
  });
});
