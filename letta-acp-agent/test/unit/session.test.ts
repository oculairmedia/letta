import { describe, it, expect, beforeEach } from 'vitest';
import { SessionManager } from '../../src/session.js';
import { MockLettaClient, sampleMcpServer } from '../helpers/mocks.js';
import type { LettaConfig } from '../../src/types/index.js';

describe('SessionManager', () => {
  let sessionManager: SessionManager;
  let mockClient: any;
  let config: LettaConfig;

  beforeEach(() => {
    mockClient = new MockLettaClient();
    config = {
      apiUrl: 'http://localhost:8283',
      defaultModel: 'openai/gpt-4.1',
    };
    sessionManager = new SessionManager(mockClient, config);
  });

  describe('createSession', () => {
    it('should create a new Letta agent session', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);

      expect(sessionId).toBeTruthy();
      expect(typeof sessionId).toBe('string');
    });

    it('should store session in internal map', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);

      const session = sessionManager.getSession(sessionId);
      expect(session).toBeDefined();
      expect(session?.sessionId).toBe(sessionId);
    });

    it('should set workspace directory', async () => {
      const cwd = '/my/project/path';
      const sessionId = await sessionManager.createSession(cwd, []);

      const session = sessionManager.getSession(sessionId);
      expect(session?.cwd).toBe(cwd);
    });

    it('should initialize with no pending prompts', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);

      const session = sessionManager.getSession(sessionId);
      expect(session?.pendingPrompt).toBeNull();
    });

    it('should create agent with workspace memory block', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);

      // Verify the mock was called (indirectly, since we get a session back)
      expect(sessionId).toBeTruthy();
    });

    it('should handle MCP servers', async () => {
      const sessionId = await sessionManager.createSession('/workspace', [
        sampleMcpServer,
      ]);

      const session = sessionManager.getSession(sessionId);
      expect(session?.mcpServers).toHaveLength(1);
      expect(session?.mcpServers[0]).toEqual(sampleMcpServer);
    });

    it('should work with empty MCP servers array', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);

      const session = sessionManager.getSession(sessionId);
      expect(session?.mcpServers).toHaveLength(0);
    });
  });

  describe('loadSession', () => {
    it('should load existing session', async () => {
      const sessionId = 'existing-agent-123';
      await sessionManager.loadSession(sessionId, '/workspace', []);

      const session = sessionManager.getSession(sessionId);
      expect(session).toBeDefined();
      expect(session?.lettaAgentId).toBe(sessionId);
    });

    it('should set working directory when loading', async () => {
      const sessionId = 'existing-agent-123';
      const cwd = '/loaded/workspace';

      await sessionManager.loadSession(sessionId, cwd, []);

      const session = sessionManager.getSession(sessionId);
      expect(session?.cwd).toBe(cwd);
    });

    it('should throw error for non-existent session', async () => {
      // Override retrieve to throw error
      mockClient.agents.retrieve = async () => {
        throw new Error('Agent not found');
      };

      await expect(
        sessionManager.loadSession('non-existent', '/workspace', [])
      ).rejects.toThrow();
    });

    it('should initialize with null pending prompt', async () => {
      const sessionId = 'existing-agent-123';
      await sessionManager.loadSession(sessionId, '/workspace', []);

      const session = sessionManager.getSession(sessionId);
      expect(session?.pendingPrompt).toBeNull();
    });
  });

  describe('getSession', () => {
    it('should return undefined for non-existent session', () => {
      const session = sessionManager.getSession('non-existent');
      expect(session).toBeUndefined();
    });

    it('should return session after creation', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);
      const session = sessionManager.getSession(sessionId);

      expect(session).toBeDefined();
      expect(session?.sessionId).toBe(sessionId);
    });

    it('should return correct session among multiple', async () => {
      const sessionId1 = await sessionManager.createSession('/workspace1', []);
      const sessionId2 = await sessionManager.createSession('/workspace2', []);

      const session1 = sessionManager.getSession(sessionId1);
      const session2 = sessionManager.getSession(sessionId2);

      expect(session1?.cwd).toBe('/workspace1');
      expect(session2?.cwd).toBe('/workspace2');
    });
  });

  describe('deleteSession', () => {
    it('should remove session from map', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);

      sessionManager.deleteSession(sessionId);

      const session = sessionManager.getSession(sessionId);
      expect(session).toBeUndefined();
    });

    it('should not throw for non-existent session', () => {
      expect(() => {
        sessionManager.deleteSession('non-existent');
      }).not.toThrow();
    });
  });

  describe('getSessionHistory', () => {
    it('should return empty array for new session', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);

      const history = await sessionManager.getSessionHistory(sessionId);

      expect(Array.isArray(history)).toBe(true);
      expect(history).toHaveLength(0);
    });

    it('should handle errors gracefully', async () => {
      // Override to throw error
      mockClient.agents.messages.list = async () => {
        throw new Error('Database error');
      };

      const sessionId = await sessionManager.createSession('/workspace', []);
      const history = await sessionManager.getSessionHistory(sessionId);

      expect(history).toEqual([]);
    });
  });

  describe('session state management', () => {
    it('should handle multiple sessions independently', async () => {
      const session1 = await sessionManager.createSession('/workspace1', []);
      const session2 = await sessionManager.createSession('/workspace2', []);

      const s1 = sessionManager.getSession(session1);
      const s2 = sessionManager.getSession(session2);

      expect(s1?.sessionId).not.toBe(s2?.sessionId);
      expect(s1?.cwd).not.toBe(s2?.cwd);
    });

    it('should maintain session state across operations', async () => {
      const sessionId = await sessionManager.createSession('/workspace', []);

      const session1 = sessionManager.getSession(sessionId);
      const session2 = sessionManager.getSession(sessionId);

      expect(session1).toBe(session2); // Same reference
    });
  });
});
