import { describe, it, expect } from 'vitest';
import {
  convertACPContentToLettaMessage,
  convertLettaMessageToACPContent,
  convertLettaChunkToACPUpdate,
  createDiffContent,
  formatToolResultAsContent,
} from '../../src/converters/content.js';
import type * as acp from '@agentclientprotocol/sdk';

describe('Content Converters', () => {
  describe('convertACPContentToLettaMessage', () => {
    it('should convert simple text content', () => {
      const acpContent: acp.ContentBlock[] = [
        { type: 'text', text: 'Hello world' },
      ];

      const result = convertACPContentToLettaMessage(acpContent);

      expect(result.role).toBe('user');
      expect(result.content).toBe('Hello world');
    });

    it('should convert multiple text blocks', () => {
      const acpContent: acp.ContentBlock[] = [
        { type: 'text', text: 'First line' },
        { type: 'text', text: 'Second line' },
      ];

      const result = convertACPContentToLettaMessage(acpContent);

      expect(result.role).toBe('user');
      expect(result.content).toContain('First line');
      expect(result.content).toContain('Second line');
    });

    it('should convert resource content with text', () => {
      const acpContent: acp.ContentBlock[] = [
        {
          type: 'resource',
          resource: {
            uri: 'file:///test.txt',
            mimeType: 'text/plain',
            text: 'File content here',
          },
        },
      ];

      const result = convertACPContentToLettaMessage(acpContent);

      expect(result.content).toContain('file:///test.txt');
      expect(result.content).toContain('File content here');
    });

    it('should convert resource content without text', () => {
      const acpContent: acp.ContentBlock[] = [
        {
          type: 'resource',
          resource: {
            uri: 'file:///binary.dat',
            mimeType: 'application/octet-stream',
            data: 'base64data',
          },
        },
      ];

      const result = convertACPContentToLettaMessage(acpContent);

      expect(result.content).toContain('file:///binary.dat');
    });

    it('should convert resource_link content', () => {
      const acpContent: acp.ContentBlock[] = [
        {
          type: 'resource_link',
          uri: 'file:///linked.txt',
        },
      ];

      const result = convertACPContentToLettaMessage(acpContent);

      expect(result.content).toContain('file:///linked.txt');
    });

    it('should handle image content with placeholder', () => {
      const acpContent: acp.ContentBlock[] = [
        {
          type: 'image',
          mimeType: 'image/png',
          data: 'base64imagedata',
        },
      ];

      const result = convertACPContentToLettaMessage(acpContent);

      expect(result.content).toContain('Image content not supported');
    });

    it('should handle audio content with placeholder', () => {
      const acpContent: acp.ContentBlock[] = [
        {
          type: 'audio',
          mimeType: 'audio/mp3',
          data: 'base64audiodata',
        },
      ];

      const result = convertACPContentToLettaMessage(acpContent);

      expect(result.content).toContain('Audio content not supported');
    });

    it('should handle mixed content types', () => {
      const acpContent: acp.ContentBlock[] = [
        { type: 'text', text: 'Question:' },
        {
          type: 'resource',
          resource: {
            uri: 'file:///code.js',
            mimeType: 'text/javascript',
            text: 'console.log("hello");',
          },
        },
        { type: 'text', text: 'What does this do?' },
      ];

      const result = convertACPContentToLettaMessage(acpContent);

      expect(result.content).toContain('Question:');
      expect(result.content).toContain('file:///code.js');
      expect(result.content).toContain('console.log("hello");');
      expect(result.content).toContain('What does this do?');
    });
  });

  describe('convertLettaMessageToACPContent', () => {
    it('should convert simple message to text content', () => {
      const message = {
        role: 'assistant' as const,
        content: 'Hello from Letta',
      };

      const result = convertLettaMessageToACPContent(message);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        type: 'text',
        text: 'Hello from Letta',
      });
    });

    it('should handle empty content', () => {
      const message = {
        role: 'assistant' as const,
        content: '',
      };

      const result = convertLettaMessageToACPContent(message);

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        type: 'text',
        text: '',
      });
    });
  });

  describe('convertLettaChunkToACPUpdate', () => {
    it('should convert message chunk', () => {
      const chunk = {
        type: 'message',
        content: 'Chunk of text',
      };

      const result = convertLettaChunkToACPUpdate(chunk, 'session-123');

      expect(result).not.toBeNull();
      expect(result?.sessionId).toBe('session-123');
      expect(result?.update.sessionUpdate).toBe('agent_message_chunk');
      if (result?.update.sessionUpdate === 'agent_message_chunk') {
        expect(result.update.content.type).toBe('text');
        expect(result.update.content.text).toBe('Chunk of text');
      }
    });

    it('should convert tool call chunk', () => {
      const chunk = {
        type: 'tool_call',
        tool_call: {
          id: 'tool-123',
          function: {
            name: 'search',
            arguments: { query: 'test' },
          },
        },
      };

      const result = convertLettaChunkToACPUpdate(chunk, 'session-123');

      expect(result).not.toBeNull();
      expect(result?.sessionId).toBe('session-123');
      expect(result?.update.sessionUpdate).toBe('tool_call');
      if (result?.update.sessionUpdate === 'tool_call') {
        expect(result.update.toolCallId).toBe('tool-123');
        expect(result.update.title).toContain('search');
      }
    });

    it('should convert tool result chunk', () => {
      const chunk = {
        type: 'tool_result',
        tool_result: {
          tool_call_id: 'tool-123',
          content: 'Result data',
        },
      };

      const result = convertLettaChunkToACPUpdate(chunk, 'session-123');

      expect(result).not.toBeNull();
      expect(result?.sessionId).toBe('session-123');
      expect(result?.update.sessionUpdate).toBe('tool_call_update');
      if (result?.update.sessionUpdate === 'tool_call_update') {
        expect(result.update.status).toBe('completed');
      }
    });

    it('should handle null/invalid chunks', () => {
      expect(convertLettaChunkToACPUpdate(null, 'session-123')).toBeNull();
      expect(convertLettaChunkToACPUpdate(undefined, 'session-123')).toBeNull();
      expect(convertLettaChunkToACPUpdate('string', 'session-123')).toBeNull();
    });
  });

  describe('createDiffContent', () => {
    it('should create diff content block', () => {
      const result = createDiffContent(
        '/path/to/file.txt',
        'old content',
        'new content'
      );

      expect(result.type).toBe('diff');
      expect(result.path).toBe('/path/to/file.txt');
      expect(result.oldText).toBe('old content');
      expect(result.newText).toBe('new content');
    });
  });

  describe('formatToolResultAsContent', () => {
    it('should format string result', () => {
      const result = formatToolResultAsContent('Simple string result');

      expect(result).toHaveLength(1);
      expect(result[0].type).toBe('content');
      if (result[0].type === 'content') {
        expect(result[0].content.type).toBe('text');
        expect(result[0].content.text).toBe('Simple string result');
      }
    });

    it('should format diff result', () => {
      const result = formatToolResultAsContent({
        path: '/file.txt',
        oldText: 'old',
        newText: 'new',
      });

      expect(result).toHaveLength(1);
      expect(result[0].type).toBe('diff');
      if (result[0].type === 'diff') {
        expect(result[0].path).toBe('/file.txt');
      }
    });

    it('should format terminal result', () => {
      const result = formatToolResultAsContent({
        terminalId: 'term-123',
      });

      expect(result).toHaveLength(1);
      expect(result[0].type).toBe('terminal');
      if (result[0].type === 'terminal') {
        expect(result[0].terminalId).toBe('term-123');
      }
    });

    it('should format object as JSON', () => {
      const result = formatToolResultAsContent({
        status: 'success',
        data: { key: 'value' },
      });

      expect(result).toHaveLength(1);
      expect(result[0].type).toBe('content');
      if (result[0].type === 'content') {
        expect(result[0].content.type).toBe('text');
        expect(result[0].content.text).toContain('status');
        expect(result[0].content.text).toContain('success');
      }
    });

    it('should handle null result', () => {
      const result = formatToolResultAsContent(null);
      expect(result).toEqual([]);
    });

    it('should handle undefined result', () => {
      const result = formatToolResultAsContent(undefined);
      expect(result).toEqual([]);
    });
  });
});
