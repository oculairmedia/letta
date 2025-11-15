import type * as acp from '@agentclientprotocol/sdk';
import type { LettaMessage } from '../types/index.js';

/**
 * Convert ACP ContentBlocks to Letta message format
 */
export function convertACPContentToLettaMessage(content: acp.ContentBlock[]): LettaMessage {
  const textParts: string[] = [];
  const resources: string[] = [];

  for (const block of content) {
    switch (block.type) {
      case 'text':
        textParts.push(block.text);
        break;

      case 'resource':
        // Include resource content in the message
        if ('text' in block.resource && block.resource.text) {
          resources.push(
            `\n--- File: ${block.resource.uri} ---\n${block.resource.text}\n--- End of file ---\n`
          );
        } else {
          resources.push(`\n[Resource: ${block.resource.uri}]\n`);
        }
        break;

      case 'resource_link':
        // Include reference to linked resource
        resources.push(`\n[Reference: ${block.uri}]\n`);
        break;

      case 'image':
        // Letta doesn't support images yet, but include a placeholder
        textParts.push(`[Image content not supported]`);
        break;

      case 'audio':
        // Letta doesn't support audio yet, but include a placeholder
        textParts.push(`[Audio content not supported]`);
        break;
    }
  }

  // Combine text and resources
  const fullContent = [...textParts, ...resources].join('\n').trim();

  return {
    role: 'user',
    content: fullContent,
  };
}

/**
 * Convert Letta message to ACP ContentBlocks
 */
export function convertLettaMessageToACPContent(message: LettaMessage): acp.ContentBlock[] {
  const blocks: acp.ContentBlock[] = [];

  if (message.content) {
    blocks.push({
      type: 'text',
      text: message.content,
    });
  }

  return blocks;
}

/**
 * Convert Letta streaming chunk to ACP SessionUpdate
 */
export function convertLettaChunkToACPUpdate(
  chunk: any,
  sessionId: string
): acp.SessionNotification | null {
  // Handle different chunk types from Letta's streaming API
  if (!chunk || typeof chunk !== 'object') {
    return null;
  }

  // Message chunk (text content)
  if (chunk.type === 'message' || chunk.content) {
    return {
      sessionId,
      update: {
        sessionUpdate: 'agent_message_chunk',
        content: {
          type: 'text',
          text: chunk.content || chunk.text || '',
        },
      },
    };
  }

  // Tool call chunk
  if (chunk.type === 'tool_call' || chunk.tool_call) {
    const toolCall = chunk.tool_call || chunk;
    return {
      sessionId,
      update: {
        sessionUpdate: 'tool_call',
        toolCallId: toolCall.id || `tool_${Date.now()}`,
        title: toolCall.function?.name || 'Tool call',
        kind: 'other',
        status: 'pending',
        rawInput: toolCall.function?.arguments || {},
      },
    };
  }

  // Tool result chunk
  if (chunk.type === 'tool_result' || chunk.tool_result) {
    const result = chunk.tool_result || chunk;
    return {
      sessionId,
      update: {
        sessionUpdate: 'tool_call_update',
        toolCallId: result.tool_call_id || result.id || `tool_${Date.now()}`,
        status: 'completed',
        content: result.content
          ? [
              {
                type: 'content',
                content: {
                  type: 'text',
                  text: typeof result.content === 'string' ? result.content : JSON.stringify(result.content),
                },
              },
            ]
          : undefined,
        rawOutput: result.output || result.content,
      },
    };
  }

  // Memory edit chunk
  if (chunk.type === 'memory_edit' || chunk.memory_edit) {
    const memEdit = chunk.memory_edit || chunk;
    return {
      sessionId,
      update: {
        sessionUpdate: 'agent_message_chunk',
        content: {
          type: 'text',
          text: `[Updated memory: ${memEdit.block_label || 'memory'}]`,
        },
      },
    };
  }

  return null;
}

/**
 * Create a diff content block for file edits
 */
export function createDiffContent(path: string, oldText: string, newText: string): acp.ToolCallContent {
  return {
    type: 'diff',
    path,
    oldText,
    newText,
  };
}

/**
 * Create a terminal content block
 */
export function createTerminalContent(terminalId: string): acp.ToolCallContent {
  return {
    type: 'terminal',
    terminalId,
  };
}

/**
 * Format tool result as ACP content
 */
export function formatToolResultAsContent(result: unknown): acp.ToolCallContent[] {
  if (!result) {
    return [];
  }

  // If result is a string, return as text content
  if (typeof result === 'string') {
    return [
      {
        type: 'content',
        content: {
          type: 'text',
          text: result,
        },
      },
    ];
  }

  // If result is an object with specific fields, format accordingly
  if (typeof result === 'object' && result !== null) {
    const resultObj = result as Record<string, unknown>;

    // File edit result with diff
    if (resultObj.path && resultObj.oldText !== undefined && resultObj.newText !== undefined) {
      return [
        createDiffContent(
          resultObj.path as string,
          resultObj.oldText as string,
          resultObj.newText as string
        ),
      ];
    }

    // Terminal result
    if (resultObj.terminalId) {
      return [createTerminalContent(resultObj.terminalId as string)];
    }

    // Default: return as formatted JSON
    return [
      {
        type: 'content',
        content: {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      },
    ];
  }

  return [];
}
