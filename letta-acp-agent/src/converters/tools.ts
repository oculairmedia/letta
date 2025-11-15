import type * as acp from '@agentclientprotocol/sdk';
import type { ToolMapping, LettaToolCall } from '../types/index.js';

/**
 * Mappings from Letta tool names to ACP tool kinds and permission requirements
 */
export const TOOL_MAPPINGS: ToolMapping[] = [
  { lettaToolName: 'open_file', acpKind: 'read', requiresPermission: false },
  { lettaToolName: 'grep_file', acpKind: 'search', requiresPermission: false },
  { lettaToolName: 'search_file', acpKind: 'search', requiresPermission: false },
  { lettaToolName: 'run_code', acpKind: 'execute', requiresPermission: true },
  { lettaToolName: 'web_search', acpKind: 'fetch', requiresPermission: true },
  { lettaToolName: 'edit_file', acpKind: 'edit', requiresPermission: true },
  { lettaToolName: 'write_file', acpKind: 'edit', requiresPermission: true },
  { lettaToolName: 'delete_file', acpKind: 'delete', requiresPermission: true },
  { lettaToolName: 'move_file', acpKind: 'move', requiresPermission: true },
  { lettaToolName: 'core_memory_append', acpKind: 'think', requiresPermission: false },
  { lettaToolName: 'core_memory_replace', acpKind: 'think', requiresPermission: false },
  { lettaToolName: 'archival_memory_insert', acpKind: 'think', requiresPermission: false },
  { lettaToolName: 'archival_memory_search', acpKind: 'search', requiresPermission: false },
];

/**
 * Map Letta tool name to ACP ToolKind
 */
export function mapLettaToolKindToACP(toolName: string): acp.ToolKind {
  const mapping = TOOL_MAPPINGS.find(m => m.lettaToolName === toolName);
  return mapping?.acpKind || 'other';
}

/**
 * Check if a tool requires user permission before execution
 */
export function shouldRequestPermission(toolName: string): boolean {
  const mapping = TOOL_MAPPINGS.find(m => m.lettaToolName === toolName);
  return mapping?.requiresPermission ?? false;
}

/**
 * Convert Letta tool call to ACP tool call update format
 */
export function convertLettaToolCallToACP(lettaTool: LettaToolCall) {
  const kind = mapLettaToolKindToACP(lettaTool.name);

  return {
    toolCallId: lettaTool.id,
    title: formatToolTitle(lettaTool.name, lettaTool.arguments),
    kind,
    status: mapLettaToolStatusToACP(lettaTool.status),
    rawInput: lettaTool.arguments,
    rawOutput: lettaTool.result,
    locations: extractLocations(lettaTool.name, lettaTool.arguments),
  };
}

/**
 * Map Letta tool status to ACP ToolCallStatus
 */
function mapLettaToolStatusToACP(status: string): acp.ToolCallStatus {
  switch (status) {
    case 'pending':
      return 'pending';
    case 'in_progress':
      return 'in_progress';
    case 'completed':
      return 'completed';
    case 'failed':
      return 'failed';
    default:
      return 'pending';
  }
}

/**
 * Format a human-readable title for the tool call
 */
function formatToolTitle(toolName: string, args: Record<string, unknown>): string {
  switch (toolName) {
    case 'open_file':
    case 'grep_file':
    case 'search_file':
      return `Reading ${args.path || 'file'}`;
    case 'edit_file':
    case 'write_file':
      return `Editing ${args.path || 'file'}`;
    case 'delete_file':
      return `Deleting ${args.path || 'file'}`;
    case 'move_file':
      return `Moving ${args.source || 'file'}`;
    case 'run_code':
      return `Executing ${args.language || 'code'}`;
    case 'web_search':
      return `Searching: ${args.query || 'web'}`;
    case 'core_memory_append':
    case 'core_memory_replace':
      return `Updating memory`;
    case 'archival_memory_insert':
      return `Saving to long-term memory`;
    case 'archival_memory_search':
      return `Searching long-term memory`;
    default:
      return toolName.replace(/_/g, ' ');
  }
}

/**
 * Extract file locations from tool arguments
 */
function extractLocations(
  _toolName: string,
  args: Record<string, unknown>
): acp.ToolCallLocation[] | undefined {
  const locations: acp.ToolCallLocation[] = [];

  // File operation tools
  if (args.path && typeof args.path === 'string') {
    locations.push({ path: args.path });
  }

  if (args.source && typeof args.source === 'string') {
    locations.push({ path: args.source });
  }

  if (args.destination && typeof args.destination === 'string') {
    locations.push({ path: args.destination });
  }

  return locations.length > 0 ? locations : undefined;
}

/**
 * Create permission options for a tool call
 */
export function createPermissionOptions(toolName: string): acp.PermissionOption[] {
  const options: acp.PermissionOption[] = [
    {
      optionId: 'allow',
      name: 'Allow',
      kind: 'allow_once',
    },
    {
      optionId: 'reject',
      name: 'Reject',
      kind: 'reject_once',
    },
  ];

  // Add "always allow" option for certain tools
  const allowAlways = ['web_search', 'run_code'];
  if (allowAlways.includes(toolName)) {
    options.push({
      optionId: 'allow_always',
      name: 'Always allow',
      kind: 'allow_always',
    });
  }

  return options;
}
