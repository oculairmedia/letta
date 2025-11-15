import { describe, it, expect } from 'vitest';
import {
  TOOL_MAPPINGS,
  mapLettaToolKindToACP,
  shouldRequestPermission,
  convertLettaToolCallToACP,
  createPermissionOptions,
} from '../../src/converters/tools.js';
import type { LettaToolCall } from '../../src/types/index.js';

describe('Tool Converters', () => {
  describe('TOOL_MAPPINGS', () => {
    it('should have mappings for common tools', () => {
      expect(TOOL_MAPPINGS).toContainEqual(
        expect.objectContaining({ lettaToolName: 'open_file' })
      );
      expect(TOOL_MAPPINGS).toContainEqual(
        expect.objectContaining({ lettaToolName: 'edit_file' })
      );
      expect(TOOL_MAPPINGS).toContainEqual(
        expect.objectContaining({ lettaToolName: 'run_code' })
      );
    });

    it('should have valid ACP kinds', () => {
      const validKinds = [
        'read',
        'edit',
        'delete',
        'move',
        'search',
        'execute',
        'think',
        'fetch',
        'other',
      ];

      for (const mapping of TOOL_MAPPINGS) {
        expect(validKinds).toContain(mapping.acpKind);
      }
    });
  });

  describe('mapLettaToolKindToACP', () => {
    it('should map file read tools to "read"', () => {
      expect(mapLettaToolKindToACP('open_file')).toBe('read');
      expect(mapLettaToolKindToACP('grep_file')).toBe('search');
    });

    it('should map file edit tools to "edit"', () => {
      expect(mapLettaToolKindToACP('edit_file')).toBe('edit');
      expect(mapLettaToolKindToACP('write_file')).toBe('edit');
    });

    it('should map file management tools correctly', () => {
      expect(mapLettaToolKindToACP('delete_file')).toBe('delete');
      expect(mapLettaToolKindToACP('move_file')).toBe('move');
    });

    it('should map execution tools to "execute"', () => {
      expect(mapLettaToolKindToACP('run_code')).toBe('execute');
    });

    it('should map search tools to "search"', () => {
      expect(mapLettaToolKindToACP('search_file')).toBe('search');
      expect(mapLettaToolKindToACP('grep_file')).toBe('search');
      expect(mapLettaToolKindToACP('archival_memory_search')).toBe('search');
    });

    it('should map web_search to "fetch"', () => {
      expect(mapLettaToolKindToACP('web_search')).toBe('fetch');
    });

    it('should map memory tools to "think"', () => {
      expect(mapLettaToolKindToACP('core_memory_append')).toBe('think');
      expect(mapLettaToolKindToACP('core_memory_replace')).toBe('think');
    });

    it('should default to "other" for unknown tools', () => {
      expect(mapLettaToolKindToACP('unknown_tool')).toBe('other');
      expect(mapLettaToolKindToACP('custom_function')).toBe('other');
    });
  });

  describe('shouldRequestPermission', () => {
    it('should require permission for destructive operations', () => {
      expect(shouldRequestPermission('edit_file')).toBe(true);
      expect(shouldRequestPermission('write_file')).toBe(true);
      expect(shouldRequestPermission('delete_file')).toBe(true);
      expect(shouldRequestPermission('move_file')).toBe(true);
    });

    it('should require permission for code execution', () => {
      expect(shouldRequestPermission('run_code')).toBe(true);
    });

    it('should require permission for web searches', () => {
      expect(shouldRequestPermission('web_search')).toBe(true);
    });

    it('should not require permission for read operations', () => {
      expect(shouldRequestPermission('open_file')).toBe(false);
      expect(shouldRequestPermission('grep_file')).toBe(false);
      expect(shouldRequestPermission('search_file')).toBe(false);
    });

    it('should not require permission for memory operations', () => {
      expect(shouldRequestPermission('core_memory_append')).toBe(false);
      expect(shouldRequestPermission('core_memory_replace')).toBe(false);
      expect(shouldRequestPermission('archival_memory_search')).toBe(false);
    });

    it('should default to false for unknown tools', () => {
      expect(shouldRequestPermission('unknown_tool')).toBe(false);
    });
  });

  describe('convertLettaToolCallToACP', () => {
    it('should convert basic tool call', () => {
      const lettaTool: LettaToolCall = {
        id: 'tool-123',
        name: 'open_file',
        arguments: { path: '/test/file.txt' },
        status: 'pending',
      };

      const result = convertLettaToolCallToACP(lettaTool);

      expect(result.toolCallId).toBe('tool-123');
      expect(result.kind).toBe('read');
      expect(result.status).toBe('pending');
      expect(result.rawInput).toEqual({ path: '/test/file.txt' });
    });

    it('should include locations for file operations', () => {
      const lettaTool: LettaToolCall = {
        id: 'tool-456',
        name: 'edit_file',
        arguments: { path: '/code/main.js' },
        status: 'in_progress',
      };

      const result = convertLettaToolCallToACP(lettaTool);

      expect(result.locations).toBeDefined();
      expect(result.locations).toHaveLength(1);
      expect(result.locations?.[0].path).toBe('/code/main.js');
    });

    it('should include result when completed', () => {
      const lettaTool: LettaToolCall = {
        id: 'tool-789',
        name: 'run_code',
        arguments: { language: 'python', code: 'print("hello")' },
        status: 'completed',
        result: { stdout: 'hello\n', exitCode: 0 },
      };

      const result = convertLettaToolCallToACP(lettaTool);

      expect(result.status).toBe('completed');
      expect(result.rawOutput).toEqual({ stdout: 'hello\n', exitCode: 0 });
    });

    it('should format title for file operations', () => {
      const lettaTool: LettaToolCall = {
        id: 'tool-1',
        name: 'open_file',
        arguments: { path: '/test.txt' },
        status: 'pending',
      };

      const result = convertLettaToolCallToACP(lettaTool);
      expect(result.title).toContain('/test.txt');
    });

    it('should format title for web search', () => {
      const lettaTool: LettaToolCall = {
        id: 'tool-2',
        name: 'web_search',
        arguments: { query: 'TypeScript testing' },
        status: 'pending',
      };

      const result = convertLettaToolCallToACP(lettaTool);
      expect(result.title).toContain('TypeScript testing');
    });

    it('should format title for code execution', () => {
      const lettaTool: LettaToolCall = {
        id: 'tool-3',
        name: 'run_code',
        arguments: { language: 'python' },
        status: 'pending',
      };

      const result = convertLettaToolCallToACP(lettaTool);
      expect(result.title).toContain('python');
    });

    it('should handle move operations with source and destination', () => {
      const lettaTool: LettaToolCall = {
        id: 'tool-4',
        name: 'move_file',
        arguments: {
          source: '/old/path.txt',
          destination: '/new/path.txt',
        },
        status: 'pending',
      };

      const result = convertLettaToolCallToACP(lettaTool);

      expect(result.locations).toBeDefined();
      expect(result.locations?.length).toBeGreaterThan(0);
    });
  });

  describe('createPermissionOptions', () => {
    it('should create basic permission options', () => {
      const options = createPermissionOptions('edit_file');

      expect(options).toHaveLength(2);
      expect(options).toContainEqual(
        expect.objectContaining({
          optionId: 'allow',
          kind: 'allow_once',
        })
      );
      expect(options).toContainEqual(
        expect.objectContaining({
          optionId: 'reject',
          kind: 'reject_once',
        })
      );
    });

    it('should include "always allow" for certain tools', () => {
      const options = createPermissionOptions('run_code');

      expect(options.length).toBeGreaterThan(2);
      expect(options).toContainEqual(
        expect.objectContaining({
          optionId: 'allow_always',
          kind: 'allow_always',
        })
      );
    });

    it('should include "always allow" for web_search', () => {
      const options = createPermissionOptions('web_search');

      expect(options).toContainEqual(
        expect.objectContaining({
          optionId: 'allow_always',
          kind: 'allow_always',
        })
      );
    });

    it('should not include "always allow" for file edits', () => {
      const options = createPermissionOptions('edit_file');

      expect(options).not.toContainEqual(
        expect.objectContaining({
          kind: 'allow_always',
        })
      );
    });

    it('should have descriptive option names', () => {
      const options = createPermissionOptions('delete_file');

      for (const option of options) {
        expect(option.name).toBeTruthy();
        expect(option.name.length).toBeGreaterThan(0);
      }
    });
  });
});
