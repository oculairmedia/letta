#!/usr/bin/env node

import * as acp from '@agentclientprotocol/sdk';
import { Readable, Writable } from 'node:stream';
import { LettaACPAgent } from './agent.js';
import type { LettaConfig } from './types/index.js';

/**
 * Get configuration from environment variables
 */
function getConfig(): LettaConfig {
  const apiUrl = process.env.LETTA_API_URL || 'http://localhost:8283';
  const apiKey = process.env.LETTA_API_KEY;
  const defaultModel = process.env.LETTA_DEFAULT_MODEL || 'openai/gpt-4.1';

  return {
    apiUrl,
    apiKey,
    defaultModel,
  };
}

/**
 * Main entry point
 */
function main() {
  console.error('========================================');
  console.error('Letta ACP Agent Starting...');
  console.error('========================================');

  const config = getConfig();

  console.error('Configuration:');
  console.error(`  API URL: ${config.apiUrl}`);
  console.error(`  API Key: ${config.apiKey ? '***' : '(not set)'}`);
  console.error(`  Default Model: ${config.defaultModel}`);
  console.error('========================================');

  try {
    // Create stdio streams for ACP communication
    // Note: ACP uses stdin for reading and stdout for writing
    const input = Writable.toWeb(process.stdout);
    const output = Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>;

    console.error('[Main] Creating ACP stream...');

    // Create ACP stream
    const stream = acp.ndJsonStream(input, output);

    console.error('[Main] Creating agent connection...');

    // Create and start agent
    const connection = new acp.AgentSideConnection(
      (conn) => new LettaACPAgent(conn, config),
      stream
    );

    console.error('[Main] Agent connection established');
    console.error('[Main] Ready to receive requests');
    console.error('========================================');

    // Listen for connection close
    connection.closed.then(() => {
      console.error('========================================');
      console.error('[Main] Connection closed');
      console.error('========================================');
      process.exit(0);
    });

    // Handle process signals
    process.on('SIGINT', () => {
      console.error('[Main] Received SIGINT, shutting down...');
      process.exit(0);
    });

    process.on('SIGTERM', () => {
      console.error('[Main] Received SIGTERM, shutting down...');
      process.exit(0);
    });
  } catch (error) {
    console.error('========================================');
    console.error('[Main] Fatal error:', error);
    console.error('========================================');
    process.exit(1);
  }
}

// Run the agent
main();
