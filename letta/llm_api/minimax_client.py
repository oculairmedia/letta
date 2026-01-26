import os
from typing import List, Optional, Union

import anthropic
from anthropic import AsyncStream
from anthropic.types import Message as AnthropicMessage, RawMessageStreamEvent

from letta.llm_api.anthropic_client import AnthropicClient
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.enums import AgentType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as PydanticMessage
from letta.settings import model_settings

logger = get_logger(__name__)

# MiniMax Anthropic-compatible API base URL
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"


class MiniMaxClient(AnthropicClient):
    """
    MiniMax LLM client using Anthropic-compatible API.

    Key differences from AnthropicClient:
    - Uses standard messages API (client.messages.create), NOT beta API
    - Thinking blocks are natively supported without beta headers
    - Temperature must be in range (0.0, 1.0]
    - Some Anthropic params are ignored: top_k, stop_sequences, service_tier, etc.

    Documentation: https://platform.minimax.io/docs/api-reference/text-anthropic-api
    """

    @trace_method
    def _get_anthropic_client(
        self, llm_config: LLMConfig, async_client: bool = False
    ) -> Union[anthropic.AsyncAnthropic, anthropic.Anthropic]:
        """Create Anthropic client configured for MiniMax API."""
        api_key, _, _ = self.get_byok_overrides(llm_config)

        if not api_key:
            api_key = model_settings.minimax_api_key or os.environ.get("MINIMAX_API_KEY")

        if async_client:
            if api_key:
                return anthropic.AsyncAnthropic(api_key=api_key, base_url=MINIMAX_BASE_URL)
            return anthropic.AsyncAnthropic(base_url=MINIMAX_BASE_URL)

        if api_key:
            return anthropic.Anthropic(api_key=api_key, base_url=MINIMAX_BASE_URL)
        return anthropic.Anthropic(base_url=MINIMAX_BASE_URL)

    @trace_method
    async def _get_anthropic_client_async(
        self, llm_config: LLMConfig, async_client: bool = False
    ) -> Union[anthropic.AsyncAnthropic, anthropic.Anthropic]:
        """Create Anthropic client configured for MiniMax API (async version)."""
        api_key, _, _ = await self.get_byok_overrides_async(llm_config)

        if not api_key:
            api_key = model_settings.minimax_api_key or os.environ.get("MINIMAX_API_KEY")

        if async_client:
            if api_key:
                return anthropic.AsyncAnthropic(api_key=api_key, base_url=MINIMAX_BASE_URL)
            return anthropic.AsyncAnthropic(base_url=MINIMAX_BASE_URL)

        if api_key:
            return anthropic.Anthropic(api_key=api_key, base_url=MINIMAX_BASE_URL)
        return anthropic.Anthropic(base_url=MINIMAX_BASE_URL)

    @trace_method
    def request(self, request_data: dict, llm_config: LLMConfig) -> dict:
        """
        Synchronous request to MiniMax API.

        Uses standard messages API (NOT beta) - MiniMax natively supports thinking blocks.
        """
        client = self._get_anthropic_client(llm_config, async_client=False)

        # MiniMax uses client.messages.create() - NOT client.beta.messages.create()
        # Thinking blocks are natively supported without beta headers
        response: AnthropicMessage = client.messages.create(**request_data)
        return response.model_dump()

    @trace_method
    async def request_async(self, request_data: dict, llm_config: LLMConfig) -> dict:
        """
        Asynchronous request to MiniMax API.

        Uses standard messages API (NOT beta) - MiniMax natively supports thinking blocks.
        """
        client = await self._get_anthropic_client_async(llm_config, async_client=True)

        # MiniMax uses client.messages.create() - NOT client.beta.messages.create()
        # Thinking blocks are natively supported without beta headers
        try:
            response: AnthropicMessage = await client.messages.create(**request_data)
            return response.model_dump()
        except ValueError as e:
            # Handle streaming fallback if needed (similar to Anthropic client)
            if "streaming is required" in str(e).lower():
                logger.warning(
                    "[MiniMax] Non-streaming request rejected. Falling back to streaming mode. Error: %s",
                    str(e),
                )
                return await self._request_via_streaming(request_data, llm_config, betas=[])
            raise

    @trace_method
    async def stream_async(self, request_data: dict, llm_config: LLMConfig) -> AsyncStream[RawMessageStreamEvent]:
        """
        Asynchronous streaming request to MiniMax API.

        Uses standard messages API (NOT beta) - MiniMax natively supports thinking blocks.
        """
        client = await self._get_anthropic_client_async(llm_config, async_client=True)
        request_data["stream"] = True

        # MiniMax uses client.messages.create() - NOT client.beta.messages.create()
        # No beta headers needed - thinking blocks are natively supported
        try:
            return await client.messages.create(**request_data)
        except Exception as e:
            logger.error(f"Error streaming MiniMax request: {e}")
            raise e

    @trace_method
    def build_request_data(
        self,
        agent_type: AgentType,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
        requires_subsequent_tool_call: bool = False,
        tool_return_truncation_chars: Optional[int] = None,
    ) -> dict:
        """
        Build request data for MiniMax API.

        Inherits most logic from AnthropicClient, with MiniMax-specific adjustments:
        - Temperature must be in range (0.0, 1.0]
        - Removes extended thinking params (natively supported)
        """
        data = super().build_request_data(
            agent_type,
            messages,
            llm_config,
            tools,
            force_tool_call,
            requires_subsequent_tool_call,
            tool_return_truncation_chars,
        )

        # MiniMax temperature range is (0.0, 1.0], recommended value: 1
        if data.get("temperature") is not None:
            temp = data["temperature"]
            if temp <= 0:
                data["temperature"] = 0.01  # Minimum valid value (exclusive of 0)
                logger.warning(f"[MiniMax] Temperature {temp} is invalid. Clamped to 0.01.")
            elif temp > 1.0:
                data["temperature"] = 1.0  # Maximum valid value
                logger.warning(f"[MiniMax] Temperature {temp} is invalid. Clamped to 1.0.")

        # MiniMax ignores these Anthropic-specific parameters, but we can remove them
        # to avoid potential issues (they won't cause errors, just ignored)
        # Note: We don't remove them since MiniMax silently ignores them

        return data

    def is_reasoning_model(self, llm_config: LLMConfig) -> bool:
        """
        All MiniMax M2.x models support native interleaved thinking.

        Unlike Anthropic where only certain models (Claude 3.7+) support extended thinking,
        all MiniMax models natively support thinking blocks without beta headers.
        """
        return True

    def requires_auto_tool_choice(self, llm_config: LLMConfig) -> bool:
        """MiniMax models support all tool choice modes."""
        return False

    def supports_structured_output(self, llm_config: LLMConfig) -> bool:
        """MiniMax doesn't currently advertise structured output support."""
        return False
