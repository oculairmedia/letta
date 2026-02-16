from letta.llm_api.openai_client import OpenAIClient
from letta.schemas.enums import AgentType, MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message


def _message_with_ids(agent_id: str, conversation_id: str | None, text: str = "hello") -> Message:
    return Message(
        role=MessageRole.user,
        content=[TextContent(text=text)],
        agent_id=agent_id,
        conversation_id=conversation_id,
    )


def _openai_config(model: str, endpoint_type: str = "openai", provider_name: str | None = "openai") -> LLMConfig:
    return LLMConfig(
        model=model,
        model_endpoint_type=endpoint_type,
        model_endpoint="https://api.openai.com/v1",
        context_window=256000,
        provider_name=provider_name,
    )


def test_responses_request_sets_prompt_cache_fields_for_supported_openai_model():
    client = OpenAIClient()
    llm_config = _openai_config(model="gpt-5.1")
    messages = [_message_with_ids(agent_id="agent-abc", conversation_id="conversation-123")]

    request_data = client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )

    assert "input" in request_data
    assert request_data.get("prompt_cache_key") == "letta:agent-abc:conversation-123"
    assert request_data.get("prompt_cache_retention") == "24h"


def test_responses_request_uses_defaultconv_when_conversation_missing():
    client = OpenAIClient()
    llm_config = _openai_config(model="gpt-5.1")
    messages = [_message_with_ids(agent_id="agent-abc", conversation_id=None)]

    request_data = client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )

    assert request_data.get("prompt_cache_key") == "letta:agent-abc:defaultconv"
    assert request_data.get("prompt_cache_retention") == "24h"


def test_responses_request_omits_24h_for_unsupported_extended_retention_model():
    client = OpenAIClient()
    llm_config = _openai_config(model="o3-mini")
    messages = [_message_with_ids(agent_id="agent-abc", conversation_id="conversation-123")]

    request_data = client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )

    assert request_data.get("prompt_cache_key") == "letta:agent-abc:conversation-123"
    assert "prompt_cache_retention" not in request_data


def test_chat_completions_request_sets_prompt_cache_fields_for_supported_openai_model():
    client = OpenAIClient()
    llm_config = _openai_config(model="gpt-4.1")
    messages = [_message_with_ids(agent_id="agent-abc", conversation_id="conversation-123")]

    request_data = client.build_request_data(
        agent_type=AgentType.memgpt_v2_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )

    assert "messages" in request_data
    assert request_data.get("prompt_cache_key") == "letta:agent-abc:conversation-123"
    assert request_data.get("prompt_cache_retention") == "24h"


def test_chat_completions_request_omits_24h_for_unsupported_extended_retention_model():
    client = OpenAIClient()
    llm_config = _openai_config(model="gpt-4o-mini")
    messages = [_message_with_ids(agent_id="agent-abc", conversation_id="conversation-123")]

    request_data = client.build_request_data(
        agent_type=AgentType.memgpt_v2_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )

    assert request_data.get("prompt_cache_key") == "letta:agent-abc:conversation-123"
    assert "prompt_cache_retention" not in request_data


def test_openrouter_request_omits_prompt_cache_fields_on_both_paths():
    client = OpenAIClient()
    llm_config = LLMConfig(
        model="gpt-5.1",
        handle="openrouter/gpt-5.1",
        model_endpoint_type="openai",
        model_endpoint="https://openrouter.ai/api/v1",
        context_window=256000,
        provider_name="openrouter",
    )
    messages = [_message_with_ids(agent_id="agent-abc", conversation_id="conversation-123")]

    responses_request_data = client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )
    chat_request_data = client.build_request_data(
        agent_type=AgentType.memgpt_v2_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )

    assert "prompt_cache_key" not in responses_request_data
    assert "prompt_cache_retention" not in responses_request_data
    assert "prompt_cache_key" not in chat_request_data
    assert "prompt_cache_retention" not in chat_request_data
