"""Standalone compaction functions for message summarization."""

from dataclasses import dataclass
from typing import List, Optional

from letta.helpers.message_helper import convert_message_creates_to_messages
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, MessageCreate
from letta.schemas.tool import Tool
from letta.schemas.user import User
from letta.services.summarizer.summarizer_all import summarize_all
from letta.services.summarizer.summarizer_config import CompactionSettings
from letta.services.summarizer.summarizer_sliding_window import (
    count_tokens,
    count_tokens_with_tools,
    summarize_via_sliding_window,
)
from letta.system import package_summarize_message_no_counts

logger = get_logger(__name__)


@dataclass
class CompactResult:
    """Result of a compaction operation."""

    summary_message: Message
    compacted_messages: list[Message]
    summary_text: str
    context_token_estimate: Optional[int]


async def build_summarizer_llm_config(
    agent_llm_config: LLMConfig,
    summarizer_config: CompactionSettings,
    actor: User,
) -> LLMConfig:
    """Derive an LLMConfig for summarization from a model handle.

    This mirrors the agent-creation path: start from the agent's LLMConfig,
    override provider/model/handle from ``compaction_settings.model``, and
    then apply any explicit ``compaction_settings.model_settings`` via
    ``_to_legacy_config_params``.

    Args:
        agent_llm_config: The agent's LLM configuration to use as base.
        summarizer_config: Compaction settings with optional model override.
        actor: The user performing the operation.

    Returns:
        LLMConfig configured for summarization.
    """
    # If no summarizer model handle is provided, fall back to the agent's config
    if not summarizer_config.model:
        return agent_llm_config

    try:
        # Parse provider/model from the handle, falling back to the agent's
        # provider type when only a model name is given.
        if "/" in summarizer_config.model:
            provider_name, model_name = summarizer_config.model.split("/", 1)
        else:
            provider_name = agent_llm_config.provider_name
            model_name = summarizer_config.model

        # Start from the agent's config and override model + provider_name + handle
        # Check if the summarizer's provider matches the agent's provider
        # If they match, we can safely use the agent's config as a base
        # If they don't match, we need to load the default config for the new provider
        from letta.schemas.enums import ProviderType

        provider_matches = False
        try:
            # Check if provider_name is a valid ProviderType that matches agent's endpoint type
            provider_type = ProviderType(provider_name)
            provider_matches = provider_type.value == agent_llm_config.model_endpoint_type
        except ValueError:
            # provider_name is a custom label - check if it matches agent's provider_name
            provider_matches = provider_name == agent_llm_config.provider_name

        if provider_matches:
            # Same provider - use agent's config as base and override model/handle
            base = agent_llm_config.model_copy()
            base.model = model_name
            base.handle = summarizer_config.model
        else:
            # Different provider - load default config for this handle
            from letta.services.provider_manager import ProviderManager

            provider_manager = ProviderManager()
            try:
                base = await provider_manager.get_llm_config_from_handle(
                    handle=summarizer_config.model,
                    actor=actor,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load LLM config for summarizer handle '{summarizer_config.model}': {e}. Falling back to agent's LLM config."
                )
                return agent_llm_config

        # If explicit model_settings are provided for the summarizer, apply
        # them just like server.create_agent_async does for agents.
        if summarizer_config.model_settings is not None:
            update_params = summarizer_config.model_settings._to_legacy_config_params()
            return base.model_copy(update=update_params)

        return base
    except Exception:
        # On any error, do not break the agent – just fall back
        return agent_llm_config


@trace_method
async def compact_messages(
    actor: User,
    agent_id: str,
    agent_llm_config: LLMConfig,
    messages: List[Message],
    timezone: str,
    compaction_settings: Optional[CompactionSettings] = None,
    agent_model_handle: Optional[str] = None,
    agent_tags: Optional[List[str]] = None,
    tools: Optional[List[Tool]] = None,
    trigger_threshold: Optional[int] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    use_summary_role: bool = True,
    trigger: Optional[str] = None,
    context_tokens_before: Optional[int] = None,
    messages_count_before: Optional[int] = None,
) -> CompactResult:
    """Compact in-context messages using summarization.

    Args:
        actor: The user performing the operation.
        agent_id: The agent's ID.
        agent_llm_config: The agent's LLM configuration.
        messages: The in-context messages to compact.
        timezone: The agent's timezone for message formatting.
        compaction_settings: Optional compaction settings override.
        agent_model_handle: The agent's model handle (used if compaction_settings is None).
        agent_tags: The agent's tags for telemetry.
        tools: The agent's tools (for token counting).
        trigger_threshold: If provided, verify context stays below this after compaction.
        run_id: Optional run ID for telemetry.
        step_id: Optional step ID for telemetry.
        use_summary_role: If True, create summary message with role=summary.
        trigger: What triggered the compaction (for stats).
        context_tokens_before: Token count before compaction (for stats).
        messages_count_before: Message count before compaction (for stats).

    Returns:
        CompactResult containing the summary message, compacted messages, summary text,
        and updated context token estimate.
    """
    # Determine compaction settings
    if compaction_settings is not None:
        summarizer_config = compaction_settings
    elif agent_model_handle is not None:
        summarizer_config = CompactionSettings(model=agent_model_handle)
    else:
        # Fall back to deriving from llm_config
        handle = agent_llm_config.handle or f"{agent_llm_config.model_endpoint_type}/{agent_llm_config.model}"
        summarizer_config = CompactionSettings(model=handle)

    # Build the LLMConfig used for summarization
    summarizer_llm_config = await build_summarizer_llm_config(
        agent_llm_config=agent_llm_config,
        summarizer_config=summarizer_config,
        actor=actor,
    )

    summarization_mode_used = summarizer_config.mode
    if summarizer_config.mode == "all":
        summary, compacted_messages = await summarize_all(
            actor=actor,
            llm_config=summarizer_llm_config,
            summarizer_config=summarizer_config,
            in_context_messages=messages,
            agent_id=agent_id,
            agent_tags=agent_tags,
            run_id=run_id,
            step_id=step_id,
        )
    elif summarizer_config.mode == "sliding_window":
        try:
            summary, compacted_messages = await summarize_via_sliding_window(
                actor=actor,
                llm_config=summarizer_llm_config,
                summarizer_config=summarizer_config,
                in_context_messages=messages,
                agent_id=agent_id,
                agent_tags=agent_tags,
                run_id=run_id,
                step_id=step_id,
            )
        except Exception as e:
            logger.error(f"Sliding window summarization failed with exception: {str(e)}. Falling back to all mode.")
            summary, compacted_messages = await summarize_all(
                actor=actor,
                llm_config=summarizer_llm_config,
                summarizer_config=summarizer_config,
                in_context_messages=messages,
                agent_id=agent_id,
                agent_tags=agent_tags,
                run_id=run_id,
                step_id=step_id,
            )
            summarization_mode_used = "all"
    else:
        raise ValueError(f"Invalid summarizer mode: {summarizer_config.mode}")

    # Update the token count (including tools for accurate comparison with LLM's prompt_tokens)
    context_token_estimate = await count_tokens_with_tools(
        actor=actor,
        llm_config=agent_llm_config,
        messages=compacted_messages,
        tools=tools or [],
    )
    logger.info(f"Context token estimate after summarization: {context_token_estimate}")

    # If the trigger_threshold is provided, verify the new token count is below it
    if trigger_threshold is not None and context_token_estimate is not None and context_token_estimate >= trigger_threshold:
        logger.error(
            "Summarization failed to sufficiently reduce context size: "
            f"post-summarization tokens={context_token_estimate}, "
            f"threshold={trigger_threshold}. "
            "Attempting fallback strategies.",
        )

        # If we used the sliding window mode, try to summarize again with the all mode
        if summarization_mode_used == "sliding_window":
            summary, compacted_messages = await summarize_all(
                actor=actor,
                llm_config=agent_llm_config,
                summarizer_config=summarizer_config,
                in_context_messages=compacted_messages,
                agent_id=agent_id,
                agent_tags=agent_tags,
                run_id=run_id,
                step_id=step_id,
            )
            summarization_mode_used = "all"

        context_token_estimate = await count_tokens_with_tools(
            actor=actor,
            llm_config=agent_llm_config,
            messages=compacted_messages,
            tools=tools or [],
        )

        # Final edge case: check if we're still over threshold
        if context_token_estimate is not None and context_token_estimate >= trigger_threshold:
            # Check if system prompt is the cause
            system_prompt_token_estimate = await count_tokens(
                actor=actor,
                llm_config=agent_llm_config,
                messages=[compacted_messages[0]],
            )
            if system_prompt_token_estimate is not None and system_prompt_token_estimate >= agent_llm_config.context_window:
                from letta.errors import SystemPromptTokenExceededError

                raise SystemPromptTokenExceededError(
                    system_prompt_token_estimate=system_prompt_token_estimate,
                    context_window=agent_llm_config.context_window,
                )

            # Log error but don't brick the agent
            logger.error(f"Failed to summarize messages after fallback: {context_token_estimate} > {trigger_threshold}")
        else:
            logger.info(f"Summarization fallback succeeded: {context_token_estimate} < {trigger_threshold}")

    # Build compaction stats if we have the before values
    compaction_stats = None
    if trigger and context_tokens_before is not None and messages_count_before is not None:
        compaction_stats = {
            "trigger": trigger,
            "context_tokens_before": context_tokens_before,
            "context_tokens_after": context_token_estimate,
            "context_window": agent_llm_config.context_window,
            "messages_count_before": messages_count_before,
            "messages_count_after": len(compacted_messages) + 1,
        }

    # Create the summary message
    summary_message_str_packed = package_summarize_message_no_counts(
        summary=summary,
        timezone=timezone,
        compaction_stats=compaction_stats,
    )

    if use_summary_role:
        # New behavior: Create Message directly with role=summary
        summary_message_obj = Message(
            role=MessageRole.summary,
            content=[TextContent(text=summary_message_str_packed)],
            agent_id=agent_id,
            run_id=run_id,
            step_id=step_id,
        )
    else:
        # Legacy behavior: Use convert_message_creates_to_messages with role=user
        summary_messages = await convert_message_creates_to_messages(
            message_creates=[
                MessageCreate(
                    role=MessageRole.user,
                    content=[TextContent(text=summary_message_str_packed)],
                )
            ],
            agent_id=agent_id,
            timezone=timezone,
            wrap_user_message=False,
            wrap_system_message=False,
            run_id=run_id,
        )
        if len(summary_messages) != 1:
            logger.error(f"Expected only one summary message, got {len(summary_messages)}")
        summary_message_obj = summary_messages[0]

    # Build final messages: [system] + [summary] + remaining compacted messages
    final_messages = [compacted_messages[0], summary_message_obj]
    if len(compacted_messages) > 1:
        final_messages += compacted_messages[1:]

    return CompactResult(
        summary_message=summary_message_obj,
        compacted_messages=final_messages,
        summary_text=summary,
        context_token_estimate=context_token_estimate,
    )
