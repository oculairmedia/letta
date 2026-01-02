import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from letta.helpers.webhook_validation import (
    DEFAULT_BLOCKED_HOSTS,
    is_private_ip,
    resolve_hostname,
    validate_webhook_url,
    validate_webhook_url_strict,
)
from letta.schemas.webhook import (
    AgentWebhookConfig,
    AgentWebhookConfigUpdate,
    WebhookConfig,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookEvent,
    WebhookEventPayload,
    WebhookEventType,
)
from letta.services.webhook_manager import WebhookManager


class TestIsPrivateIp:
    def test_loopback_ipv4(self):
        assert is_private_ip("127.0.0.1") is True
        assert is_private_ip("127.0.0.255") is True

    def test_loopback_ipv6(self):
        assert is_private_ip("::1") is True

    def test_private_class_a(self):
        assert is_private_ip("10.0.0.1") is True
        assert is_private_ip("10.255.255.255") is True

    def test_private_class_b(self):
        assert is_private_ip("172.16.0.1") is True
        assert is_private_ip("172.31.255.255") is True

    def test_private_class_c(self):
        assert is_private_ip("192.168.0.1") is True
        assert is_private_ip("192.168.255.255") is True

    def test_link_local(self):
        assert is_private_ip("169.254.0.1") is True
        assert is_private_ip("169.254.169.254") is True

    def test_public_ips(self):
        assert is_private_ip("8.8.8.8") is False
        assert is_private_ip("1.1.1.1") is False
        assert is_private_ip("142.250.80.46") is False

    def test_invalid_ip(self):
        assert is_private_ip("not-an-ip") is False
        assert is_private_ip("") is False


class TestValidateWebhookUrl:
    def test_valid_https_url(self):
        is_valid, error = validate_webhook_url("https://example.com/webhook")
        assert is_valid is True
        assert error is None

    def test_valid_http_url(self):
        is_valid, error = validate_webhook_url("http://example.com/webhook")
        assert is_valid is True
        assert error is None

    def test_empty_url(self):
        is_valid, error = validate_webhook_url("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_missing_scheme(self):
        is_valid, error = validate_webhook_url("example.com/webhook")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_invalid_scheme(self):
        is_valid, error = validate_webhook_url("ftp://example.com/webhook")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_localhost_blocked(self):
        is_valid, error = validate_webhook_url("http://localhost/webhook")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_127_0_0_1_blocked(self):
        is_valid, error = validate_webhook_url("http://127.0.0.1/webhook")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_metadata_google_blocked(self):
        is_valid, error = validate_webhook_url("http://metadata.google.internal/webhook")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_aws_metadata_blocked(self):
        is_valid, error = validate_webhook_url("http://169.254.169.254/latest/meta-data/")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_private_ip_blocked(self):
        is_valid, error = validate_webhook_url("http://10.0.0.1/webhook")
        assert is_valid is False
        assert "private" in error.lower()

        is_valid, error = validate_webhook_url("http://192.168.1.1/webhook")
        assert is_valid is False
        assert "private" in error.lower()

    def test_require_https(self):
        is_valid, error = validate_webhook_url("http://example.com/webhook", require_https=True)
        assert is_valid is False
        assert "HTTPS" in error

        is_valid, error = validate_webhook_url("https://example.com/webhook", require_https=True)
        assert is_valid is True
        assert error is None

    def test_allowed_hosts_override(self):
        is_valid, error = validate_webhook_url(
            "http://internal.mycompany.com/webhook",
            allowed_hosts=["internal.mycompany.com"],
        )
        assert is_valid is True
        assert error is None

    def test_custom_blocked_hosts(self):
        is_valid, error = validate_webhook_url(
            "http://blocked.example.com/webhook",
            blocked_hosts=["blocked.example.com"],
        )
        assert is_valid is False
        assert "blocked" in error.lower()


class TestValidateWebhookUrlStrict:
    def test_requires_https(self):
        is_valid, error = validate_webhook_url_strict("http://example.com/webhook")
        assert is_valid is False
        assert "HTTPS" in error

    def test_allows_https(self):
        is_valid, error = validate_webhook_url_strict("https://example.com/webhook")
        assert is_valid is True
        assert error is None


class TestWebhookEventType:
    def test_all_event_types_defined(self):
        expected_events = [
            "agent.step.completed",
            "agent.step.failed",
            "agent.message.sent",
            "agent.state.updated",
            "agent.tool.attached",
            "agent.tool.detached",
            "tool.execution.completed",
            "tool.execution.failed",
            "agent.run.started",
            "agent.run.completed",
            "agent.run.failed",
            "agent.job.completed",
            "agent.job.failed",
            "agent.memory.updated",
            "tool.created",
            "tool.updated",
            "tool.deleted",
        ]
        actual_events = [e.value for e in WebhookEventType]
        for expected in expected_events:
            assert expected in actual_events


class TestWebhookConfig:
    def test_valid_config(self):
        config = WebhookConfig(
            url="https://example.com/webhook",
            secret="my-secret",
            events=[WebhookEventType.AGENT_RUN_COMPLETED],
            enabled=True,
        )
        assert config.url == "https://example.com/webhook"
        assert config.secret == "my-secret"
        assert config.enabled is True

    def test_url_validation_rejects_empty(self):
        with pytest.raises(ValueError):
            WebhookConfig(url="")

    def test_url_validation_rejects_invalid_scheme(self):
        with pytest.raises(ValueError):
            WebhookConfig(url="ftp://example.com")

    def test_is_subscribed_to_empty_events(self):
        config = WebhookConfig(url="https://example.com", events=[])
        assert config.is_subscribed_to(WebhookEventType.AGENT_RUN_COMPLETED) is True
        assert config.is_subscribed_to(WebhookEventType.AGENT_STEP_FAILED) is True

    def test_is_subscribed_to_specific_events(self):
        config = WebhookConfig(
            url="https://example.com",
            events=[WebhookEventType.AGENT_RUN_COMPLETED, WebhookEventType.AGENT_RUN_FAILED],
        )
        assert config.is_subscribed_to(WebhookEventType.AGENT_RUN_COMPLETED) is True
        assert config.is_subscribed_to(WebhookEventType.AGENT_RUN_FAILED) is True
        assert config.is_subscribed_to(WebhookEventType.AGENT_STEP_COMPLETED) is False


class TestWebhookEvent:
    def test_generate_event_id(self):
        event_id = WebhookEvent.generate_event_id()
        assert event_id.startswith("evt-")
        assert len(event_id) > 10

    def test_event_creation(self):
        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-123",
            organization_id="org-456",
            data={"key": "value"},
        )
        assert event.event_type == WebhookEventType.AGENT_RUN_COMPLETED
        assert event.agent_id == "agent-123"
        assert event.data == {"key": "value"}


class TestWebhookDelivery:
    def test_generate_id(self):
        delivery_id = WebhookDelivery.generate_id()
        assert delivery_id.startswith("whd-")
        assert len(delivery_id) > 10

    def test_should_retry_pending(self):
        delivery = WebhookDelivery(
            id="whd-123",
            event_id="evt-456",
            agent_id="agent-789",
            webhook_url="https://example.com",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            status=WebhookDeliveryStatus.PENDING,
            attempt_count=1,
            max_attempts=3,
        )
        assert delivery.should_retry() is True

    def test_should_retry_exhausted(self):
        delivery = WebhookDelivery(
            id="whd-123",
            event_id="evt-456",
            agent_id="agent-789",
            webhook_url="https://example.com",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            status=WebhookDeliveryStatus.FAILED,
            attempt_count=3,
            max_attempts=3,
        )
        assert delivery.should_retry() is False

    def test_should_retry_delivered(self):
        delivery = WebhookDelivery(
            id="whd-123",
            event_id="evt-456",
            agent_id="agent-789",
            webhook_url="https://example.com",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            status=WebhookDeliveryStatus.DELIVERED,
            attempt_count=1,
            max_attempts=3,
        )
        assert delivery.should_retry() is False

    def test_get_next_retry_delay_seconds(self):
        delivery = WebhookDelivery(
            id="whd-123",
            event_id="evt-456",
            agent_id="agent-789",
            webhook_url="https://example.com",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            attempt_count=0,
        )
        delay = delivery.get_next_retry_delay_seconds()
        assert 3 <= delay <= 8  # 5 +/- 25% jitter


class TestWebhookEventPayload:
    def test_create_without_signature(self):
        event = WebhookEvent(
            id="evt-123",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-456",
            organization_id="org-789",
            data={},
        )
        payload = WebhookEventPayload.create(event)
        assert "Content-Type" in payload.headers
        assert payload.headers["Content-Type"] == "application/json"
        assert "X-Letta-Event-Type" in payload.headers
        assert payload.headers["X-Letta-Event-Type"] == "agent.run.completed"
        assert "X-Letta-Signature" not in payload.headers

    def test_create_with_signature(self):
        event = WebhookEvent(
            id="evt-123",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-456",
            organization_id="org-789",
            data={},
        )
        payload = WebhookEventPayload.create(event, signature="t=123,v1=abc")
        assert "X-Letta-Signature" in payload.headers
        assert payload.headers["X-Letta-Signature"] == "t=123,v1=abc"


class TestAgentWebhookConfig:
    def test_response_schema(self):
        config = AgentWebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEventType.AGENT_RUN_COMPLETED],
            enabled=True,
            has_secret=True,
        )
        assert config.url == "https://example.com/webhook"
        assert config.enabled is True
        assert config.has_secret is True


class TestAgentWebhookConfigUpdate:
    def test_update_schema(self):
        update = AgentWebhookConfigUpdate(
            url="https://new-url.com/webhook",
            secret="new-secret",
            events=[WebhookEventType.AGENT_RUN_COMPLETED],
            enabled=True,
        )
        assert update.url == "https://new-url.com/webhook"
        assert update.secret == "new-secret"
        assert update.enabled is True

    def test_url_validation(self):
        with pytest.raises(ValueError):
            AgentWebhookConfigUpdate(url="invalid-url")


class TestWebhookManager:
    @pytest.fixture
    def mock_actor(self):
        actor = MagicMock()
        actor.organization_id = "org-123"
        return actor

    @pytest.fixture
    def webhook_manager(self, mock_actor):
        return WebhookManager(actor=mock_actor, timeout_seconds=5, max_retries=2)

    @pytest.mark.asyncio
    async def test_publish_event_no_config(self, webhook_manager):
        result = await webhook_manager.publish_event(
            agent_id="agent-123",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            payload={"test": "data"},
            webhook_config=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_publish_event_disabled(self, webhook_manager):
        config = WebhookConfig(url="https://example.com", enabled=False)
        result = await webhook_manager.publish_event(
            agent_id="agent-123",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            payload={"test": "data"},
            webhook_config=config,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_publish_event_not_subscribed(self, webhook_manager):
        config = WebhookConfig(
            url="https://example.com",
            enabled=True,
            events=[WebhookEventType.AGENT_STEP_COMPLETED],
        )
        result = await webhook_manager.publish_event(
            agent_id="agent-123",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            payload={"test": "data"},
            webhook_config=config,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_publish_event_invalid_url(self, webhook_manager):
        config = WebhookConfig(url="http://localhost/webhook", enabled=True)
        result = await webhook_manager.publish_event(
            agent_id="agent-123",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            payload={"test": "data"},
            webhook_config=config,
        )
        assert result is None

    def test_generate_signature(self, webhook_manager):
        payload = '{"test": "data"}'
        secret = "my-secret"
        signature = webhook_manager._generate_signature(payload, secret)
        assert signature.startswith("t=")
        assert ",v1=" in signature

    def test_verify_signature_valid(self):
        secret = "test-secret"
        payload = '{"test": "data"}'
        timestamp = str(int(time.time()))
        signed_payload = f"{timestamp}.{payload}"

        import hashlib
        import hmac

        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        signature = f"t={timestamp},v1={expected_sig}"
        assert WebhookManager.verify_signature(payload, signature, secret) is True

    def test_verify_signature_invalid(self):
        assert WebhookManager.verify_signature('{"test": "data"}', "invalid", "secret") is False

    def test_verify_signature_expired(self):
        secret = "test-secret"
        payload = '{"test": "data"}'
        old_timestamp = str(int(time.time()) - 600)
        signed_payload = f"{old_timestamp}.{payload}"

        import hashlib
        import hmac

        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        signature = f"t={old_timestamp},v1={expected_sig}"
        assert WebhookManager.verify_signature(payload, signature, secret, tolerance_seconds=300) is False

    @pytest.mark.asyncio
    async def test_dispatch_webhook_success(self, webhook_manager):
        event = WebhookEvent(
            id="evt-123",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-456",
            organization_id="org-789",
            data={"message": "test"},
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        with patch("letta.services.webhook_manager.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            delivery = await webhook_manager._dispatch_webhook_async(
                webhook_url="https://example.com/webhook",
                event=event,
            )

            assert delivery.status == WebhookDeliveryStatus.DELIVERED
            assert delivery.status_code == 200
            assert delivery.attempt_count == 1

    @pytest.mark.asyncio
    async def test_dispatch_webhook_failure_retries(self, webhook_manager):
        event = WebhookEvent(
            id="evt-123",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-456",
            organization_id="org-789",
            data={},
        )

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("letta.services.webhook_manager.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            delivery = await webhook_manager._dispatch_webhook_async(
                webhook_url="https://example.com/webhook",
                event=event,
            )

            assert delivery.status == WebhookDeliveryStatus.FAILED
            assert delivery.attempt_count == 2  # max_retries=2


class TestWebhookMessagePayload:
    """
    Integration tests for AGENT_RUN_COMPLETED webhook message payload structure (LCORE-11).
    Verifies webhook payload format for downstream consumers like Matrix handlers.
    """

    def test_assistant_message_serialization(self):
        """Verify AssistantMessage serializes with required fields for webhook payload."""
        from letta.schemas.letta_message import AssistantMessage, MessageType
        from datetime import datetime, timezone

        msg = AssistantMessage(
            id="msg-test-123",
            date=datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc),
            message_type=MessageType.assistant_message,
            content="Hello! How can I help you today?",
        )

        serialized = msg.model_dump(mode="json")

        assert "id" in serialized
        assert "date" in serialized
        assert "message_type" in serialized
        assert "content" in serialized
        assert serialized["id"] == "msg-test-123"
        assert serialized["message_type"] == "assistant_message"
        assert serialized["content"] == "Hello! How can I help you today?"

    def test_webhook_event_wraps_payload_in_data_field(self):
        """
        Verify WebhookEvent wraps payload in 'data' field.
        Critical: consumers expect payload.data.messages, not payload.messages.
        """
        run_payload = {
            "run_id": "run-test-456",
            "stop_reason": {"stop_reason": "end_turn"},
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "message_count": 1,
            "messages": [
                {
                    "id": "msg-123",
                    "date": "2025-01-02T14:30:00+00:00",
                    "message_type": "assistant_message",
                    "content": "Test message",
                }
            ],
        }

        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-789",
            organization_id="org-123",
            data=run_payload,
        )

        event_json = event.model_dump(mode="json")

        assert "data" in event_json
        assert "messages" in event_json["data"]
        assert event_json["data"]["message_count"] == 1
        assert len(event_json["data"]["messages"]) == 1

        message = event_json["data"]["messages"][0]
        assert message["message_type"] == "assistant_message"
        assert message["content"] == "Test message"

    def test_webhook_event_payload_http_structure(self):
        """Verify WebhookEventPayload creates correct HTTP headers for delivery."""
        event = WebhookEvent(
            id="evt-test-789",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-123",
            organization_id="org-456",
            data={
                "run_id": "run-001",
                "message_count": 2,
                "messages": [
                    {"message_type": "reasoning_message", "content": "Thinking..."},
                    {"message_type": "assistant_message", "content": "Hello!"},
                ],
            },
        )

        payload = WebhookEventPayload.create(event)

        assert payload.headers["Content-Type"] == "application/json"
        assert payload.headers["X-Letta-Event-Type"] == "agent.run.completed"
        assert payload.headers["X-Letta-Event-Id"] == "evt-test-789"
        assert "X-Letta-Timestamp" in payload.headers
        assert payload.event.data["message_count"] == 2
        assert len(payload.event.data["messages"]) == 2

    def test_multiple_message_types_in_payload(self):
        """Verify webhook payload can contain multiple message types (reasoning, tool calls, assistant)."""
        from letta.schemas.letta_message import (
            AssistantMessage,
            ReasoningMessage,
            ToolCallMessage,
            MessageType,
            ToolCall as LettaToolCall,
        )
        from datetime import datetime, timezone

        now = datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

        messages = [
            ReasoningMessage(
                id="msg-1",
                date=now,
                message_type=MessageType.reasoning_message,
                reasoning="Let me think about this...",
            ),
            ToolCallMessage(
                id="msg-2",
                date=now,
                message_type=MessageType.tool_call_message,
                tool_call=LettaToolCall(
                    name="search",
                    arguments='{"query": "test"}',
                    tool_call_id="call-1",
                ),
            ),
            AssistantMessage(
                id="msg-3",
                date=now,
                message_type=MessageType.assistant_message,
                content="Based on my search, here's what I found.",
            ),
        ]

        serialized_messages = [m.model_dump(mode="json") for m in messages]

        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-multi",
            organization_id="org-test",
            data={
                "run_id": "run-multi",
                "message_count": len(serialized_messages),
                "messages": serialized_messages,
            },
        )

        event_json = event.model_dump(mode="json")

        message_types = [m["message_type"] for m in event_json["data"]["messages"]]
        assert "reasoning_message" in message_types
        assert "tool_call_message" in message_types
        assert "assistant_message" in message_types

        reasoning_msg = next(m for m in event_json["data"]["messages"] if m["message_type"] == "reasoning_message")
        assert "reasoning" in reasoning_msg

        tool_call_msg = next(m for m in event_json["data"]["messages"] if m["message_type"] == "tool_call_message")
        assert "tool_call" in tool_call_msg

        assistant_msg = next(m for m in event_json["data"]["messages"] if m["message_type"] == "assistant_message")
        assert "content" in assistant_msg

    @pytest.mark.asyncio
    async def test_dispatch_webhook_with_message_payload(self):
        """Verify _dispatch_webhook_async sends correctly structured message payload."""
        mock_actor = MagicMock()
        mock_actor.organization_id = "org-integration"

        webhook_manager = WebhookManager(actor=mock_actor, timeout_seconds=5, max_retries=1)

        test_payload = {
            "run_id": "run-integration-test",
            "stop_reason": {"stop_reason": "end_turn"},
            "usage": {"prompt_tokens": 150, "completion_tokens": 75, "total_tokens": 225},
            "message_count": 1,
            "messages": [
                {
                    "id": "msg-integration",
                    "date": "2025-01-02T14:30:00+00:00",
                    "message_type": "assistant_message",
                    "content": "Integration test response",
                }
            ],
        }

        event = WebhookEvent(
            id="evt-integration-test",
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-integration",
            organization_id="org-integration",
            data=test_payload,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        with patch("letta.services.webhook_manager.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            delivery = await webhook_manager._dispatch_webhook_async(
                webhook_url="https://example.com/webhook",
                event=event,
                secret="test-secret",
            )

            assert delivery.status == WebhookDeliveryStatus.DELIVERED
            assert delivery.status_code == 200

            call_args = mock_instance.post.call_args
            sent_json = call_args.kwargs.get("json") or call_args.args[1] if len(call_args.args) > 1 else None

            if sent_json:
                assert sent_json["id"] == "evt-integration-test"
                assert sent_json["event_type"] == "agent.run.completed"
                assert sent_json["agent_id"] == "agent-integration"
                assert "data" in sent_json
                assert sent_json["data"]["message_count"] == 1
                messages = sent_json["data"]["messages"]
                assert len(messages) == 1
                assert messages[0]["message_type"] == "assistant_message"
                assert messages[0]["content"] == "Integration test response"

    def test_empty_messages_payload(self):
        """Verify webhook handles empty messages array (edge case)."""
        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.AGENT_RUN_COMPLETED,
            agent_id="agent-empty",
            organization_id="org-empty",
            data={
                "run_id": "run-empty",
                "stop_reason": {"stop_reason": "max_steps"},
                "usage": None,
                "message_count": 0,
                "messages": [],
            },
        )

        event_json = event.model_dump(mode="json")

        assert event_json["data"]["message_count"] == 0
        assert event_json["data"]["messages"] == []
        assert event_json["data"]["usage"] is None

    def test_assistant_message_with_complex_content(self):
        """Verify AssistantMessage handles list content (multimodal)."""
        from letta.schemas.letta_message import AssistantMessage, MessageType
        from letta.schemas.letta_message_content import TextContent, MessageContentType
        from datetime import datetime, timezone

        content_parts = [
            TextContent(type=MessageContentType.text, text="Here is my response."),
            TextContent(type=MessageContentType.text, text="And here is more."),
        ]

        msg = AssistantMessage(
            id="msg-complex",
            date=datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc),
            message_type=MessageType.assistant_message,
            content=content_parts,
        )

        serialized = msg.model_dump(mode="json")

        assert isinstance(serialized["content"], list)
        assert len(serialized["content"]) == 2
        assert serialized["content"][0]["type"] == "text"
        assert serialized["content"][0]["text"] == "Here is my response."


class TestToolWebhookEvents:
    """Tests for tool.created, tool.updated, tool.deleted webhook events (LCORE-12)."""

    def test_tool_event_types_exist(self):
        """Verify tool webhook event types are defined."""
        assert WebhookEventType.TOOL_CREATED.value == "tool.created"
        assert WebhookEventType.TOOL_UPDATED.value == "tool.updated"
        assert WebhookEventType.TOOL_DELETED.value == "tool.deleted"

    def test_tool_created_event_structure(self):
        """Verify tool.created webhook event has correct structure."""
        tool_payload = {
            "tool_id": "tool-abc123",
            "tool_name": "my_custom_tool",
            "tool_type": "custom",
            "description": "A custom tool for testing",
            "json_schema": {"name": "my_custom_tool", "parameters": {}},
            "tags": ["test", "custom"],
        }

        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.TOOL_CREATED,
            tool_id="tool-abc123",
            organization_id="org-456",
            data=tool_payload,
        )

        assert event.agent_id is None
        assert event.tool_id == "tool-abc123"
        assert event.event_type == WebhookEventType.TOOL_CREATED

        event_json = event.model_dump(mode="json")
        assert event_json["event_type"] == "tool.created"
        assert event_json["tool_id"] == "tool-abc123"
        assert event_json["agent_id"] is None
        assert event_json["data"]["tool_name"] == "my_custom_tool"
        assert event_json["data"]["tool_type"] == "custom"

    def test_tool_updated_event_structure(self):
        """Verify tool.updated webhook event has correct structure."""
        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.TOOL_UPDATED,
            tool_id="tool-xyz789",
            organization_id="org-456",
            data={
                "tool_id": "tool-xyz789",
                "tool_name": "updated_tool",
                "tool_type": "custom",
                "description": "Updated description",
                "json_schema": {"name": "updated_tool"},
                "tags": ["updated"],
            },
        )

        event_json = event.model_dump(mode="json")
        assert event_json["event_type"] == "tool.updated"
        assert event_json["tool_id"] == "tool-xyz789"
        assert event_json["data"]["tool_name"] == "updated_tool"

    def test_tool_deleted_event_structure(self):
        """Verify tool.deleted webhook event has correct structure."""
        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.TOOL_DELETED,
            tool_id="tool-deleted",
            organization_id="org-456",
            data={
                "tool_id": "tool-deleted",
                "tool_name": "deleted_tool",
                "tool_type": "custom",
                "description": None,
                "json_schema": None,
                "tags": [],
            },
        )

        event_json = event.model_dump(mode="json")
        assert event_json["event_type"] == "tool.deleted"
        assert event_json["tool_id"] == "tool-deleted"

    def test_webhook_delivery_with_tool_id(self):
        """Verify WebhookDelivery works with tool_id instead of agent_id."""
        delivery = WebhookDelivery(
            id=WebhookDelivery.generate_id(),
            event_id="evt-tool-123",
            tool_id="tool-abc",
            webhook_url="https://example.com/webhook",
            event_type=WebhookEventType.TOOL_CREATED,
        )

        assert delivery.agent_id is None
        assert delivery.tool_id == "tool-abc"
        assert delivery.event_type == WebhookEventType.TOOL_CREATED

    @pytest.mark.asyncio
    async def test_publish_tool_event_without_global_url(self):
        """Verify publish_tool_event returns None when global_url not configured."""
        mock_actor = MagicMock()
        mock_actor.organization_id = "org-test"

        webhook_manager = WebhookManager(actor=mock_actor)

        with patch("letta.services.webhook_manager.webhook_settings") as mock_settings:
            mock_settings.global_url = None

            result = await webhook_manager.publish_tool_event(
                tool_id="tool-123",
                event_type=WebhookEventType.TOOL_CREATED,
                payload={"tool_name": "test"},
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_publish_tool_event_with_global_url(self):
        """Verify publish_tool_event dispatches when global_url is configured."""
        mock_actor = MagicMock()
        mock_actor.organization_id = "org-test"

        webhook_manager = WebhookManager(actor=mock_actor, persist_deliveries=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        with patch("letta.services.webhook_manager.webhook_settings") as mock_settings:
            mock_settings.global_url = "https://example.com/global-webhook"
            mock_settings.global_secret = "test-secret"

            with patch("letta.services.webhook_manager.AsyncClient") as mock_client:
                mock_instance = AsyncMock()
                mock_instance.__aenter__.return_value = mock_instance
                mock_instance.__aexit__.return_value = None
                mock_instance.post.return_value = mock_response
                mock_client.return_value = mock_instance

                result = await webhook_manager.publish_tool_event(
                    tool_id="tool-456",
                    event_type=WebhookEventType.TOOL_CREATED,
                    payload={
                        "tool_id": "tool-456",
                        "tool_name": "new_tool",
                        "tool_type": "custom",
                    },
                )

                assert result is not None
                assert result.status == WebhookDeliveryStatus.DELIVERED
                assert result.tool_id == "tool-456"

                call_args = mock_instance.post.call_args
                sent_json = call_args.kwargs.get("json")
                assert sent_json["event_type"] == "tool.created"
                assert sent_json["tool_id"] == "tool-456"


class TestAgentToolWebhookEvents:
    """Tests for agent.tool.attached and agent.tool.detached webhook events (LCORE-12)."""

    def test_agent_tool_event_types_exist(self):
        """Verify agent tool webhook event types are defined."""
        assert WebhookEventType.AGENT_TOOL_ATTACHED.value == "agent.tool.attached"
        assert WebhookEventType.AGENT_TOOL_DETACHED.value == "agent.tool.detached"

    def test_agent_tool_attached_event_structure(self):
        """Verify agent.tool.attached webhook event has correct structure."""
        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.AGENT_TOOL_ATTACHED,
            agent_id="agent-123",
            tool_id="tool-456",
            organization_id="org-789",
            data={
                "agent_id": "agent-123",
                "tool_id": "tool-456",
                "tool_name": "search_tool",
            },
        )

        assert event.agent_id == "agent-123"
        assert event.tool_id == "tool-456"

        event_json = event.model_dump(mode="json")
        assert event_json["event_type"] == "agent.tool.attached"
        assert event_json["agent_id"] == "agent-123"
        assert event_json["tool_id"] == "tool-456"
        assert event_json["data"]["tool_name"] == "search_tool"

    def test_agent_tool_detached_event_structure(self):
        """Verify agent.tool.detached webhook event has correct structure."""
        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=WebhookEventType.AGENT_TOOL_DETACHED,
            agent_id="agent-123",
            tool_id="tool-456",
            organization_id="org-789",
            data={
                "agent_id": "agent-123",
                "tool_id": "tool-456",
                "tool_name": "removed_tool",
            },
        )

        event_json = event.model_dump(mode="json")
        assert event_json["event_type"] == "agent.tool.detached"
        assert event_json["data"]["tool_name"] == "removed_tool"

    def test_agent_tool_event_with_both_ids(self):
        """Verify webhook event can have both agent_id and tool_id."""
        event = WebhookEvent(
            id="evt-both-ids",
            event_type=WebhookEventType.AGENT_TOOL_ATTACHED,
            agent_id="agent-abc",
            tool_id="tool-xyz",
            organization_id="org-123",
            data={"agent_id": "agent-abc", "tool_id": "tool-xyz", "tool_name": "dual_tool"},
        )

        assert event.agent_id == "agent-abc"
        assert event.tool_id == "tool-xyz"

        delivery = WebhookDelivery(
            id=WebhookDelivery.generate_id(),
            event_id=event.id,
            agent_id=event.agent_id,
            tool_id=event.tool_id,
            webhook_url="https://example.com/webhook",
            event_type=event.event_type,
        )

        assert delivery.agent_id == "agent-abc"
        assert delivery.tool_id == "tool-xyz"

    def test_webhook_config_subscribes_to_tool_events(self):
        """Verify WebhookConfig can subscribe to tool events."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[
                WebhookEventType.AGENT_TOOL_ATTACHED,
                WebhookEventType.AGENT_TOOL_DETACHED,
                WebhookEventType.TOOL_CREATED,
            ],
            enabled=True,
        )

        assert config.is_subscribed_to(WebhookEventType.AGENT_TOOL_ATTACHED)
        assert config.is_subscribed_to(WebhookEventType.AGENT_TOOL_DETACHED)
        assert config.is_subscribed_to(WebhookEventType.TOOL_CREATED)
        assert not config.is_subscribed_to(WebhookEventType.AGENT_RUN_COMPLETED)


class TestWebhookMessageConsolidation:
    """Tests for consolidating streaming token chunks before webhook delivery (LCORE-14)."""

    @pytest.fixture
    def now(self):
        from datetime import datetime, timezone
        return datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

    def test_consolidate_consecutive_assistant_chunks(self, now):
        from letta.helpers.message_helper import consolidate_streaming_messages
        from letta.schemas.letta_message import AssistantMessage, MessageType

        chunks = [
            AssistantMessage(id="msg-1", date=now, message_type=MessageType.assistant_message, content="Let"),
            AssistantMessage(id="msg-2", date=now, message_type=MessageType.assistant_message, content=" me"),
            AssistantMessage(id="msg-3", date=now, message_type=MessageType.assistant_message, content=" help"),
            AssistantMessage(id="msg-4", date=now, message_type=MessageType.assistant_message, content=" you."),
        ]

        result = consolidate_streaming_messages(chunks)

        assert len(result) == 1
        assert isinstance(result[0], AssistantMessage)
        assert result[0].content == "Let me help you."
        assert result[0].id == "msg-1"

    def test_consolidate_preserves_non_assistant_messages(self, now):
        from letta.helpers.message_helper import consolidate_streaming_messages
        from letta.schemas.letta_message import (
            AssistantMessage,
            MessageType,
            ReasoningMessage,
            ToolCallMessage,
            ToolCall,
            ToolReturnMessage,
        )

        messages = [
            ReasoningMessage(id="msg-1", date=now, message_type=MessageType.reasoning_message, reasoning="Thinking..."),
            AssistantMessage(id="msg-2", date=now, message_type=MessageType.assistant_message, content="Let"),
            AssistantMessage(id="msg-3", date=now, message_type=MessageType.assistant_message, content=" me"),
            ToolCallMessage(
                id="msg-4",
                date=now,
                message_type=MessageType.tool_call_message,
                tool_call=ToolCall(name="search", arguments="{}", tool_call_id="call-1"),
            ),
            ToolReturnMessage(
                id="msg-5",
                date=now,
                message_type=MessageType.tool_return_message,
                tool_return="result",
                status="success",
                tool_call_id="call-1",
                stdout=None,
                stderr=None,
            ),
            AssistantMessage(id="msg-6", date=now, message_type=MessageType.assistant_message, content="Here's"),
            AssistantMessage(id="msg-7", date=now, message_type=MessageType.assistant_message, content=" the result."),
        ]

        result = consolidate_streaming_messages(messages)

        assert len(result) == 5
        assert isinstance(result[0], ReasoningMessage)
        assert result[0].reasoning == "Thinking..."
        assert isinstance(result[1], AssistantMessage)
        assert result[1].content == "Let me"
        assert isinstance(result[2], ToolCallMessage)
        assert isinstance(result[3], ToolReturnMessage)
        assert isinstance(result[4], AssistantMessage)
        assert result[4].content == "Here's the result."

    def test_consolidate_single_message_unchanged(self, now):
        from letta.helpers.message_helper import consolidate_streaming_messages
        from letta.schemas.letta_message import AssistantMessage, MessageType

        messages = [
            AssistantMessage(id="msg-1", date=now, message_type=MessageType.assistant_message, content="Hello!"),
        ]

        result = consolidate_streaming_messages(messages)

        assert len(result) == 1
        assert result[0].content == "Hello!"

    def test_consolidate_empty_list(self):
        from letta.helpers.message_helper import consolidate_streaming_messages

        result = consolidate_streaming_messages([])

        assert result == []

    def test_consolidate_skips_empty_content(self, now):
        from letta.helpers.message_helper import consolidate_streaming_messages
        from letta.schemas.letta_message import AssistantMessage, MessageType

        chunks = [
            AssistantMessage(id="msg-1", date=now, message_type=MessageType.assistant_message, content=""),
            AssistantMessage(id="msg-2", date=now, message_type=MessageType.assistant_message, content="Hello"),
            AssistantMessage(id="msg-3", date=now, message_type=MessageType.assistant_message, content=""),
        ]

        result = consolidate_streaming_messages(chunks)

        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_consolidate_handles_list_content(self, now):
        from letta.helpers.message_helper import consolidate_streaming_messages
        from letta.schemas.letta_message import AssistantMessage, MessageType
        from letta.schemas.letta_message_content import TextContent, MessageContentType

        chunks = [
            AssistantMessage(
                id="msg-1",
                date=now,
                message_type=MessageType.assistant_message,
                content=[TextContent(type=MessageContentType.text, text="Part 1")],
            ),
            AssistantMessage(
                id="msg-2",
                date=now,
                message_type=MessageType.assistant_message,
                content=[TextContent(type=MessageContentType.text, text=" Part 2")],
            ),
        ]

        result = consolidate_streaming_messages(chunks)

        assert len(result) == 1
        assert result[0].content == "Part 1 Part 2"

    def test_consolidate_preserves_metadata_from_first_chunk(self, now):
        from letta.helpers.message_helper import consolidate_streaming_messages
        from letta.schemas.letta_message import AssistantMessage, MessageType

        chunks = [
            AssistantMessage(
                id="msg-first",
                date=now,
                message_type=MessageType.assistant_message,
                content="First",
                step_id="step-123",
                run_id="run-456",
                sender_id="sender-789",
            ),
            AssistantMessage(
                id="msg-second",
                date=now,
                message_type=MessageType.assistant_message,
                content=" Second",
                step_id="step-other",
                run_id="run-other",
            ),
        ]

        result = consolidate_streaming_messages(chunks)

        assert len(result) == 1
        assert result[0].id == "msg-first"
        assert result[0].step_id == "step-123"
        assert result[0].run_id == "run-456"
        assert result[0].sender_id == "sender-789"

    def test_consolidate_only_whitespace_skipped(self, now):
        from letta.helpers.message_helper import consolidate_streaming_messages
        from letta.schemas.letta_message import AssistantMessage, MessageType

        chunks = [
            AssistantMessage(id="msg-1", date=now, message_type=MessageType.assistant_message, content="   "),
            AssistantMessage(id="msg-2", date=now, message_type=MessageType.assistant_message, content="\n"),
        ]

        result = consolidate_streaming_messages(chunks)

        assert result == []
