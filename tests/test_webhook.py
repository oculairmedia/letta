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
