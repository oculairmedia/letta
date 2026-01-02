import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional

from httpx import AsyncClient, TimeoutException

from letta.helpers.datetime_helpers import get_utc_time
from letta.helpers.webhook_validation import validate_webhook_url
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.user import User as PydanticUser
from letta.schemas.webhook import (
    WebhookConfig,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookEvent,
    WebhookEventPayload,
    WebhookEventType,
)

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3


class WebhookManager:
    def __init__(
        self,
        actor: PydanticUser,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        blocked_hosts: Optional[List[str]] = None,
        allowed_hosts: Optional[List[str]] = None,
    ):
        self.actor = actor
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.blocked_hosts = blocked_hosts
        self.allowed_hosts = allowed_hosts

    @trace_method
    async def publish_event(
        self,
        agent_id: str,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        webhook_config: Optional[WebhookConfig] = None,
    ) -> Optional[WebhookDelivery]:
        if not webhook_config:
            logger.debug(f"No webhook config for agent {agent_id}, skipping event {event_type.value}")
            return None

        if not webhook_config.enabled:
            logger.debug(f"Webhook disabled for agent {agent_id}, skipping event {event_type.value}")
            return None

        if not webhook_config.is_subscribed_to(event_type):
            logger.debug(f"Agent {agent_id} not subscribed to {event_type.value}")
            return None

        is_valid, error = validate_webhook_url(
            webhook_config.url,
            blocked_hosts=self.blocked_hosts,
            allowed_hosts=self.allowed_hosts,
        )
        if not is_valid:
            logger.warning(f"Invalid webhook URL for agent {agent_id}: {error}")
            return None

        organization_id = self.actor.organization_id
        if not organization_id:
            logger.error(f"Actor has no organization_id, cannot publish webhook for agent {agent_id}")
            return None

        event = WebhookEvent(
            id=WebhookEvent.generate_event_id(),
            event_type=event_type,
            agent_id=agent_id,
            organization_id=organization_id,
            data=payload,
        )

        return await self._dispatch_webhook_async(
            webhook_url=webhook_config.url,
            event=event,
            secret=webhook_config.secret,
        )

    @trace_method
    async def _dispatch_webhook_async(
        self,
        webhook_url: str,
        event: WebhookEvent,
        secret: Optional[str] = None,
    ) -> WebhookDelivery:
        delivery = WebhookDelivery(
            id=WebhookDelivery.generate_id(),
            event_id=event.id,
            agent_id=event.agent_id,
            webhook_url=webhook_url,
            event_type=event.event_type,
            max_attempts=self.max_retries,
        )

        payload_json = event.model_dump_json()
        signature = self._generate_signature(payload_json, secret) if secret else None
        webhook_payload = WebhookEventPayload.create(event, signature)

        while delivery.attempt_count < delivery.max_attempts:
            delivery.attempt_count += 1
            delivery.last_attempt_at = get_utc_time()

            start_time = time.monotonic()

            try:
                async with AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        webhook_url,
                        json=event.model_dump(mode="json"),
                        headers=webhook_payload.headers,
                    )

                delivery.response_time_ms = int((time.monotonic() - start_time) * 1000)
                delivery.status_code = response.status_code

                if 200 <= response.status_code < 300:
                    delivery.status = WebhookDeliveryStatus.DELIVERED
                    delivery.delivered_at = get_utc_time()
                    logger.info(
                        f"Webhook delivered successfully: event={event.id} "
                        f"url={webhook_url} status={response.status_code}"
                    )
                    return delivery

                delivery.error_message = f"HTTP {response.status_code}: {response.text[:500]}"
                logger.warning(
                    f"Webhook delivery failed: event={event.id} "
                    f"url={webhook_url} status={response.status_code} "
                    f"attempt={delivery.attempt_count}/{delivery.max_attempts}"
                )

            except TimeoutException:
                delivery.response_time_ms = int((time.monotonic() - start_time) * 1000)
                delivery.error_message = f"Timeout after {self.timeout_seconds}s"
                logger.warning(
                    f"Webhook timeout: event={event.id} url={webhook_url} "
                    f"attempt={delivery.attempt_count}/{delivery.max_attempts}"
                )

            except Exception as e:
                delivery.response_time_ms = int((time.monotonic() - start_time) * 1000)
                delivery.error_message = str(e)[:500]
                logger.error(
                    f"Webhook error: event={event.id} url={webhook_url} "
                    f"error={e} attempt={delivery.attempt_count}/{delivery.max_attempts}"
                )

            if delivery.should_retry():
                delivery.status = WebhookDeliveryStatus.RETRYING
                delay = delivery.get_next_retry_delay_seconds()
                delivery.next_retry_at = get_utc_time()
                logger.debug(f"Scheduling retry in {delay}s for event={event.id}")
            else:
                delivery.status = WebhookDeliveryStatus.FAILED
                logger.error(
                    f"Webhook delivery exhausted retries: event={event.id} "
                    f"url={webhook_url} last_error={delivery.error_message}"
                )

        return delivery

    def _generate_signature(self, payload: str, secret: str) -> str:
        timestamp = str(int(time.time()))
        signed_payload = f"{timestamp}.{payload}"
        signature = hmac.new(
            secret.encode("utf-8"),
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"t={timestamp},v1={signature}"

    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str, tolerance_seconds: int = 300) -> bool:
        try:
            parts = dict(part.split("=", 1) for part in signature.split(","))
            timestamp = int(parts.get("t", "0"))
            expected_sig = parts.get("v1", "")

            current_time = int(time.time())
            if abs(current_time - timestamp) > tolerance_seconds:
                return False

            signed_payload = f"{timestamp}.{payload}"
            computed_sig = hmac.new(
                secret.encode("utf-8"),
                signed_payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            return hmac.compare_digest(computed_sig, expected_sig)
        except Exception:
            return False
