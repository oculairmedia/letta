from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from letta.helpers.datetime_helpers import get_utc_time


class WebhookEventType(str, Enum):
    AGENT_STEP_COMPLETED = "agent.step.completed"
    AGENT_STEP_FAILED = "agent.step.failed"
    AGENT_MESSAGE_SENT = "agent.message.sent"
    AGENT_STATE_UPDATED = "agent.state.updated"
    AGENT_TOOL_ATTACHED = "agent.tool.attached"
    AGENT_TOOL_DETACHED = "agent.tool.detached"
    TOOL_EXECUTION_COMPLETED = "tool.execution.completed"
    TOOL_EXECUTION_FAILED = "tool.execution.failed"
    AGENT_RUN_STARTED = "agent.run.started"
    AGENT_RUN_COMPLETED = "agent.run.completed"
    AGENT_RUN_FAILED = "agent.run.failed"
    AGENT_JOB_COMPLETED = "agent.job.completed"
    AGENT_JOB_FAILED = "agent.job.failed"
    AGENT_MEMORY_UPDATED = "agent.memory.updated"


class WebhookDeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class WebhookEvent(BaseModel):
    id: str = Field(..., description="Unique identifier for this event (format: evt-{uuid})")
    event_type: WebhookEventType = Field(..., description="The type of event that occurred")
    timestamp: datetime = Field(default_factory=get_utc_time, description="ISO 8601 timestamp when the event was created")
    agent_id: str = Field(..., description="The agent that triggered this event")
    organization_id: str = Field(..., description="The organization that owns the agent")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific payload data")

    @classmethod
    def generate_event_id(cls) -> str:
        import uuid

        return f"evt-{uuid.uuid4()}"


class WebhookConfig(BaseModel):
    url: str = Field(..., description="The HTTPS URL to send webhook events to")
    secret: Optional[str] = Field(default=None, description="Shared secret for HMAC-SHA256 signature verification")
    events: List[WebhookEventType] = Field(default_factory=list, description="Event types to subscribe to (empty = all)")
    enabled: bool = Field(default=True, description="Whether this webhook is currently enabled")
    description: Optional[str] = Field(default=None, description="Optional description for this webhook")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v:
            raise ValueError("Webhook URL cannot be empty")
        if not v.startswith(("http://", "https://")):
            raise ValueError("Webhook URL must start with http:// or https://")
        return v

    def is_subscribed_to(self, event_type: WebhookEventType) -> bool:
        return not self.events or event_type in self.events


class WebhookDelivery(BaseModel):
    id: str = Field(..., description="Unique identifier for this delivery attempt")
    event_id: str = Field(..., description="The event ID being delivered")
    agent_id: str = Field(..., description="The agent this delivery is for")
    webhook_url: str = Field(..., description="The target URL for delivery")
    event_type: WebhookEventType = Field(..., description="The type of event")

    status: WebhookDeliveryStatus = Field(default=WebhookDeliveryStatus.PENDING, description="Current delivery status")
    attempt_count: int = Field(default=0, description="Number of delivery attempts made")
    max_attempts: int = Field(default=3, description="Maximum number of delivery attempts")

    created_at: datetime = Field(default_factory=get_utc_time, description="When the delivery was created")
    last_attempt_at: Optional[datetime] = Field(default=None, description="Timestamp of the last delivery attempt")
    delivered_at: Optional[datetime] = Field(default=None, description="Timestamp when successfully delivered")
    next_retry_at: Optional[datetime] = Field(default=None, description="Scheduled time for next retry attempt")

    status_code: Optional[int] = Field(default=None, description="HTTP status code from the last attempt")
    error_message: Optional[str] = Field(default=None, description="Error message from the last failed attempt")
    response_time_ms: Optional[int] = Field(default=None, description="Response time in milliseconds")

    def should_retry(self) -> bool:
        if self.status == WebhookDeliveryStatus.DELIVERED:
            return False
        return self.attempt_count < self.max_attempts

    def get_next_retry_delay_seconds(self) -> int:
        import random

        BASE_DELAY_SECONDS = 5
        MAX_DELAY_SECONDS = 3600
        JITTER_FACTOR = 0.25

        delay = min(BASE_DELAY_SECONDS * (2**self.attempt_count), MAX_DELAY_SECONDS)
        jitter = delay * JITTER_FACTOR * (random.random() * 2 - 1)
        return int(delay + jitter)

    @classmethod
    def generate_id(cls) -> str:
        import uuid

        return f"whd-{uuid.uuid4()}"


class WebhookEventPayload(BaseModel):
    event: WebhookEvent = Field(..., description="The webhook event data")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers for the request")

    @classmethod
    def create(cls, event: WebhookEvent, signature: Optional[str] = None) -> "WebhookEventPayload":
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Letta-Webhooks/1.0",
            "X-Letta-Event-Type": event.event_type.value,
            "X-Letta-Event-Id": event.id,
            "X-Letta-Timestamp": str(int(event.timestamp.timestamp())),
        }
        if signature:
            headers["X-Letta-Signature"] = signature
        return cls(event=event, headers=headers)


class AgentWebhookConfig(BaseModel):
    """Agent webhook configuration returned by the API (secret is masked)."""

    url: Optional[str] = Field(None, description="The URL to send webhook events to.")
    events: List[WebhookEventType] = Field(default_factory=list, description="List of event types to send to the webhook (empty = all).")
    enabled: bool = Field(False, description="Whether webhooks are enabled for this agent.")
    has_secret: bool = Field(False, description="Whether a webhook secret is configured (actual secret is not exposed).")


class AgentWebhookConfigUpdate(BaseModel):
    """Request body for updating agent webhook configuration."""

    url: Optional[str] = Field(None, description="The URL to send webhook events to. Set to null to remove.")
    secret: Optional[str] = Field(None, description="Shared secret for HMAC-SHA256 signature verification. Set to null to remove.")
    events: Optional[List[WebhookEventType]] = Field(None, description="Event types to subscribe to (empty list = all events).")
    enabled: Optional[bool] = Field(None, description="Whether webhooks are enabled for this agent.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != "":
            if not v.startswith(("http://", "https://")):
                raise ValueError("Webhook URL must start with http:// or https://")
        return v
