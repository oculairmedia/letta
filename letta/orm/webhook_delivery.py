import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.webhook import WebhookDelivery as PydanticWebhookDelivery
from letta.schemas.webhook import WebhookDeliveryStatus, WebhookEventType

if TYPE_CHECKING:
    from letta.orm.agent import Agent
    from letta.orm.organization import Organization


class WebhookDeliveryORM(SqlalchemyBase, OrganizationMixin):
    """Persists webhook delivery attempts for auditing and retry tracking."""

    __tablename__ = "webhook_deliveries"
    __pydantic_model__ = PydanticWebhookDelivery
    __table_args__ = (
        Index("ix_webhook_deliveries_created_at", "created_at", "id"),
        Index("ix_webhook_deliveries_agent_id", "agent_id"),
        Index("ix_webhook_deliveries_event_id", "event_id"),
        Index("ix_webhook_deliveries_status", "status"),
        Index("ix_webhook_deliveries_organization_id", "organization_id"),
    )

    # Generate delivery ID with whd- prefix
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"whd-{uuid.uuid4()}")

    # Event identification
    event_id: Mapped[str] = mapped_column(String, nullable=False, doc="The event ID being delivered")
    event_type: Mapped[str] = mapped_column(String, nullable=False, doc="The type of event")

    # Target information
    webhook_url: Mapped[str] = mapped_column(String, nullable=False, doc="The target URL for delivery")

    # Delivery status
    status: Mapped[str] = mapped_column(
        String, default=WebhookDeliveryStatus.PENDING.value, doc="Current delivery status"
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, doc="Number of delivery attempts made")
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, doc="Maximum number of delivery attempts")

    # Timestamps
    last_attempt_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="Timestamp of the last delivery attempt"
    )
    delivered_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="Timestamp when successfully delivered"
    )
    next_retry_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="Scheduled time for next retry attempt"
    )

    # Response information
    status_code: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, doc="HTTP status code from the last attempt"
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="Error message from the last failed attempt"
    )
    response_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, doc="Response time in milliseconds"
    )

    # Agent relationship
    agent_id: Mapped[str] = mapped_column(
        String, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, doc="The agent this delivery is for"
    )

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", lazy="raise")
    organization: Mapped[Optional["Organization"]] = relationship("Organization", lazy="raise")

    def to_pydantic(self) -> PydanticWebhookDelivery:
        """Converts the SQLAlchemy model to its corresponding Pydantic model."""
        from letta.helpers.datetime_helpers import get_utc_time

        return PydanticWebhookDelivery(
            id=self.id,
            event_id=self.event_id,
            agent_id=self.agent_id,
            webhook_url=self.webhook_url,
            event_type=WebhookEventType(self.event_type),
            status=WebhookDeliveryStatus(self.status),
            attempt_count=self.attempt_count,
            max_attempts=self.max_attempts,
            created_at=self.created_at or get_utc_time(),
            last_attempt_at=self.last_attempt_at,
            delivered_at=self.delivered_at,
            next_retry_at=self.next_retry_at,
            status_code=self.status_code,
            error_message=self.error_message,
            response_time_ms=self.response_time_ms,
        )

    @classmethod
    def from_pydantic(
        cls,
        delivery: PydanticWebhookDelivery,
        organization_id: str,
    ) -> "WebhookDeliveryORM":
        """Creates an ORM instance from a Pydantic model."""
        return cls(
            id=delivery.id,
            event_id=delivery.event_id,
            agent_id=delivery.agent_id,
            webhook_url=delivery.webhook_url,
            event_type=delivery.event_type.value,
            status=delivery.status.value,
            attempt_count=delivery.attempt_count,
            max_attempts=delivery.max_attempts,
            last_attempt_at=delivery.last_attempt_at,
            delivered_at=delivery.delivered_at,
            next_retry_at=delivery.next_retry_at,
            status_code=delivery.status_code,
            error_message=delivery.error_message,
            response_time_ms=delivery.response_time_ms,
            organization_id=organization_id,
        )
