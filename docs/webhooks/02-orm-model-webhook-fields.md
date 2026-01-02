# Issue: Add webhook configuration fields to Agent ORM model

**Type:** Enhancement  
**Priority:** High  
**Labels:** `enhancement`, `webhooks`, `database`, `orm`

## Overview
Extend the Agent ORM model to store webhook configuration (URL, secret, enabled events). This follows the existing pattern used in Job and Run models which already have `callback_url` fields.

## Scope
Add webhook configuration fields to the Agent model so each agent can have its own webhook endpoint configuration.

## Database Schema Changes

### Agent Model (`letta/orm/agent.py`)
Add the following fields:
```python
webhook_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
webhook_secret: Mapped[Optional[str]] = mapped_column(String, nullable=True)
webhook_events: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True, default=list)
webhook_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
```

### Webhook Delivery Tracking (New Model)
Create `letta/orm/webhook_delivery.py`:
```python
class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    organization_id: Mapped[str] = mapped_column(String, ForeignKey("organizations.id"))
    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id"))
    
    event_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    webhook_url: Mapped[str] = mapped_column(String, nullable=False)
    
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=1)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_retry_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    agent: Mapped["AgentModel"] = relationship("AgentModel", back_populates="webhook_deliveries")
```

## Pydantic Schema Changes

### Agent Schema (`letta/schemas/agent.py`)
Add fields to `AgentState`:
```python
webhook_url: Optional[str] = None
webhook_secret: Optional[str] = None
webhook_events: Optional[List[str]] = None
webhook_enabled: bool = False
```

### New Schema: WebhookConfig
Create `letta/schemas/webhook.py`:
```python
class WebhookConfig(LettaBase):
    webhook_url: HttpUrl
    webhook_secret: Optional[str] = None
    webhook_events: List[str] = []
    webhook_enabled: bool = True

class WebhookDelivery(LettaBase):
    id: str
    agent_id: str
    event_id: str
    event_type: str
    webhook_url: str
    status_code: Optional[int]
    error_message: Optional[str]
    attempt_count: int
    created_at: datetime
    delivered_at: Optional[datetime]
    next_retry_at: Optional[datetime]
```

## Database Migration

### Alembic Migration Script
Create migration: `alembic revision -m "add_webhook_fields_to_agent"`

```python
def upgrade():
    # Add webhook fields to agents table
    op.add_column('agents', sa.Column('webhook_url', sa.String(), nullable=True))
    op.add_column('agents', sa.Column('webhook_secret', sa.String(), nullable=True))
    op.add_column('agents', sa.Column('webhook_events', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('webhook_enabled', sa.Boolean(), default=False))
    
    # Create webhook_deliveries table
    op.create_table(
        'webhook_deliveries',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('organization_id', sa.String(), sa.ForeignKey('organizations.id')),
        sa.Column('agent_id', sa.String(), sa.ForeignKey('agents.id')),
        sa.Column('event_id', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('webhook_url', sa.String(), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('attempt_count', sa.Integer(), default=1),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('delivered_at', sa.DateTime(), nullable=True),
        sa.Column('next_retry_at', sa.DateTime(), nullable=True),
    )
    
    # Create indexes
    op.create_index('ix_webhook_deliveries_event_id', 'webhook_deliveries', ['event_id'])
    op.create_index('ix_webhook_deliveries_agent_id', 'webhook_deliveries', ['agent_id'])

def downgrade():
    op.drop_table('webhook_deliveries')
    op.drop_column('agents', 'webhook_enabled')
    op.drop_column('agents', 'webhook_events')
    op.drop_column('agents', 'webhook_secret')
    op.drop_column('agents', 'webhook_url')
```

## Acceptance Criteria
- [ ] Agent model has webhook configuration fields
- [ ] WebhookDelivery model created for tracking delivery history
- [ ] Pydantic schemas updated (AgentState, new WebhookConfig)
- [ ] Alembic migration script created and tested
- [ ] Migration runs cleanly on existing databases
- [ ] Backward compatibility maintained (all fields nullable/optional)

## Security Considerations
- `webhook_secret` should be encrypted at rest (consider using SQLAlchemy EncryptedType)
- Webhook secrets should not be returned in API responses (use `Field(exclude=True)` in Pydantic)
- Consider adding webhook URL validation (no localhost/private IPs in production)

## Files to Modify/Create
- `letta/orm/agent.py` (modify)
- `letta/orm/webhook_delivery.py` (create)
- `letta/schemas/agent.py` (modify)
- `letta/schemas/webhook.py` (create)
- `alembic/versions/XXXX_add_webhook_fields_to_agent.py` (create)

## Related Issues
- `01-webhook-module-foundation.md` - WebhookManager will query these fields
- `03-webhook-settings-config.md` - Global webhook settings
- `05-webhook-crud-api.md` - API endpoints to manage these fields

## References
- Existing callback fields in Job: `letta/orm/job.py:49` (`callback_url`, `callback_sent_at`, etc.)
- Existing callback fields in Run: `letta/orm/run.py:54`
- Agent model: `letta/orm/agent.py`
