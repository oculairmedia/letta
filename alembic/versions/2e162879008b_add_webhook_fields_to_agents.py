"""add webhook fields to agents

Revision ID: 2e162879008b
Revises: 39577145c45d
Create Date: 2026-01-01 19:42:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "2e162879008b"
down_revision: Union[str, None] = "39577145c45d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agents", sa.Column("webhook_url", sa.String(), nullable=True))
    op.add_column("agents", sa.Column("webhook_secret", sa.String(), nullable=True))
    op.add_column("agents", sa.Column("webhook_events", sa.JSON(), nullable=True))
    op.add_column("agents", sa.Column("webhook_enabled", sa.Boolean(), nullable=False, server_default="false"))


def downgrade() -> None:
    op.drop_column("agents", "webhook_enabled")
    op.drop_column("agents", "webhook_events")
    op.drop_column("agents", "webhook_secret")
    op.drop_column("agents", "webhook_url")
