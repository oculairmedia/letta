"""add created_by fields to webhook_deliveries

Revision ID: 4b9c3d2e5f67
Revises: 3a8f2c1d4e56
Create Date: 2026-01-02 17:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "4b9c3d2e5f67"
down_revision: Union[str, None] = "3a8f2c1d4e56"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add missing columns from CommonSqlalchemyMetaMixins
    op.add_column("webhook_deliveries", sa.Column("_created_by_id", sa.String(), nullable=True))
    op.add_column("webhook_deliveries", sa.Column("_last_updated_by_id", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("webhook_deliveries", "_last_updated_by_id")
    op.drop_column("webhook_deliveries", "_created_by_id")
