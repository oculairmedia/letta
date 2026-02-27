"""merge heads for 0.16.5 upgrade

Revision ID: merge_b2c3_a1b2
Revises: b2c3d4e5f6a8, merge_a1b2_27de
Create Date: 2026-02-26 12:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "merge_b2c3_a1b2"
down_revision: Union[str, Sequence[str]] = ("b2c3d4e5f6a8", "merge_a1b2_27de")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
