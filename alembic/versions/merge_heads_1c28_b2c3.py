"""merge heads for 0.16.7 upgrade

Revision ID: merge_1c28_b2c3
Revises: 1c28e167b74f, merge_b2c3_a1b2
Create Date: 2026-04-01 17:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "merge_1c28_b2c3"
down_revision: Union[str, Sequence[str]] = ("1c28e167b74f", "merge_b2c3_a1b2")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
