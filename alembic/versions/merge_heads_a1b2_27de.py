"""merge multiple heads after upstream 0.16.4 sync

Revision ID: merge_a1b2_27de
Revises: a1b2c3d4e5f8, merge_27de_4b9c
Create Date: 2026-02-02 06:50:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "merge_a1b2_27de"
down_revision: tuple = ("a1b2c3d4e5f8", "merge_27de_4b9c")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
