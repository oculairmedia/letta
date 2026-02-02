"""merge multiple heads

Revision ID: merge_27de_4b9c
Revises: 27de0f58e076, 4b9c3d2e5f67
Create Date: 2026-01-26 02:30:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "merge_27de_4b9c"
down_revision: tuple = ("27de0f58e076", "4b9c3d2e5f67")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
