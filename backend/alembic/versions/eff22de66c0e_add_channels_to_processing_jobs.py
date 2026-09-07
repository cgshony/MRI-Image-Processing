"""add channels to processing_jobs

Revision ID: eff22de66c0e
Revises: 8a5f6b742b9b
Create Date: 2026-09-07 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'eff22de66c0e'
down_revision: Union[str, Sequence[str], None] = '8a5f6b742b9b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "processing_jobs",
        sa.Column("channels", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("processing_jobs", "channels")
