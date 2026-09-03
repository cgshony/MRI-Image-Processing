"""initial images and processing_jobs tables

Revision ID: 8a5f6b742b9b
Revises:
Create Date: 2026-09-03 10:00:30.689039

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '8a5f6b742b9b'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "images",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("parent_image_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["parent_image_id"], ["images.id"]),
    )

    job_status = postgresql.ENUM(
        "pending", "running", "done", "failed", name="job_status"
    )
    job_status.create(op.get_bind())

    op.create_table(
        "processing_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("image_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("operation", sa.String(length=100), nullable=False),
        sa.Column("params", postgresql.JSONB(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("pending", "running", "done", "failed", name="job_status", create_type=False),
            nullable=False,
        ),
        sa.Column("result_image_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"]),
        sa.ForeignKeyConstraint(["result_image_id"], ["images.id"]),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("processing_jobs")
    postgresql.ENUM(name="job_status").drop(op.get_bind())
    op.drop_table("images")
