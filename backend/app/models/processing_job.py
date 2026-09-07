import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id"), nullable=False
    )
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[JobStatus] = mapped_column(
        # values_callable: persist members by their .value ("pending", ...) rather than
        # SQLAlchemy's default of .name ("PENDING", ...), to match the lowercase labels
        # the "job_status" Postgres enum type was created with (see the initial migration).
        Enum(
            JobStatus,
            name="job_status",
            values_callable=lambda cls: [member.value for member in cls],
        ),
        nullable=False,
        default=JobStatus.PENDING,
    )
    result_image_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id"), nullable=True
    )
    # Every result image the operation produced, in display order:
    # [{"key": "lh", "label": "LH - Horizontal detail", "image_id": "<uuid str>"}, ...].
    # One entry for most operations, several for wavelet_enhance. `result_image_id`
    # above always mirrors the first entry's image_id.
    channels: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
