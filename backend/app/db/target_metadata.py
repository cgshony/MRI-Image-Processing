"""Single import point that pulls in every ORM model.

`db/base.py` deliberately does not import model modules itself (see its
docstring), to avoid a circular import since each model imports `Base` from
there. Alembic's `env.py` imports `target_metadata` from here so
autogenerate can see the full schema; import this module (rather than
`db.base` directly) anywhere you need `Base.metadata` to reflect every table.
"""

from app.db.base import Base
from app.models.image import Image  # noqa: F401
from app.models.processing_job import ProcessingJob  # noqa: F401

target_metadata = Base.metadata
