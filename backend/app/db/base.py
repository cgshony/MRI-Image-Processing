from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models.

    Deliberately does not import any model modules (that would create a
    circular import, since models import Base from here). See
    target_metadata.py for the module that pulls every model in so
    Alembic autogenerate can see the full schema.
    """
