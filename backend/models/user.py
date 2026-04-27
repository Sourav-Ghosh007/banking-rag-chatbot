"""User model — SQLite compatible."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, String
from backend.db.session import Base


class User(Base):
    __tablename__ = "users"

    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email          = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name      = Column(String, default="")
    is_active      = Column(Boolean, default=True)
    created_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login     = Column(DateTime, nullable=True)
