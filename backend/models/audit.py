"""Audit Log model — SQLite compatible."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String
from backend.db.session import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, nullable=False, index=True)
    action     = Column(String, nullable=False)
    details    = Column(String, default="")
    ip_address = Column(String, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
