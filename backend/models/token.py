"""OAuth Token model — SQLite compatible."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String
from backend.db.session import Base


class OAuthToken(Base):
    __tablename__ = "user_oauth_tokens"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id         = Column(String, nullable=False, index=True)
    provider        = Column(String, nullable=False)   # gmail, google_calendar, slack
    encrypted_token = Column(String, nullable=False)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))
