"""
Database Session - SQLite version (no PostgreSQL needed)
Works locally without any database installation.
"""

import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# SQLite file-based database — no installation needed
# Creates a file called banking.db in your project folder
SQLITE_URL = "sqlite+aiosqlite:///./banking.db"

engine = create_async_engine(
    SQLITE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


async def init_db():
    """Create all database tables."""
    from backend.models.user  import User
    from backend.models.token import OAuthToken
    from backend.models.audit import AuditLog
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ SQLite database tables created: banking.db")


async def get_db():
    """FastAPI dependency — yields a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
