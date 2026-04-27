"""
CRUD Operations
===============
CRUD = Create, Read, Update, Delete

Simple explanation:
- This file talks to PostgreSQL via SQLAlchemy
- Every time we need to save or fetch a user, token, or audit log — we call functions here
- Keeps all database logic in one place instead of scattered across the app
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.user  import User
from backend.models.token import OAuthToken
from backend.models.audit import AuditLog

logger = logging.getLogger(__name__)


# ===========================================================================
# USER operations
# ===========================================================================

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Find a user by their email address. Returns None if not found."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Find a user by their ID. Returns None if not found."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, email: str, hashed_password: str, full_name: str = "") -> User:
    """
    Create a new user account.
    Password must already be hashed before calling this — never store plain text passwords.
    """
    user = User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info("Created new user: %s", email)
    return user


async def update_user_last_login(db: AsyncSession, user_id: str) -> None:
    """Update the last login timestamp for a user."""
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(last_login=datetime.now(timezone.utc))
    )
    await db.commit()


async def deactivate_user(db: AsyncSession, user_id: str) -> None:
    """Deactivate a user account (soft delete — does not remove from DB)."""
    await db.execute(
        update(User).where(User.id == user_id).values(is_active=False)
    )
    await db.commit()
    logger.info("Deactivated user: %s", user_id)


# ===========================================================================
# OAUTH TOKEN operations
# ===========================================================================

async def save_oauth_token(
    db: AsyncSession,
    user_id: str,
    provider: str,          # "gmail", "google_calendar", "slack"
    encrypted_token: str,   # Fernet-encrypted JSON token
) -> OAuthToken:
    """
    Save or update an OAuth token for a user+provider pair.
    If a token already exists for this user+provider, it is replaced.
    """
    # Check if token already exists
    result = await db.execute(
        select(OAuthToken).where(
            OAuthToken.user_id == user_id,
            OAuthToken.provider == provider,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        # Update existing token
        existing.encrypted_token = encrypted_token
        existing.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(existing)
        logger.info("Updated OAuth token for user %s / provider %s", user_id, provider)
        return existing
    else:
        # Create new token record
        token = OAuthToken(
            user_id=user_id,
            provider=provider,
            encrypted_token=encrypted_token,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(token)
        await db.commit()
        await db.refresh(token)
        logger.info("Saved new OAuth token for user %s / provider %s", user_id, provider)
        return token


async def get_oauth_token(db: AsyncSession, user_id: str, provider: str) -> Optional[OAuthToken]:
    """Fetch the stored OAuth token for a user+provider. Returns None if not connected."""
    result = await db.execute(
        select(OAuthToken).where(
            OAuthToken.user_id == user_id,
            OAuthToken.provider == provider,
        )
    )
    return result.scalar_one_or_none()


async def delete_oauth_token(db: AsyncSession, user_id: str, provider: str) -> None:
    """Remove an OAuth token (user disconnected a service)."""
    await db.execute(
        delete(OAuthToken).where(
            OAuthToken.user_id == user_id,
            OAuthToken.provider == provider,
        )
    )
    await db.commit()
    logger.info("Deleted OAuth token for user %s / provider %s", user_id, provider)


async def get_connected_providers(db: AsyncSession, user_id: str) -> list[str]:
    """Return list of providers the user has connected (e.g. ['gmail', 'slack'])."""
    result = await db.execute(
        select(OAuthToken.provider).where(OAuthToken.user_id == user_id)
    )
    return [row[0] for row in result.fetchall()]


# ===========================================================================
# AUDIT LOG operations
# ===========================================================================

async def create_audit_log(
    db: AsyncSession,
    user_id: str,
    action: str,        # e.g. "login", "file_upload", "query"
    details: str = "",  # extra info about the action
    ip_address: str = "",
) -> AuditLog:
    """
    Write an audit log entry.
    Audit logs track who did what and when — important for banking compliance.
    """
    log = AuditLog(
        user_id=user_id,
        action=action,
        details=details,
        ip_address=ip_address,
        created_at=datetime.now(timezone.utc),
    )
    db.add(log)
    await db.commit()
    return log


async def get_user_audit_logs(db: AsyncSession, user_id: str, limit: int = 50) -> list[AuditLog]:
    """Fetch the last N audit log entries for a user."""
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()
