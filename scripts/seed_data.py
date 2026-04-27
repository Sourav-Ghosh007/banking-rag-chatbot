"""
Seed Data Script
================
Creates a test user and ingests sample data files so you can demo immediately.
Usage: python scripts/seed_data.py
"""

import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.db.session      import AsyncSessionLocal
from backend.db.crud         import create_user, get_user_by_email
from backend.utils.encryption import hash_password
from backend.utils.logger    import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)

# Test credentials — change before real deployment
TEST_EMAIL    = "demo@bankingchatbot.com"
TEST_PASSWORD = "Demo@1234"
TEST_NAME     = "Demo User"


async def seed_users():
    """Create a demo user account."""
    async with AsyncSessionLocal() as db:
        existing = await get_user_by_email(db, TEST_EMAIL)
        if existing:
            logger.info("Demo user already exists: %s", TEST_EMAIL)
            return str(existing.id)

        hashed = hash_password(TEST_PASSWORD)
        user   = await create_user(db, TEST_EMAIL, hashed, TEST_NAME)
        logger.info("✅ Created demo user: %s (id=%s)", TEST_EMAIL, user.id)
        return str(user.id)


def seed_sample_files(user_id: str):
    """Ingest the sample data files from the /data folder."""
    from backend.rag.ingest import ingest_file

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    sample_files = [
        "transactions.csv",
        "sample_loans.xlsx",
        "rates.xlsx",
    ]

    for filename in sample_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            logger.warning("Sample file not found: %s — skipping", filepath)
            continue
        try:
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            result = ingest_file(file_bytes, filename, user_id)
            logger.info(
                "✅ Ingested %s → %d rows, %d chunks",
                filename, result["total_rows"], result["chunks_ingested"],
            )
        except Exception as e:
            logger.error("Failed to ingest %s: %s", filename, e)


async def main():
    logger.info("Seeding database with demo data...")
    user_id = await seed_users()
    seed_sample_files(user_id)
    logger.info("✅ Seed complete!")
    logger.info("Login with: email=%s  password=%s", TEST_EMAIL, TEST_PASSWORD)


if __name__ == "__main__":
    asyncio.run(main())
