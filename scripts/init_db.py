"""
Database Initializer
====================
Run this once before starting the app to create all PostgreSQL tables.
Usage: python scripts/init_db.py
"""

import asyncio
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.db.session  import init_db
from backend.rag.metrics import ensure_metrics_table
from backend.utils.logger import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


async def main():
    logger.info("Creating database tables...")
    await init_db()
    await ensure_metrics_table()
    logger.info("✅ All tables created successfully!")
    logger.info("Tables created: users, oauth_tokens, audit_logs, rag_metrics")


if __name__ == "__main__":
    asyncio.run(main())
