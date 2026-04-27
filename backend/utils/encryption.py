"""
Encryption Utilities
====================
Simple explanation:
- We use Fernet encryption (from the cryptography library)
- Fernet = symmetric encryption — same key encrypts and decrypts
- We use it to safely store OAuth tokens (Gmail, Slack) in the database
- Even if the database is leaked, tokens are unreadable without the FERNET_KEY

The FERNET_KEY is stored in your .env file — never commit it to git.

How to generate a new key (run once, paste in .env):
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

import json
import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    """
    Get the Fernet cipher using the key from environment.
    Raises RuntimeError if FERNET_KEY is not set.
    """
    key = os.environ.get("FERNET_KEY", "")
    if not key:
        raise RuntimeError(
            "FERNET_KEY is not set. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode())


# ---------------------------------------------------------------------------
# String encryption (for tokens stored in DB)
# ---------------------------------------------------------------------------

def encrypt_string(plain_text: str) -> str:
    """
    Encrypt a plain string and return an encrypted string.
    Safe to store in PostgreSQL TEXT columns.
    """
    f = _get_fernet()
    return f.encrypt(plain_text.encode()).decode()


def decrypt_string(encrypted_text: str) -> str:
    """
    Decrypt an encrypted string back to plain text.
    Raises InvalidToken if the key is wrong or data is corrupted.
    """
    f = _get_fernet()
    try:
        return f.decrypt(encrypted_text.encode()).decode()
    except InvalidToken:
        logger.error("Decryption failed — wrong key or corrupted data")
        raise ValueError("Could not decrypt data. The encryption key may have changed.")


# ---------------------------------------------------------------------------
# Dict encryption (for OAuth tokens which are JSON objects)
# ---------------------------------------------------------------------------

def encrypt_dict(data: dict) -> str:
    """
    Encrypt a Python dict (converts to JSON first, then encrypts).
    Used for storing OAuth tokens like:
        {"access_token": "...", "refresh_token": "...", "expires_in": 3600}
    """
    json_str = json.dumps(data)
    return encrypt_string(json_str)


def decrypt_dict(encrypted_text: str) -> dict:
    """
    Decrypt an encrypted string back to a Python dict.
    Used for reading OAuth tokens from the database.
    """
    json_str = decrypt_string(encrypted_text)
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Key generation helper
# ---------------------------------------------------------------------------

def generate_fernet_key() -> str:
    """
    Generate a new Fernet key.
    Run this once to create your FERNET_KEY — paste it in .env.
    """
    return Fernet.generate_key().decode()


# ---------------------------------------------------------------------------
# Password hashing (for user login — separate from Fernet)
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """
    Hash a password using bcrypt via passlib.
    We NEVER store plain text passwords — always store the hash.
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check if a plain password matches the stored hash.
    Returns True if correct, False if wrong password.
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.verify(plain_password, hashed_password)
