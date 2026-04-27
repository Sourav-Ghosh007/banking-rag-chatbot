"""
OAuth2 Handler
==============
Simple explanation:
- OAuth2 lets users connect their Gmail / Google Calendar / Slack
  WITHOUT sharing their password with our app
- Google gives us a temporary "access token" after the user approves
- We encrypt that token with Fernet and store it in PostgreSQL
- When the comms/notify agent needs to send an email, it fetches + decrypts the token

Flow:
  1. User clicks "Connect Gmail" in the UI
  2. We redirect them to Google's login page (with our client_id)
  3. Google redirects back to our /auth/callback with a "code"
  4. We exchange that code for an access_token + refresh_token
  5. We encrypt and save the tokens to the database
"""

import json
import logging
import os
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — set these in your .env file
# ---------------------------------------------------------------------------
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

SLACK_CLIENT_ID      = os.environ.get("SLACK_CLIENT_ID", "")
SLACK_CLIENT_SECRET  = os.environ.get("SLACK_CLIENT_SECRET", "")
SLACK_REDIRECT_URI   = os.environ.get("SLACK_REDIRECT_URI", "http://localhost:8000/auth/slack/callback")

FERNET_KEY = os.environ.get("FERNET_KEY", "")

# Google OAuth endpoints
GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Slack OAuth endpoints
SLACK_AUTH_URL  = "https://slack.com/oauth/v2/authorize"
SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"

# What permissions we ask for
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",
    "openid",
    "email",
    "profile",
]
SLACK_SCOPES = ["chat:write", "channels:read", "channels:history"]


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------
def _get_fernet() -> Fernet:
    """Get the Fernet encryption object. Raises if key not configured."""
    if not FERNET_KEY:
        raise RuntimeError("FERNET_KEY not set in environment variables.")
    return Fernet(FERNET_KEY.encode())


def encrypt_token(token_data: dict) -> str:
    """Encrypt a token dict to a string for safe database storage."""
    f = _get_fernet()
    return f.encrypt(json.dumps(token_data).encode()).decode()


def decrypt_token(encrypted: str) -> dict:
    """Decrypt a stored token string back to a dict."""
    f = _get_fernet()
    return json.loads(f.decrypt(encrypted.encode()).decode())


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------
def get_google_auth_url(state: str) -> str:
    """
    Build the Google login URL to redirect the user to.
    'state' is a random string we use to verify the callback is legitimate.
    """
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         " ".join(GOOGLE_SCOPES),
        "access_type":   "offline",   # get a refresh_token so we can refresh without re-login
        "prompt":        "consent",   # always show consent screen
        "state":         state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


async def exchange_google_code(code: str) -> dict:
    """
    Exchange the authorization code (from Google callback) for access + refresh tokens.
    This is Step 4 of the OAuth flow.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  GOOGLE_REDIRECT_URI,
            "grant_type":    "authorization_code",
        })

    if resp.status_code != 200:
        logger.error("Google token exchange failed: %s", resp.text)
        raise HTTPException(status_code=400, detail="Google authentication failed.")

    token_data = resp.json()
    logger.info("Google OAuth token exchange successful")
    return token_data


async def refresh_google_token(refresh_token: str) -> dict:
    """
    Use the refresh_token to get a new access_token when the old one expires.
    Google access tokens expire after 1 hour.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(GOOGLE_TOKEN_URL, data={
            "refresh_token": refresh_token,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "grant_type":    "refresh_token",
        })

    if resp.status_code != 200:
        logger.error("Google token refresh failed: %s", resp.text)
        raise HTTPException(status_code=401, detail="Google token refresh failed. Please reconnect.")

    return resp.json()


# ---------------------------------------------------------------------------
# Slack OAuth
# ---------------------------------------------------------------------------
def get_slack_auth_url(state: str) -> str:
    """Build the Slack login URL to redirect the user to."""
    params = {
        "client_id":    SLACK_CLIENT_ID,
        "redirect_uri": SLACK_REDIRECT_URI,
        "scope":        ",".join(SLACK_SCOPES),
        "state":        state,
    }
    return f"{SLACK_AUTH_URL}?{urlencode(params)}"


async def exchange_slack_code(code: str) -> dict:
    """Exchange the Slack authorization code for an access token."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(SLACK_TOKEN_URL, data={
            "code":          code,
            "client_id":     SLACK_CLIENT_ID,
            "client_secret": SLACK_CLIENT_SECRET,
            "redirect_uri":  SLACK_REDIRECT_URI,
        })

    data = resp.json()
    if not data.get("ok"):
        logger.error("Slack token exchange failed: %s", data.get("error"))
        raise HTTPException(status_code=400, detail=f"Slack authentication failed: {data.get('error')}")

    logger.info("Slack OAuth token exchange successful")
    return {"access_token": data["access_token"], "team": data.get("team", {}).get("name", "")}


# ---------------------------------------------------------------------------
# Token save/load helpers (used by routers and agents)
# ---------------------------------------------------------------------------
async def save_provider_token(db, user_id: str, provider: str, token_data: dict) -> None:
    """Encrypt and save a provider token to the database."""
    from backend.db.crud import save_oauth_token
    encrypted = encrypt_token(token_data)
    await save_oauth_token(db, user_id, provider, encrypted)
    logger.info("Saved %s token for user %s", provider, user_id)


async def load_provider_token(db, user_id: str, provider: str) -> dict | None:
    """Load and decrypt a provider token from the database. Returns None if not found."""
    from backend.db.crud import get_oauth_token
    record = await get_oauth_token(db, user_id, provider)
    if not record:
        return None
    return decrypt_token(record.encrypted_token)
