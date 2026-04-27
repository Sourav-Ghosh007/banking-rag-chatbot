"""
PKCE (Proof Key for Code Exchange)
====================================
Simple explanation:
- PKCE is a security upgrade for OAuth2
- Problem: Without PKCE, if someone intercepts the "code" from Google's callback,
  they could steal your tokens
- Solution: PKCE adds a secret "code_verifier" that only our app knows
  Even if someone steals the code, they can't use it without the verifier

How it works:
  1. We generate a random "code_verifier" string
  2. We hash it → "code_challenge"
  3. We send code_challenge to Google when starting login
  4. When exchanging the code for tokens, we send the original code_verifier
  5. Google checks: hash(code_verifier) == code_challenge → only our app can do this

Used in banking apps because they handle sensitive financial data.
"""

import base64
import hashlib
import os
import secrets
import string


def generate_code_verifier(length: int = 64) -> str:
    """
    Generate a random code_verifier string.
    Must be 43-128 characters, URL-safe characters only.

    Example output: "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    """
    # Use only URL-safe characters (letters, digits, -, ., _, ~)
    alphabet = string.ascii_letters + string.digits + "-._~"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_code_challenge(code_verifier: str) -> str:
    """
    Hash the code_verifier using SHA-256 → code_challenge.
    This is what we send to Google. Google stores it.
    When we send back the verifier, Google hashes it and compares.
    """
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    # Base64 URL-encode without padding (=)
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def generate_state_token() -> str:
    """
    Generate a random 'state' token for CSRF protection.
    We send this with the OAuth request and verify we get it back in the callback.
    This prevents Cross-Site Request Forgery attacks.
    """
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# In-memory store for PKCE verifiers (per session)
# In production: store in Redis or the database with expiry
# ---------------------------------------------------------------------------
_pkce_store: dict[str, dict] = {}


def store_pkce_session(state: str, code_verifier: str, user_id: str | None = None) -> None:
    """
    Temporarily store the code_verifier linked to a state token.
    We need this when the OAuth callback comes back.
    """
    _pkce_store[state] = {
        "code_verifier": code_verifier,
        "user_id": user_id,
    }


def retrieve_pkce_session(state: str) -> dict | None:
    """
    Retrieve and remove the PKCE session for a given state token.
    Returns None if state is invalid or expired.
    """
    return _pkce_store.pop(state, None)


def create_pkce_pair() -> dict:
    """
    Convenience function: create a full PKCE pair in one call.

    Returns:
        {
            "code_verifier":  "...",   ← keep secret, send when exchanging code
            "code_challenge": "...",   ← send to Google when starting login
            "state":          "...",   ← send with request, verify in callback
        }
    """
    verifier   = generate_code_verifier()
    challenge  = generate_code_challenge(verifier)
    state      = generate_state_token()

    return {
        "code_verifier":  verifier,
        "code_challenge": challenge,
        "state":          state,
    }
