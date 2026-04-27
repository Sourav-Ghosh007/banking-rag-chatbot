"""
Auth Router
===========
Simple explanation:
- These are the API endpoints for user login and registration
- POST /api/auth/register  → create a new account
- POST /api/auth/login     → login, get a JWT token back
- GET  /api/auth/me        → get current user's profile
- POST /api/auth/logout    → logout (client deletes the token)
- GET  /api/auth/google    → start Google OAuth login
- GET  /api/auth/google/callback → Google calls this after user approves

JWT = JSON Web Token — a signed string the client sends with every request
to prove who they are (like a session ID but stateless)
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session  import get_db
from backend.db.crud     import (
    create_user, get_user_by_email, update_user_last_login, create_audit_log
)
from backend.auth.oauth  import (
    get_google_auth_url, exchange_google_code, save_provider_token
)
from backend.auth.pkce   import create_pkce_pair, store_pkce_session, retrieve_pkce_session
from backend.utils.encryption import hash_password, verify_password

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# ---------------------------------------------------------------------------
# JWT config — set SECRET_KEY in .env
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-in-production")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ---------------------------------------------------------------------------
# Pydantic models — define what the API accepts and returns
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email:     EmailStr
    password:  str
    full_name: str = ""

class LoginResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    email:        str

class UserProfile(BaseModel):
    user_id:    str
    email:      str
    full_name:  str
    created_at: datetime


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def create_jwt_token(user_id: str, email: str) -> str:
    """Create a JWT token that expires in 24 hours."""
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """
    Dependency — validates the JWT token and returns the current user.
    Add Depends(get_current_user) to any endpoint that requires login.
    """
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please login again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise credentials_error
    except JWTError:
        raise credentials_error

    from backend.db.crud import get_user_by_id
    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise credentials_error
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", summary="Create a new user account")
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Register a new user.
    - Checks email is not already taken
    - Hashes the password (never stored in plain text)
    - Creates the user in PostgreSQL
    """
    # Check if email already registered
    existing = await get_user_by_email(db, data.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    # Hash password and create user
    hashed_pw = hash_password(data.password)
    user = await create_user(db, data.email, hashed_pw, data.full_name)

    await create_audit_log(db, str(user.id), "register", f"New account: {data.email}")
    logger.info("New user registered: %s", data.email)

    return {"message": "Account created successfully.", "user_id": str(user.id)}


@router.post("/login", response_model=LoginResponse, summary="Login and get JWT token")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db:   AsyncSession = Depends(get_db),
):
    """
    Login with email + password.
    Returns a JWT token — the frontend stores this and sends it with every request.
    """
    user = await get_user_by_email(db, form.username)   # username = email in OAuth2 form

    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated.")

    token = create_jwt_token(str(user.id), user.email)
    await update_user_last_login(db, str(user.id))
    await create_audit_log(db, str(user.id), "login", f"Login from email/password")

    logger.info("User logged in: %s", user.email)
    return LoginResponse(
        access_token=token,
        user_id=str(user.id),
        email=user.email,
    )


@router.get("/me", response_model=UserProfile, summary="Get current user profile")
async def get_me(current_user=Depends(get_current_user)):
    """Returns the profile of the currently logged-in user."""
    return UserProfile(
        user_id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name or "",
        created_at=current_user.created_at,
    )


@router.post("/logout", summary="Logout (client should delete the token)")
async def logout(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Logout endpoint.
    Since JWT tokens are stateless, the client just needs to delete the token.
    We log the event for audit purposes.
    """
    await create_audit_log(db, str(current_user.id), "logout", "User logged out")
    return {"message": "Logged out successfully."}


# ---------------------------------------------------------------------------
# Google OAuth endpoints
# ---------------------------------------------------------------------------

@router.get("/google", summary="Start Google OAuth login")
async def google_login():
    """
    Redirect the user to Google's login page.
    After they approve, Google redirects back to /google/callback.
    """
    pkce = create_pkce_pair()
    store_pkce_session(pkce["state"], pkce["code_verifier"])

    auth_url = get_google_auth_url(state=pkce["state"])
    return RedirectResponse(auth_url)


@router.get("/google/callback", summary="Google OAuth callback")
async def google_callback(
    code:  str,
    state: str,
    db:    AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Google calls this URL after the user approves.
    We exchange the code for tokens and save them encrypted.
    """
    # Verify state matches what we sent (CSRF protection)
    pkce_session = retrieve_pkce_session(state)
    if not pkce_session:
        raise HTTPException(status_code=400, detail="Invalid OAuth state. Please try again.")

    # Exchange code for tokens
    token_data = await exchange_google_code(code)

    # Save Gmail token
    await save_provider_token(db, str(current_user.id), "gmail", {
        "access_token":  token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_in":    token_data.get("expires_in"),
    })
    # Save Calendar token (same token works for both)
    await save_provider_token(db, str(current_user.id), "google_calendar", {
        "access_token":  token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_in":    token_data.get("expires_in"),
    })

    await create_audit_log(db, str(current_user.id), "oauth_connect", "Connected Google (Gmail + Calendar)")
    logger.info("Google OAuth connected for user %s", current_user.email)

    # Redirect back to the frontend settings page
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:8501")
    return RedirectResponse(f"{frontend_url}?connected=google")


@router.get("/connected-services", summary="List which services user has connected")
async def connected_services(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns which OAuth services the user has connected."""
    from backend.db.crud import get_connected_providers
    providers = await get_connected_providers(db, str(current_user.id))
    return {
        "gmail":            "gmail" in providers,
        "google_calendar":  "google_calendar" in providers,
        "slack":            "slack" in providers,
    }
