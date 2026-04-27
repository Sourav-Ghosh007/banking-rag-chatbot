"""
API Helper
==========
All frontend → backend API calls in one place.
Every page imports from here instead of writing requests.post() everywhere.
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 60  # seconds


def get_headers() -> dict:
    """Return auth headers using the stored JWT token."""
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"}


def login(email: str, password: str) -> dict:
    """Login and return {access_token, user_id, email} or raise on failure."""
    resp = requests.post(
        f"{API_URL}/api/auth/login",
        data={"username": email, "password": password},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def register(email: str, password: str, full_name: str = "") -> dict:
    """Register a new user."""
    resp = requests.post(
        f"{API_URL}/api/auth/register",
        json={"email": email, "password": password, "full_name": full_name},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_chat(message: str, session_id: str) -> dict:
    """Send a chat message and return the agent response."""
    resp = requests.post(
        f"{API_URL}/api/chat",
        json={"message": message, "session_id": session_id},
        headers=get_headers(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def upload_file(filename: str, file_bytes: bytes, mime_type: str) -> dict:
    """Upload a file and return ingestion stats."""
    resp = requests.post(
        f"{API_URL}/api/files/upload",
        files={"file": (filename, file_bytes, mime_type)},
        headers=get_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def list_files() -> list:
    """Return list of uploaded files for the current user."""
    resp = requests.get(f"{API_URL}/api/files/", headers=get_headers(), timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("files", [])


def delete_file(doc_id: str) -> dict:
    """Delete an uploaded file by doc_id."""
    resp = requests.delete(f"{API_URL}/api/files/{doc_id}", headers=get_headers(), timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_connected_services() -> dict:
    """Return which OAuth services are connected."""
    resp = requests.get(f"{API_URL}/api/auth/connected-services", headers=get_headers(), timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_chat_history(limit: int = 20) -> list:
    """Return recent chat history from audit logs."""
    resp = requests.get(
        f"{API_URL}/api/chat/history",
        params={"limit": limit},
        headers=get_headers(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("history", [])
