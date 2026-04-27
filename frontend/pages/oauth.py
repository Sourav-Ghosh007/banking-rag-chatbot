"""
OAuth / Connect Services Page
==============================
Connect Gmail, Google Calendar, and Slack to enable
the communications agent to send emails, create events, post messages.
"""

import os
import requests
import streamlit as st

API_URL      = os.environ.get("API_URL",      "http://localhost:8000")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:8501")

st.set_page_config(page_title="Connect Services", page_icon="🔗", layout="wide")

if not st.session_state.get("token"):
    st.warning("⚠️ Please login first.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {st.session_state.token}"}

# ---------------------------------------------------------------------------
st.title("🔗 Connect External Services")
st.markdown("Connect your accounts so the assistant can send emails, manage your calendar, and post Slack messages.")
st.markdown("---")

# Check which services are already connected
connected = {"gmail": False, "google_calendar": False, "slack": False}
try:
    resp = requests.get(f"{API_URL}/api/auth/connected-services", headers=HEADERS, timeout=10)
    if resp.status_code == 200:
        connected = resp.json()
except Exception:
    st.warning("Could not check connection status.")

# ── Gmail & Google Calendar ──
st.subheader("📧 Google Services")
col1, col2 = st.columns(2)

with col1:
    status = "✅ Connected" if connected.get("gmail") else "❌ Not Connected"
    st.markdown(f"**Gmail** — {status}")
    st.caption("Allows the assistant to send emails on your behalf.")
    if not connected.get("gmail"):
        if st.button("Connect Gmail + Calendar", use_container_width=True, type="primary"):
            st.markdown(f'<meta http-equiv="refresh" content="0;url={API_URL}/api/auth/google">', unsafe_allow_html=True)
            st.info("Redirecting to Google login...")

with col2:
    status = "✅ Connected" if connected.get("google_calendar") else "❌ Not Connected"
    st.markdown(f"**Google Calendar** — {status}")
    st.caption("Allows the assistant to create events and reminders.")
    if connected.get("google_calendar"):
        st.success("Connected with Gmail above ✅")

st.markdown("---")

# ── Slack ──
st.subheader("💬 Slack")
col3, col4 = st.columns(2)

with col3:
    status = "✅ Connected" if connected.get("slack") else "❌ Not Connected"
    st.markdown(f"**Slack** — {status}")
    st.caption("Allows the assistant to post messages to your Slack workspace.")
    if not connected.get("slack"):
        if st.button("Connect Slack", use_container_width=True, type="primary"):
            st.markdown(f'<meta http-equiv="refresh" content="0;url={API_URL}/api/auth/slack">', unsafe_allow_html=True)
            st.info("Redirecting to Slack login...")

with col4:
    if connected.get("slack"):
        st.success("Slack is connected ✅")
    else:
        st.info("After connecting, you can ask:\n*'Post a message to #general saying loan approved'*")

st.markdown("---")
st.subheader("ℹ️ How OAuth works")
st.markdown("""
1. You click **Connect** → redirected to Google/Slack login page
2. You approve the permissions (we only ask for what we need)
3. Google/Slack sends us a secure token
4. We **encrypt** the token with Fernet encryption and store it in our database
5. The assistant uses this token when you ask it to send emails or post messages
6. **Your password is never shared with us** — this is how OAuth2 works

You can disconnect any service at any time by contacting support.
""")

# Show URL param feedback (e.g. after redirect back from Google)
query_params = st.query_params
if query_params.get("connected"):
    service = query_params.get("connected")
    st.success(f"✅ Successfully connected {service}!")
    st.query_params.clear()
