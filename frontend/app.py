"""
Banking RAG Chatbot — Streamlit Frontend
Main entry point. Run with: streamlit run frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Banking RAG Chatbot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "email" not in st.session_state:
    st.session_state.email = None
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Sidebar — navigation + login status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Banking Assistant")
    st.markdown("---")

    if st.session_state.token:
        st.success(f"✅ Logged in as\n{st.session_state.email}")
        st.markdown("---")
        st.markdown("**Navigate:**")
        st.page_link("pages/chat.py",  label="💬 Chat",          icon="💬")
        st.page_link("pages/files.py", label="📁 Upload Files",   icon="📁")
        st.page_link("pages/oauth.py", label="🔗 Connect Services", icon="🔗")
        st.page_link("pages/log.py",   label="📋 Audit Logs",     icon="📋")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.token    = None
            st.session_state.user_id  = None
            st.session_state.email    = None
            st.rerun()
    else:
        st.info("Please login to continue.")

# ---------------------------------------------------------------------------
# Main page — Login / Register
# ---------------------------------------------------------------------------
import os
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

if st.session_state.token:
    st.title("🏦 Welcome to Banking RAG Chatbot")
    st.markdown("""
    ### What can I help you with?
    | Feature | Description |
    |---|---|
    | 💬 **Chat** | Ask questions about your transactions, loans, and rates |
    | 📁 **Upload Files** | Upload your .xlsx or .csv banking data files |
    | 🔗 **Connect Services** | Connect Gmail, Google Calendar, Slack |
    | 📋 **Audit Logs** | View your activity history |

    👈 **Use the sidebar to navigate**
    """)
else:
    st.title("🏦 Banking RAG Chatbot")
    st.markdown("AI-powered banking assistant with RAG, LangGraph agents, and OAuth2 integrations.")
    st.markdown("---")

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    # ── Login tab ──
    with tab_login:
        st.subheader("Login to your account")
        with st.form("login_form"):
            email    = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter email and password.")
            else:
                with st.spinner("Logging in..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/api/auth/login",
                            data={"username": email, "password": password},
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.token   = data["access_token"]
                            st.session_state.user_id = data["user_id"]
                            st.session_state.email   = data["email"]
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            detail = resp.json().get("detail", "Login failed.")
                            st.error(f"❌ {detail}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend. Make sure Docker is running.")

    # ── Register tab ──
    with tab_register:
        st.subheader("Create a new account")
        with st.form("register_form"):
            reg_name  = st.text_input("Full Name")
            reg_email = st.text_input("Email", placeholder="you@example.com")
            reg_pass  = st.text_input("Password", type="password")
            reg_pass2 = st.text_input("Confirm Password", type="password")
            reg_submit = st.form_submit_button("Register", use_container_width=True)

        if reg_submit:
            if not reg_email or not reg_pass:
                st.error("Please fill in all fields.")
            elif reg_pass != reg_pass2:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Creating account..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/api/auth/register",
                            json={"email": reg_email, "password": reg_pass, "full_name": reg_name},
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            st.success("✅ Account created! Please login.")
                        else:
                            detail = resp.json().get("detail", "Registration failed.")
                            st.error(f"❌ {detail}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend.")
