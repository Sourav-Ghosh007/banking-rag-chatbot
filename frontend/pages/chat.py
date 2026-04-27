"""
Chat Page
=========
The main chatbot interface.
User types a question → backend agents process it → answer shown with sources.
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")

# Guard: must be logged in
if not st.session_state.get("token"):
    st.warning("⚠️ Please login first.")
    st.stop()

# ---------------------------------------------------------------------------
# Helper: call the chat API
# ---------------------------------------------------------------------------
def send_message(message: str) -> dict:
    try:
        resp = requests.post(
            f"{API_URL}/api/chat",
            json={
                "message":    message,
                "session_id": st.session_state.get("session_id"),
            },
            headers={"Authorization": f"Bearer {st.session_state.token}"},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            st.error("Session expired. Please login again.")
            st.session_state.token = None
            st.stop()
        else:
            return {"answer": f"Error: {resp.json().get('detail', 'Unknown error')}", "sources": []}
    except requests.exceptions.Timeout:
        return {"answer": "Request timed out. The agent is taking too long. Please try again.", "sources": []}
    except requests.exceptions.ConnectionError:
        return {"answer": "Cannot connect to backend. Make sure Docker is running.", "sources": []}

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.title("💬 Banking Assistant Chat")

col1, col2 = st.columns([2, 1])

with col1:
    # ── Chat history ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display conversation
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Hi! Ask me about your transactions, loans, interest rates, or I can help you send emails and schedule meetings.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show metadata for assistant messages
                if msg["role"] == "assistant" and msg.get("meta"):
                    meta = msg["meta"]
                    cols = st.columns(3)
                    cols[0].caption(f"🎯 Intent: `{meta.get('intent', 'N/A')}`")
                    cols[1].caption(f"🤖 Agent: `{meta.get('target_agent', 'N/A')}`")
                    cols[2].caption(f"⏱️ {meta.get('latency_ms', 0):.0f} ms")

    # ── Input box ──
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            placeholder="e.g. What is my total spending this month? / Calculate EMI for ₹5L loan at 8.5% for 5 years",
            label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([4, 1])
        send   = col_send.form_submit_button("Send ➤", use_container_width=True)
        cleared = col_clear.form_submit_button("Clear", use_container_width=True)

    if cleared:
        st.session_state.chat_history = []
        st.rerun()

    if send and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Call API
        with st.spinner("🤔 Thinking..."):
            result = send_message(user_input)

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": result.get("answer", ""),
            "meta": {
                "intent":       result.get("intent"),
                "target_agent": result.get("target_agent"),
                "latency_ms":   result.get("latency_ms", 0),
            },
            "sources": result.get("sources", []),
        })
        st.rerun()

with col2:
    st.subheader("📚 Sources")

    # Show sources from latest assistant message
    latest_sources = []
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "assistant" and msg.get("sources"):
            latest_sources = msg["sources"]
            break

    if latest_sources:
        for i, src in enumerate(latest_sources, 1):
            with st.expander(f"Source {i}: {src.get('filename', 'Unknown')}"):
                st.caption(f"Sheet: `{src.get('sheet', '')}`")
                st.caption(f"Score: `{src.get('score', 0):.4f}`")
                st.markdown(f"*{src.get('snippet', '')}*")
    else:
        st.info("Sources will appear here after you ask a question.")

    st.markdown("---")
    st.subheader("💡 Example Questions")
    examples = [
        "What is my total spending this month?",
        "Calculate EMI for ₹5 lakh loan at 8.5% for 5 years",
        "What are the current FD interest rates?",
        "Show me all transactions above ₹10,000",
        "What is the USD to INR exchange rate?",
        "Send an email to my manager about loan approval",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state.chat_history.append({"role": "user", "content": ex})
            with st.spinner("🤔 Thinking..."):
                result = send_message(ex)
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": result.get("answer", ""),
                "meta": {
                    "intent":       result.get("intent"),
                    "target_agent": result.get("target_agent"),
                    "latency_ms":   result.get("latency_ms", 0),
                },
                "sources": result.get("sources", []),
            })
            st.rerun()
