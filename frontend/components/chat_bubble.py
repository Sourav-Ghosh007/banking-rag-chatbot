"""
Chat Bubble Component
=====================
Reusable component for displaying chat messages with metadata.
Used by frontend/pages/chat.py
"""

import streamlit as st


def render_user_message(message: str) -> None:
    """Render a user message bubble."""
    with st.chat_message("user"):
        st.markdown(message)


def render_assistant_message(
    message:      str,
    intent:       str | None = None,
    target_agent: str | None = None,
    latency_ms:   float = 0.0,
    sources:      list  = None,
) -> None:
    """
    Render an assistant message bubble with metadata badges.

    Parameters
    ----------
    message      : the answer text
    intent       : classified intent (loans, transactions, etc.)
    target_agent : which agent answered
    latency_ms   : response time
    sources      : list of source documents used
    """
    with st.chat_message("assistant"):
        st.markdown(message)

        # Show metadata in small colored badges
        if intent or target_agent or latency_ms:
            cols = st.columns(3)
            if intent:
                cols[0].caption(f"🎯 **Intent:** `{intent}`")
            if target_agent:
                cols[1].caption(f"🤖 **Agent:** `{target_agent}`")
            if latency_ms:
                color = "🟢" if latency_ms < 3000 else "🟡" if latency_ms < 8000 else "🔴"
                cols[2].caption(f"{color} **Time:** `{latency_ms:.0f}ms`")

        # Show sources in an expander if present
        if sources:
            with st.expander(f"📚 {len(sources)} source(s) used"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**{i}. {src.get('filename', 'Unknown')}**")
                    st.caption(f"Sheet: `{src.get('sheet', '')}` | Score: `{src.get('score', 0):.4f}`")
                    st.markdown(f"*{src.get('snippet', '')[:150]}...*")
                    if i < len(sources):
                        st.divider()


def render_error_message(error: str) -> None:
    """Render an error message in the chat."""
    with st.chat_message("assistant"):
        st.error(f"⚠️ {error}")


def render_thinking_placeholder():
    """Show a 'thinking' spinner while waiting for the response."""
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            import time
            time.sleep(0.1)   # placeholder — actual call happens outside


def render_metric_row(label: str, value: str, delta: str = None) -> None:
    """Render a single metric display."""
    st.metric(label=label, value=value, delta=delta)
