"""
Sidebar Component
=================
Shared sidebar shown on every page.
Shows login status, navigation, and connected services.
"""

import streamlit as st


def render_sidebar() -> None:
    """
    Render the sidebar navigation.
    Call this at the top of every page file.
    """
    with st.sidebar:
        st.title("Banking Assistant")
        st.markdown("---")

        if st.session_state.get("token"):
            # Logged-in state
            st.success(f"✅ **{st.session_state.get('email', 'User')}**")
            st.markdown("---")

            st.markdown("**📍 Navigation**")
            st.page_link("app.py",           label="🏠 Home")
            st.page_link("pages/chat.py",    label="💬 Chat")
            st.page_link("pages/files.py",   label="📁 Upload Files")
            st.page_link("pages/oauth.py",   label="🔗 Connect Services")
            st.page_link("pages/log.py",     label="📋 Audit Logs")

            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.token    = None
                st.session_state.user_id  = None
                st.session_state.email    = None
                st.rerun()
        else:
            st.info("Please login to use the chatbot.")
            st.page_link("app.py", label="🔑 Go to Login")
