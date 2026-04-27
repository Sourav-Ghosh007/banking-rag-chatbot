"""
Audit Log & Metrics Page
=========================
Shows the user's activity history and RAG performance metrics.
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Audit Logs", page_icon="📋", layout="wide")

if not st.session_state.get("token"):
    st.warning("⚠️ Please login first.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {st.session_state.token}"}

# ---------------------------------------------------------------------------
st.title("📋 Audit Logs & Metrics")
st.markdown("Track your activity and see how the AI performed on each query.")
st.markdown("---")

tab1, tab2 = st.tabs(["📋 Activity Log", "📊 RAG Metrics"])

# ── Tab 1: Activity Log ──
with tab1:
    st.subheader("Your Recent Activity")

    try:
        resp = requests.get(
            f"{API_URL}/api/chat/history",
            params={"limit": 50},
            headers=HEADERS,
            timeout=10,
        )
        if resp.status_code == 200:
            data    = resp.json()
            history = data.get("history", [])

            if not history:
                st.info("No activity yet. Start by uploading a file and asking questions!")
            else:
                st.caption(f"Showing {len(history)} recent queries")
                for item in history:
                    details = item.get("details", "")
                    ts      = item.get("created_at", "")[:19].replace("T", " ")

                    # Parse details string into readable parts
                    parts = dict(p.split("=", 1) for p in details.split(" | ") if "=" in p)

                    with st.container():
                        cols = st.columns([2, 1, 1, 1])
                        cols[0].markdown(f"🔹 **{item.get('action','').replace('_',' ').title()}**")
                        cols[1].caption(f"Intent: `{parts.get('intent','N/A')}`")
                        cols[2].caption(f"Agent: `{parts.get('agent','N/A')}`")
                        cols[3].caption(f"⏱️ {parts.get('latency','N/A')}")
                        st.caption(f"🕐 {ts}")
                        st.divider()
        else:
            st.error("Could not load history.")
    except Exception as e:
        st.error(f"Cannot connect to backend: {e}")

# ── Tab 2: RAG Metrics ──
with tab2:
    st.subheader("RAG Performance Metrics")
    st.caption("These metrics show how well the retrieval pipeline performed on your queries.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 What each metric means")
        st.markdown("""
        | Metric | What it means |
        |---|---|
        | **Cross-Encoder Score** | How relevant the retrieved docs were (higher = better) |
        | **Retrieval Latency** | Time to find relevant documents |
        | **Rerank Latency** | Time for cross-encoder to rerank |
        | **Total Latency** | End-to-end response time |
        | **MRR** | Mean Reciprocal Rank — ranking quality |
        | **Precision@K** | How many top-K results were relevant |
        """)

    with col2:
        st.markdown("### 📐 Architecture Summary")
        st.markdown("""
        ```
        User Query
             │
        Orchestrator (GPT-4o classifies intent)
             │
        ┌────┴─────────────────┐
        │                      │
        Dense Search        BM25 Search
        (ChromaDB +         (keyword
        OpenAI embed)       matching)
        │                      │
        └────────┬─────────────┘
                 │
           RRF Fusion
           (combines both)
                 │
        Cross-Encoder Reranker
        (ms-marco-MiniLM)
                 │
          GPT-4o Answer
        ```
        """)

    st.markdown("---")
    st.info("💡 Detailed per-query metrics are logged to PostgreSQL. Check the `rag_metrics` table for full data.")
