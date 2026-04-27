"""
Files Page
==========
Upload .xlsx and .csv files.
Shows upload status, ingestion stats, and list of uploaded files.
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Upload Files", page_icon="📁", layout="wide")

if not st.session_state.get("token"):
    st.warning("⚠️ Please login first.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {st.session_state.token}"}

# ---------------------------------------------------------------------------
st.title("📁 Upload Data Files")
st.markdown("Upload your banking data files. Only **.xlsx** and **.csv** are accepted. PDF is not supported.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

# ── Upload section ──
with col1:
    st.subheader("📤 Upload a File")

    uploaded = st.file_uploader(
        "Choose a file",
        type=["xlsx", "csv"],
        help="Only .xlsx and .csv files accepted. Max size: 10MB",
    )

    if uploaded:
        st.info(f"📄 Selected: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

        if st.button("🚀 Upload and Process", use_container_width=True, type="primary"):
            with st.spinner(f"Processing {uploaded.name}... This may take a moment."):
                try:
                    resp = requests.post(
                        f"{API_URL}/api/files/upload",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        headers=HEADERS,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("✅ File processed successfully!")
                        st.markdown("### 📊 Ingestion Summary")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Rows Ingested",   data.get("total_rows", 0))
                        col_b.metric("Chunks in ChromaDB", data.get("chunks_ingested", 0))
                        col_c.metric("Processing Time", f"{data.get('latency_ms', 0):.0f} ms")
                        st.caption(f"📋 Sheets: {', '.join(data.get('sheets', []))}")
                        st.caption(f"🆔 Doc ID: `{data.get('doc_id', '')}`")
                        st.rerun()
                    elif resp.status_code == 400:
                        st.error(f"❌ {resp.json().get('detail', 'Upload failed.')}")
                    else:
                        st.error("❌ Upload failed. Please try again.")
                except requests.exceptions.Timeout:
                    st.error("❌ Upload timed out. Try a smaller file.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend.")

    st.markdown("---")
    st.markdown("### ℹ️ What happens when you upload?")
    st.markdown("""
    1. **Validation** — checks it's .xlsx or .csv (PDF rejected)
    2. **Parsing** — reads all rows and sheets
    3. **ChromaDB** — embeds rows using OpenAI and stores for semantic search
    4. **SQLite** — loads data for fast SQL aggregations (totals, averages)
    5. **Ready** — you can now ask questions about this data in Chat
    """)

# ── Uploaded files list ──
with col2:
    st.subheader("📂 Your Uploaded Files")

    try:
        resp = requests.get(f"{API_URL}/api/files/", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("files", [])

            if not files:
                st.info("No files uploaded yet. Upload your first file on the left!")
            else:
                st.caption(f"Total: {data.get('count', 0)} file(s)")
                for f in files:
                    with st.expander(f"📄 {f['filename']}"):
                        st.caption(f"Doc ID: `{f['doc_id']}`")
                        st.caption(f"Sheets: {', '.join(f.get('sheets', []))}")
                        if st.button(
                            "🗑️ Delete",
                            key=f"del_{f['doc_id']}",
                            use_container_width=True,
                        ):
                            del_resp = requests.delete(
                                f"{API_URL}/api/files/{f['doc_id']}",
                                headers=HEADERS,
                                timeout=15,
                            )
                            if del_resp.status_code == 200:
                                st.success("Deleted!")
                                st.rerun()
                            else:
                                st.error("Delete failed.")
        else:
            st.error("Could not load file list.")
    except Exception:
        st.error("Cannot connect to backend.")
