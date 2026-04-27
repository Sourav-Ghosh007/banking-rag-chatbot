"""
RAG Ingestion Pipeline - Fixed table naming
"""

import io
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import chromadb

from backend.rag.embedder import get_batch_embeddings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".xlsx", ".csv"}
CHROMA_COLLECTION_PREFIX = "user_docs_"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

_chroma_client: Optional[chromadb.Client] = None
_sqlite_connections: dict[str, sqlite3.Connection] = {}


def _get_chroma_client() -> chromadb.Client:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_store")
    return _chroma_client


def _get_sqlite_conn(user_id: str) -> sqlite3.Connection:
    if user_id not in _sqlite_connections:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _sqlite_connections[user_id] = conn
        logger.info("Created SQLite in-memory DB for user %s", user_id)
    return _sqlite_connections[user_id]


def validate_file_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type '{ext}' is not supported. "
            f"Only {sorted(ALLOWED_EXTENSIONS)} files are accepted. "
            "PDF and other formats are explicitly rejected."
        )


def _make_table_name(filename: str, sheet_name: str) -> str:
    """
    Create a simple, clean table name from filename + sheet.
    Always starts with 't_' so it never starts with a number.
    Example: transactions_csv_transactions
    """
    stem = Path(filename).stem.lower()
    sheet = sheet_name.lower()
    # Remove all non-alphanumeric characters
    stem  = "".join(c if c.isalnum() else "_" for c in stem)
    sheet = "".join(c if c.isalnum() else "_" for c in sheet)
    # Always prefix with t_ so it never starts with a digit
    return f"t_{stem}_{sheet}"


def _parse_file(file_bytes: bytes, filename: str) -> dict[str, pd.DataFrame]:
    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(file_bytes)
    if ext == ".csv":
        df = pd.read_csv(buf, dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return {Path(filename).stem: df}
    xl = pd.ExcelFile(buf)
    sheets = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        sheets[sheet] = df
    return sheets


def _row_to_text(row: pd.Series, sheet_name: str) -> str:
    parts = [f"[{sheet_name}]"]
    for col, val in row.items():
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}: {str(val).strip()}")
    return " | ".join(parts)


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def _ingest_to_chroma(sheets: dict, user_id: str, doc_id: str, filename: str) -> int:
    client = _get_chroma_client()
    collection_name = f"{CHROMA_COLLECTION_PREFIX}{user_id}"
    # ChromaDB collection names must be alphanumeric + hyphens
    #collection_name = "".join(c if c.isalnum() or c == "-" else "-" for c in collection_name)
    collection_name = "col" + "".join(c if c.isalnum() else "" for c in user_id)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    all_texts, all_ids, all_metas = [], [], []
    for sheet_name, df in sheets.items():
        for row_idx, row in df.iterrows():
            row_text = _row_to_text(row, sheet_name)
            for chunk_idx, chunk in enumerate(_chunk_text(row_text)):
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                chunk_id = f"doc_{chunk_hash}_{row_idx}_{chunk_idx}"
                all_texts.append(chunk)
                all_ids.append(chunk_id)
                all_metas.append({
                    "doc_id": doc_id, "filename": filename,
                    "sheet": sheet_name, "row": int(row_idx),
                    "chunk": chunk_idx, "user_id": user_id,
                })

    if not all_texts:
        return 0

    BATCH_SIZE = 500
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        embeddings  = get_batch_embeddings(batch_texts)
        collection.upsert(
            ids=all_ids[i:i + BATCH_SIZE],
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=all_metas[i:i + BATCH_SIZE],
        )

    logger.info("ChromaDB: upserted %d chunks for user %s", len(all_texts), user_id)
    return len(all_texts)


def _ingest_to_sqlite(sheets: dict, user_id: str, doc_id: str, filename: str) -> None:
    conn = _get_sqlite_conn(user_id)

    for sheet_name, df in sheets.items():
        # Simple clean table name — always starts with t_
        table_name = _make_table_name(filename, sheet_name)

        df = df.copy()
        df["_doc_id"]   = doc_id
        df["_filename"] = filename
        df["_sheet"]    = sheet_name

        for col in df.columns:
            if col.startswith("_"):
                continue
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info("SQLite: wrote table '%s' (%d rows) for user %s", table_name, len(df), user_id)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS _doc_registry (
            doc_id TEXT, filename TEXT, sheet TEXT, table_name TEXT,
            PRIMARY KEY (doc_id, sheet)
        )
    """)
    for sheet_name in sheets:
        table_name = _make_table_name(filename, sheet_name)
        conn.execute(
            "INSERT OR REPLACE INTO _doc_registry VALUES (?, ?, ?, ?)",
            (doc_id, filename, sheet_name, table_name),
        )
    conn.commit()


def ingest_file(file_bytes: bytes, filename: str, user_id: str,
                doc_id: Optional[str] = None) -> dict:
    validate_file_extension(filename)
    if doc_id is None:
        content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        doc_id = f"doc_{Path(filename).stem}_{content_hash}"

    logger.info("Ingesting '%s' → doc_id=%s for user %s", filename, doc_id, user_id)
    sheets    = _parse_file(file_bytes, filename)
    total_rows = sum(len(df) for df in sheets.values())
    chunks    = _ingest_to_chroma(sheets, user_id, doc_id, filename)
    _ingest_to_sqlite(sheets, user_id, doc_id, filename)
    return {
        "doc_id": doc_id, "filename": filename,
        "sheets": list(sheets.keys()),
        "total_rows": total_rows, "chunks_ingested": chunks, "status": "success",
    }


def delete_user_data(user_id: str) -> None:
    client = _get_chroma_client()
    try:
        client.delete_collection(f"{CHROMA_COLLECTION_PREFIX}{user_id}")
    except Exception:
        pass
    if user_id in _sqlite_connections:
        _sqlite_connections[user_id].close()
        del _sqlite_connections[user_id]


def list_user_documents(user_id: str) -> list[dict]:
    conn = _get_sqlite_conn(user_id)
    try:
        rows = conn.execute(
            "SELECT DISTINCT doc_id, filename, sheet, table_name FROM _doc_registry"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
