"""
RAG Metrics Module - SQLite compatible version
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Optional, Any

logger = logging.getLogger(__name__)

METRIC_QUERY  = "query"
METRIC_INGEST = "ingest"
METRIC_RERANK = "rerank"
METRIC_ERROR  = "error"

MAX_PER_USER = 1000
_events:   dict[str, list[dict]] = defaultdict(list)
_counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))


def log_event(user_id: str, event_type: str, data: dict[str, Any], *, flush_to_db: bool = False) -> None:
    event = {"user_id": user_id, "event_type": event_type, "timestamp": time.time(), **data}
    store = _events[user_id]
    store.append(event)
    if len(store) > MAX_PER_USER:
        store.pop(0)
    _update_counters(user_id, event_type, data)
    logger.debug("metrics[%s] %s %s", user_id, event_type, data)
    if flush_to_db:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_flush_event_to_db(event))
        except RuntimeError:
            pass


def _update_counters(user_id: str, event_type: str, data: dict) -> None:
    c = _counters[user_id]
    c["total_events"] += 1
    if event_type == METRIC_QUERY:
        c["total_queries"]  += 1
        c["total_latency_ms"] += data.get("latency_total_ms", 0)
        c["total_ce_score"]   += data.get("cross_encoder_top1", 0)
    elif event_type == METRIC_INGEST:
        c["total_ingestions"]      += 1
        c["total_chunks_ingested"] += data.get("chunks_ingested", 0)
        c["total_rows_ingested"]   += data.get("total_rows", 0)
    elif event_type == METRIC_ERROR:
        c["total_errors"] += 1


def log_query(
    user_id: str, query: str,
    n_candidates: int, n_final: int,
    latency_retrieval_ms: float, latency_rerank_ms: float, latency_total_ms: float,
    cross_encoder_top1: float, cross_encoder_mean: float,
    dense_top1: float = 0.0, bm25_top1: float = 0.0, rrf_top1: float = 0.0,
    mrr: float = 0.0, precision_at_k: float = 0.0,
    agent_name: Optional[str] = None, extra: Optional[dict] = None,
) -> None:
    payload = {
        "query": query[:200],
        "n_candidates": n_candidates, "n_final": n_final,
        "latency_retrieval_ms": round(latency_retrieval_ms, 2),
        "latency_rerank_ms":    round(latency_rerank_ms, 2),
        "latency_total_ms":     round(latency_total_ms, 2),
        "cross_encoder_top1":   round(cross_encoder_top1, 4),
        "cross_encoder_mean":   round(cross_encoder_mean, 4),
        "dense_top1_score":     round(dense_top1, 4),
        "bm25_top1_score":      round(bm25_top1, 4),
        "rrf_top1_score":       round(rrf_top1, 4),
        "mrr": round(mrr, 4), "precision_at_k": round(precision_at_k, 4),
        "agent_name": agent_name,
    }
    if extra:
        payload.update(extra)
    log_event(user_id, METRIC_QUERY, payload, flush_to_db=True)


def log_ingest(user_id: str, filename: str, doc_id: str, sheets: list,
               total_rows: int, chunks_ingested: int, latency_ms: float) -> None:
    log_event(user_id, METRIC_INGEST, {
        "filename": filename, "doc_id": doc_id, "sheets": sheets,
        "total_rows": total_rows, "chunks_ingested": chunks_ingested,
        "latency_ms": round(latency_ms, 2),
    }, flush_to_db=True)


def log_error(user_id: str, error_type: str, detail: str, context: Optional[dict] = None) -> None:
    payload = {"error_type": error_type, "detail": detail[:500]}
    if context:
        payload["context"] = context
    log_event(user_id, METRIC_ERROR, payload)


def get_user_summary(user_id: str) -> dict:
    c = _counters.get(user_id, {})
    total_queries  = c.get("total_queries", 0)
    total_latency  = c.get("total_latency_ms", 0)
    total_ce       = c.get("total_ce_score", 0)
    return {
        "user_id":               user_id,
        "total_queries":         int(total_queries),
        "total_ingestions":      int(c.get("total_ingestions", 0)),
        "total_chunks_ingested": int(c.get("total_chunks_ingested", 0)),
        "total_rows_ingested":   int(c.get("total_rows_ingested", 0)),
        "total_errors":          int(c.get("total_errors", 0)),
        "avg_latency_ms":        round(total_latency / total_queries, 2) if total_queries else None,
        "avg_ce_top1":           round(total_ce / total_queries, 4)      if total_queries else None,
    }


def get_recent_events(user_id: str, n: int = 20, event_type: Optional[str] = None) -> list[dict]:
    events = _events.get(user_id, [])
    if event_type:
        events = [e for e in events if e["event_type"] == event_type]
    return events[-n:]


def export_events_json(user_id: str, event_type: Optional[str] = None) -> str:
    events = get_recent_events(user_id, n=MAX_PER_USER, event_type=event_type)
    return json.dumps(events, indent=2, default=str)


# ---------------------------------------------------------------------------
# SQLite-compatible table creation
# ---------------------------------------------------------------------------
async def ensure_metrics_table() -> None:
    """Create rag_metrics table — SQLite compatible."""
    try:
        from backend.db.session import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS rag_metrics (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT    NOT NULL,
                    event_type TEXT    NOT NULL,
                    payload    TEXT    NOT NULL DEFAULT '{}',
                    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
                )
            """))
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_rag_metrics_user
                ON rag_metrics (user_id, created_at)
            """))
            await session.commit()
        logger.info("rag_metrics table ready (SQLite)")
    except Exception as exc:
        logger.error("Could not create rag_metrics table: %s", exc)


async def _flush_event_to_db(event: dict) -> None:
    """Save metric event to SQLite."""
    try:
        from backend.db.session import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(
                text("INSERT INTO rag_metrics (user_id, event_type, payload) VALUES (:uid, :etype, :payload)"),
                {"uid": event["user_id"], "etype": event["event_type"], "payload": json.dumps(event, default=str)},
            )
            await session.commit()
    except Exception as exc:
        logger.debug("Metrics flush failed (non-critical): %s", exc)
