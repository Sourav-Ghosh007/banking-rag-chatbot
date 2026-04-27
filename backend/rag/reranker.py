"""
Cross-Encoder Re-ranker + RAG Metrics
======================================
Stage 2 of the retrieval pipeline:
  retriever.py candidates → Cross-Encoder score → re-ranked top-K

Also exposes lightweight RAG quality metrics (MRR, precision@K, latency)
that are logged per query and stored for dashboard display.
"""

import time
import logging
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5            # final docs surfaced to the LLM / agent
RELEVANCE_THRESHOLD = -100.0   # cross-encoder score gate (logit scale, no sigmoid)

# ---------------------------------------------------------------------------
# Singleton cross-encoder
# ---------------------------------------------------------------------------
_cross_encoder: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading Cross-Encoder model: %s", CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    return _cross_encoder


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrievalMetrics:
    query: str
    user_id: str
    n_candidates: int                   # docs into the re-ranker
    n_final: int                        # docs returned after re-rank
    dense_top1_score: float = 0.0       # highest dense score in candidates
    bm25_top1_score: float = 0.0        # highest BM25 score in candidates
    rrf_top1_score: float = 0.0         # highest RRF score in candidates
    cross_encoder_top1: float = 0.0     # highest CE score after re-rank
    cross_encoder_mean: float = 0.0     # mean CE score of top-K
    mrr: float = 0.0                    # MRR if ground-truth provided
    precision_at_k: float = 0.0        # P@K if ground-truth provided
    latency_retrieval_ms: float = 0.0
    latency_rerank_ms: float = 0.0
    latency_total_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metrics store (in-memory ring buffer, per-user)
# ---------------------------------------------------------------------------
_MAX_STORED_METRICS = 500   # per user

_metrics_store: dict[str, list[dict]] = {}


def _store_metrics(user_id: str, m: RetrievalMetrics) -> None:
    if user_id not in _metrics_store:
        _metrics_store[user_id] = []
    store = _metrics_store[user_id]
    store.append(m.to_dict())
    if len(store) > _MAX_STORED_METRICS:
        store.pop(0)


def get_user_metrics(user_id: str, last_n: int = 50) -> list[dict]:
    """Return the last N metrics records for a user."""
    store = _metrics_store.get(user_id, [])
    return store[-last_n:]


def get_metrics_summary(user_id: str) -> dict:
    """Aggregate summary of stored metrics for dashboard display."""
    records = _metrics_store.get(user_id, [])
    if not records:
        return {"total_queries": 0}

    ce_scores = [r["cross_encoder_top1"] for r in records if r["cross_encoder_top1"]]
    latencies = [r["latency_total_ms"] for r in records if r["latency_total_ms"]]
    mrr_vals = [r["mrr"] for r in records if r["mrr"] > 0]
    p_at_k = [r["precision_at_k"] for r in records if r["precision_at_k"] > 0]

    def _mean(lst):
        return round(statistics.mean(lst), 4) if lst else None

    return {
        "total_queries": len(records),
        "avg_ce_score": _mean(ce_scores),
        "avg_latency_ms": _mean(latencies),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else None,
        "avg_mrr": _mean(mrr_vals),
        "avg_precision_at_k": _mean(p_at_k),
        "n_with_ground_truth": len(mrr_vals),
    }


# ---------------------------------------------------------------------------
# MRR / Precision@K helpers (optional, requires ground truth)
# ---------------------------------------------------------------------------
def _compute_mrr(reranked: list[dict], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank for a single query."""
    for rank, doc in enumerate(reranked, start=1):
        if doc.get("id") in relevant_ids or doc.get("metadata", {}).get("doc_id") in relevant_ids:
            return 1.0 / rank
    return 0.0


def _compute_precision_at_k(reranked: list[dict], relevant_ids: set[str], k: int) -> float:
    """Precision@K for a single query."""
    top_k = reranked[:k]
    hits = sum(
        1 for doc in top_k
        if doc.get("id") in relevant_ids or doc.get("metadata", {}).get("doc_id") in relevant_ids
    )
    return hits / k if k > 0 else 0.0


# ---------------------------------------------------------------------------
# Public re-ranker API
# ---------------------------------------------------------------------------
def rerank(
    query: str,
    candidates: list[dict],
    user_id: str,
    top_k: int = RERANK_TOP_K,
    relevant_ids: Optional[set[str]] = None,   # for offline eval only
    retrieval_latency_ms: float = 0.0,
) -> tuple[list[dict], RetrievalMetrics]:
    """
    Cross-Encoder re-ranking.

    Parameters
    ----------
    query               : original user query
    candidates          : output of retriever.retrieve()
    user_id             : for metrics attribution
    top_k               : how many to return after re-ranking
    relevant_ids        : set of ground-truth doc_ids (optional, for MRR/P@K)
    retrieval_latency_ms: pass through the retrieval stage timing

    Returns
    -------
    (reranked_docs, metrics)
    reranked_docs: list of dicts enriched with 'cross_encoder_score'
    metrics      : RetrievalMetrics instance
    """
    t_start = time.perf_counter()

    metrics = RetrievalMetrics(
        query=query,
        user_id=user_id,
        n_candidates=len(candidates),
        n_final=0,
        latency_retrieval_ms=retrieval_latency_ms,
    )

    if not candidates:
        logger.info("Reranker: empty candidate set for user %s", user_id)
        _store_metrics(user_id, metrics)
        return [], metrics

    # Record incoming score peaks
    metrics.dense_top1_score = max((c.get("dense_score", 0.0) for c in candidates), default=0.0)
    metrics.bm25_top1_score = max((c.get("bm25_score", 0.0) for c in candidates), default=0.0)
    metrics.rrf_top1_score = max((c.get("rrf_score", 0.0) for c in candidates), default=0.0)

    # Build (query, passage) pairs for cross-encoder
    model = _get_cross_encoder()
    pairs = [(query, doc["document"]) for doc in candidates]

    ce_scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()

    # Attach CE score to each candidate
    scored = []
    for doc, score in zip(candidates, ce_scores):
        enriched = dict(doc)
        enriched["cross_encoder_score"] = round(float(score), 6)
        scored.append(enriched)

    # Sort by CE score descending
    scored.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    # Apply threshold gate
    scored = [d for d in scored if d["cross_encoder_score"] >= RELEVANCE_THRESHOLD]

    # Return top-K
    reranked = scored[:top_k]

    t_end = time.perf_counter()
    rerank_ms = (t_end - t_start) * 1000

    # Populate metrics
    ce_top_scores = [d["cross_encoder_score"] for d in reranked]
    metrics.n_final = len(reranked)
    metrics.cross_encoder_top1 = ce_top_scores[0] if ce_top_scores else 0.0
    metrics.cross_encoder_mean = round(statistics.mean(ce_top_scores), 6) if ce_top_scores else 0.0
    metrics.latency_rerank_ms = round(rerank_ms, 2)
    metrics.latency_total_ms = round(retrieval_latency_ms + rerank_ms, 2)

    if relevant_ids:
        metrics.mrr = _compute_mrr(reranked, relevant_ids)
        metrics.precision_at_k = _compute_precision_at_k(reranked, relevant_ids, top_k)

    _store_metrics(user_id, metrics)

    logger.info(
        "Reranker: %d → %d docs | CE top1=%.4f | total_latency=%.1f ms | user=%s",
        len(candidates),
        len(reranked),
        metrics.cross_encoder_top1,
        metrics.latency_total_ms,
        user_id,
    )

    return reranked, metrics


# ---------------------------------------------------------------------------
# Full pipeline convenience wrapper
# ---------------------------------------------------------------------------
def retrieve_and_rerank(
    query: str,
    user_id: str,
    top_n_retrieve: int = 20,
    top_k_rerank: int = RERANK_TOP_K,
    doc_id_filter: Optional[str] = None,
    relevant_ids: Optional[set[str]] = None,
) -> tuple[list[dict], RetrievalMetrics]:
    """
    One-call pipeline: hybrid retrieval → cross-encoder re-rank.

    Returns (reranked_docs, metrics).
    """
    # Import here to avoid circular dependency
    from backend.rag.retriever import retrieve

    t0 = time.perf_counter()
    candidates = retrieve(
        query=query,
        user_id=user_id,
        top_n=top_n_retrieve,
        doc_id_filter=doc_id_filter,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    return rerank(
        query=query,
        candidates=candidates,
        user_id=user_id,
        top_k=top_k_rerank,
        relevant_ids=relevant_ids,
        retrieval_latency_ms=retrieval_ms,
    )
