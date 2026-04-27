"""
Hybrid Retriever - Fixed for ChromaDB 1.x
Dense (OpenAI) + BM25 + RRF fusion
"""

import logging
from collections import defaultdict
from typing import Optional

from rank_bm25 import BM25Okapi

from backend.rag.embedder import get_embeddings
from backend.rag.ingest import _get_chroma_client, CHROMA_COLLECTION_PREFIX

logger = logging.getLogger(__name__)

DENSE_TOP_K = 20
BM25_TOP_K  = 20
RRF_K       = 60
FINAL_TOP_N = 10

_bm25_cache: dict[str, dict] = {}


def _build_bm25_index(user_id: str) -> None:
    client = _get_chroma_client()
    #collection_name = f"{CHROMA_COLLECTION_PREFIX}{user_id}"
    collection_name = "col" + "".join(c if c.isalnum() else "" for c in user_id)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        logger.warning("No ChromaDB collection for user %s", user_id)
        _bm25_cache[user_id] = {"index": None, "docs": [], "ids": [], "metas": []}
        return

    result = collection.get(include=["documents", "metadatas"])
    docs  = result.get("documents") or []
    ids   = result.get("ids")       or []
    metas = result.get("metadatas") or []

    if not docs:
        _bm25_cache[user_id] = {"index": None, "docs": [], "ids": [], "metas": []}
        return

    tokenised = [doc.lower().split() for doc in docs]
    _bm25_cache[user_id] = {
        "index": BM25Okapi(tokenised),
        "docs": docs, "ids": ids, "metas": metas,
    }
    logger.info("BM25: built index with %d docs for user %s", len(docs), user_id)


def invalidate_bm25_cache(user_id: str) -> None:
    _bm25_cache.pop(user_id, None)


def _get_bm25(user_id: str) -> dict:
    if user_id not in _bm25_cache:
        _build_bm25_index(user_id)
    return _bm25_cache[user_id]


def _dense_retrieve(query: str, user_id: str, top_k: int = DENSE_TOP_K,
                    doc_id_filter: Optional[str] = None) -> list[dict]:
    client = _get_chroma_client()
    collection_name = f"{CHROMA_COLLECTION_PREFIX}{user_id}"
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    query_embedding = get_embeddings(query)
    where = {"doc_id": doc_id_filter} if doc_id_filter else None

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for idx, (doc, meta, dist) in enumerate(zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    )):
        hits.append({
            "id": result["ids"][0][idx],
            "document": doc, "metadata": meta,
            "distance": dist,
            "dense_score": round(1.0 - dist, 6),
        })
    return hits


def _sparse_retrieve(query: str, user_id: str, top_k: int = BM25_TOP_K,
                     doc_id_filter: Optional[str] = None) -> list[dict]:
    cache = _get_bm25(user_id)
    if cache["index"] is None:
        return []

    scores = cache["index"].get_scores(query.lower().split())
    candidates = []
    for idx, score in enumerate(scores):
        meta = cache["metas"][idx]
        if doc_id_filter and meta.get("doc_id") != doc_id_filter:
            continue
        candidates.append({
            "id": cache["ids"][idx],
            "document": cache["docs"][idx],
            "metadata": meta,
            "bm25_score": round(float(score), 6),
        })
    candidates.sort(key=lambda x: x["bm25_score"], reverse=True)
    return candidates[:top_k]


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]],
                             id_key: str = "id", rrf_k: int = RRF_K) -> list[dict]:
    doc_registry: dict[str, dict] = {}
    rrf_scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc[id_key]
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)
            if doc_id not in doc_registry:
                doc_registry[doc_id] = doc
    fused = []
    for doc_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(doc_registry[doc_id])
        entry["rrf_score"] = round(rrf_score, 8)
        fused.append(entry)
    return fused


def retrieve(query: str, user_id: str, top_n: int = FINAL_TOP_N,
             doc_id_filter: Optional[str] = None) -> list[dict]:
    if not query.strip():
        return []
    dense_hits  = _dense_retrieve(query,  user_id, top_k=DENSE_TOP_K, doc_id_filter=doc_id_filter)
    sparse_hits = _sparse_retrieve(query, user_id, top_k=BM25_TOP_K,  doc_id_filter=doc_id_filter)
    if not dense_hits and not sparse_hits:
        return []
    fused = _reciprocal_rank_fusion([dense_hits, sparse_hits])
    dense_by_id  = {h["id"]: h for h in dense_hits}
    sparse_by_id = {h["id"]: h for h in sparse_hits}
    for doc in fused:
        doc["dense_score"] = dense_by_id.get(doc["id"],  {}).get("dense_score", 0.0)
        doc["bm25_score"]  = sparse_by_id.get(doc["id"], {}).get("bm25_score",  0.0)
    return fused[:top_n]


def retrieve_multi_query(queries: list[str], user_id: str,
                         top_n: int = FINAL_TOP_N,
                         doc_id_filter: Optional[str] = None) -> list[dict]:
    all_lists = [retrieve(q, user_id, DENSE_TOP_K, doc_id_filter) for q in queries]
    return _reciprocal_rank_fusion(all_lists)[:top_n]
