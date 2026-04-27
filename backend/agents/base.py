"""
Agent Base Layer
================
Shared state schema, LLM client, tool-call helpers, and base class
used by all 5 agents. Import from here — never instantiate OpenAI directly
in agent files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from openai import AsyncOpenAI
from langgraph.graph import MessagesState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client singleton
# ---------------------------------------------------------------------------
_openai_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


GPT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = 0.0       # deterministic for banking
DEFAULT_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------
class AgentState(MessagesState):
    """
    Shared state that flows through the entire LangGraph graph.
    All agents read from and write back to this dict-like object.
    """
    # Identity
    user_id: str
    session_id: str

    # Routing
    intent: Optional[str]           # classified intent from router
    target_agent: Optional[str]     # which agent was dispatched to
    confidence: float               # router confidence score [0-1]

    # RAG context
    retrieved_docs: list[dict]      # raw retriever output
    reranked_docs: list[dict]       # after cross-encoder
    sql_results: list[dict]         # from sql_engine (aggregations)

    # Response assembly
    agent_response: Optional[str]   # agent's final text answer
    sources: list[dict]             # citations / source refs for UI
    error: Optional[str]            # set if any agent faults

    # Metrics pass-through
    latency_ms: float
    retrieval_metrics: dict


def make_initial_state(
    user_id: str,
    session_id: str,
    user_message: str,
) -> dict:
    """Construct a fresh AgentState for a new query."""
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=user_message)],
        "user_id": user_id,
        "session_id": session_id,
        "intent": None,
        "target_agent": None,
        "confidence": 0.0,
        "retrieved_docs": [],
        "reranked_docs": [],
        "sql_results": [],
        "agent_response": None,
        "sources": [],
        "error": None,
        "latency_ms": 0.0,
        "retrieval_metrics": {},
    }


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------
async def llm_chat(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    tool_choice: str = "auto",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    response_format: Optional[dict] = None,
) -> dict:
    """
    Thin async wrapper around the OpenAI chat completions endpoint.

    Returns the full response object as a dict for uniform handling.
    Raises on API errors (let the agent handle retry logic).
    """
    client = get_openai_client()
    kwargs: dict[str, Any] = dict(
        model=GPT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if response_format:
        kwargs["response_format"] = response_format

    t0 = time.perf_counter()
    response = await client.chat.completions.create(**kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug("LLM call: %.0f ms | model=%s | tokens=%s", elapsed, GPT_MODEL,
                 response.usage.total_tokens if response.usage else "?")
    return response


def extract_text(response) -> str:
    """Pull plain text from an OpenAI response object."""
    return response.choices[0].message.content or ""


def extract_tool_calls(response) -> list[dict]:
    """Return list of {name, arguments} for any tool_calls in the response."""
    msg = response.choices[0].message
    if not msg.tool_calls:
        return []
    result = []
    for tc in msg.tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        result.append({"id": tc.id, "name": tc.function.name, "arguments": args})
    return result


# ---------------------------------------------------------------------------
# Source builder (for UI citation panel)
# ---------------------------------------------------------------------------
def build_sources(reranked_docs: list[dict]) -> list[dict]:
    """Convert re-ranked doc metadata into UI-friendly source citations."""
    sources = []
    seen = set()
    for doc in reranked_docs:
        meta = doc.get("metadata", {})
        key = (meta.get("filename", ""), meta.get("sheet", ""), meta.get("row", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "filename": meta.get("filename", "Unknown"),
                "sheet": meta.get("sheet", ""),
                "row": meta.get("row"),
                "snippet": doc.get("document", "")[:200],
                "score": doc.get("cross_encoder_score", doc.get("rrf_score", 0.0)),
            }
        )
    return sources


# ---------------------------------------------------------------------------
# Context builder (formats docs into LLM prompt context block)
# ---------------------------------------------------------------------------
def format_docs_for_prompt(docs: list[dict], max_chars: int = 3000) -> str:
    """Format reranked docs into a numbered context block for the LLM."""
    if not docs:
        return "No relevant documents found."
    parts = []
    total = 0
    for i, doc in enumerate(docs, 1):
        snippet = doc.get("document", "")
        meta = doc.get("metadata", {})
        header = f"[{i}] {meta.get('filename','')} | sheet={meta.get('sheet','')} | row={meta.get('row','')}"
        block = f"{header}\n{snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def format_sql_results(results: list[dict]) -> str:
    """Format SQL aggregation results into a readable string for the LLM."""
    if not results:
        return ""
    lines = []
    for r in results:
        if r.get("error"):
            lines.append(f"SQL Error: {r['error']}")
        elif isinstance(r.get("result"), list):
            lines.append(f"Query: {r.get('query','')}")
            for row in r["result"]:
                lines.append("  " + " | ".join(f"{k}={v}" for k, v in row.items()))
        else:
            lines.append(f"Result: {r.get('result')} (query: {r.get('query','')})")
    return "\n".join(lines)
