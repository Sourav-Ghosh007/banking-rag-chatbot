"""
Chat Router
===========
Simple explanation:
- This is the main chatbot endpoint: POST /api/chat
- Frontend sends: { "message": "What is my loan EMI?" }
- We pass it through the LangGraph agent pipeline
- We return: { "answer": "...", "sources": [...], "intent": "loans" }

Every request is tied to the logged-in user (via JWT token)
so each user only sees their own data.
"""

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session  import get_db
from backend.db.crud     import create_audit_log
from backend.routers.auth import get_current_user
from backend.agents.graph import run_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message:    str                     # the user's question
    session_id: str | None = None       # optional — groups messages into a conversation


class SourceDoc(BaseModel):
    filename: str
    sheet:    str
    snippet:  str
    score:    float


class ChatResponse(BaseModel):
    answer:       str                   # the agent's answer
    intent:       str | None           # what category was detected (loans, transactions, etc.)
    target_agent: str | None           # which agent answered
    sources:      list[SourceDoc]      # which documents were used
    latency_ms:   float                # how long it took
    session_id:   str                  # echo back the session id


# ---------------------------------------------------------------------------
# Main chat endpoint
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatResponse, summary="Send a message to the banking assistant")
async def chat(
    request:      ChatRequest,
    current_user  = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Main chatbot endpoint.

    1. Validates the user is logged in
    2. Sends the message through the LangGraph agent pipeline
    3. Returns the answer with sources and metadata
    """
    user_id = str(current_user.id)

    # Validate message is not empty
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Use provided session_id or create a new one
    session_id = request.session_id or str(uuid.uuid4())

    logger.info("Chat request from user %s: '%s'", user_id, request.message[:80])

    # Run the message through all agents
    start = time.perf_counter()
    try:
        result = await run_query(
            user_id=user_id,
            session_id=session_id,
            user_message=request.message,
        )
    except Exception as e:
        logger.error("Agent pipeline failed for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Agent processing failed. Please try again.")

    elapsed = (time.perf_counter() - start) * 1000

    # If agent returned an error, still give a friendly response
    answer = result.get("agent_response") or "I could not find an answer. Please try rephrasing."

    # Build source citations for the frontend
    raw_sources = result.get("sources", [])
    sources = [
        SourceDoc(
            filename=s.get("filename", ""),
            sheet=s.get("sheet", ""),
            snippet=s.get("snippet", "")[:200],
            score=round(s.get("score", 0.0), 4),
        )
        for s in raw_sources[:5]   # show max 5 sources
    ]

    # Log for audit trail
    await create_audit_log(
        db,
        user_id=user_id,
        action="chat_query",
        details=f"intent={result.get('intent')} | agent={result.get('target_agent')} | latency={elapsed:.0f}ms",
    )

    logger.info(
        "Chat done: user=%s intent=%s agent=%s latency=%.0fms",
        user_id, result.get("intent"), result.get("target_agent"), elapsed,
    )

    return ChatResponse(
        answer=answer,
        intent=result.get("intent"),
        target_agent=result.get("target_agent"),
        sources=sources,
        latency_ms=round(elapsed, 2),
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Chat history endpoint (simple — returns last N audit logs for the user)
# ---------------------------------------------------------------------------

@router.get("/history", summary="Get recent chat history")
async def chat_history(
    limit:        int = 20,
    current_user  = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns the user's recent chat queries from the audit log."""
    from backend.db.crud import get_user_audit_logs
    logs = await get_user_audit_logs(db, str(current_user.id), limit=limit)

    history = [
        {
            "action":     log.action,
            "details":    log.details,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
        if log.action == "chat_query"
    ]
    return {"history": history, "count": len(history)}
