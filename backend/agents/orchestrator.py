"""
Orchestrator Agent (Router)
===========================
This is the FIRST agent that runs for every user question.

Simple explanation:
- User types a question
- Orchestrator reads it and decides which specialist to send it to
- It uses GPT to classify the intent into one of 5 categories
- LangGraph then routes to the correct agent

Example:
  "What is my loan EMI?" → classified as 'loans' → sent to loan_agent
  "Send email to John"   → classified as 'comms'  → sent to notify agent
"""

import json
import logging
import time

from backend.agents.base import AgentState, llm_chat, extract_text
from backend.rag.metrics import log_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The 5 categories and which agent handles each
# ---------------------------------------------------------------------------
INTENT_TO_AGENT = {
    "transactions": "account_agent",    # account.py
    "loans":        "loan_agent",       # loan_agent.py
    "rates":        "rates_agent",      # rates_agent.py
    "comms":        "comms_agent",      # notify.py
    "general":      "account_agent",    # fallback
}

# ---------------------------------------------------------------------------
# Instructions for GPT on how to classify
# ---------------------------------------------------------------------------
ROUTER_PROMPT = """
You are a banking assistant router. Your ONLY job is to classify the user's
question into one of these 5 categories and return JSON.

Categories:
- transactions : balance, payments, spending, transaction history
- loans        : loan EMI, eligibility, repayment, outstanding balance
- rates        : interest rates, forex, FD rates, exchange rates
- comms        : send email, schedule meeting, Slack message, calendar
- general      : anything else banking related

Return ONLY this JSON (no extra text):
{
  "intent": "<category>",
  "confidence": <0.0 to 1.0>,
  "reason": "<one short sentence>"
}
"""


async def orchestrator_agent(state: AgentState) -> dict:
    """
    Router node — classifies intent and sets target_agent.
    LangGraph calls this first for every query.
    """
    start_time = time.perf_counter()
    user_id = state["user_id"]

    # Get the user's question from state
    user_question = ""
    for msg in reversed(state.get("messages", [])):
        role = getattr(msg, "type", getattr(msg, "role", None))
        if role in ("human", "user"):
            user_question = msg.content
            break

    if not user_question.strip():
        return {
            "intent": "general",
            "target_agent": "account_agent",
            "confidence": 0.0,
            "error": "Empty question received.",
        }

    # Ask GPT to classify the question
    try:
        response = await llm_chat(
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": user_question},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=100,
        )
        result = json.loads(extract_text(response))

    except Exception as e:
        logger.error("Orchestrator failed for user %s: %s", user_id, e)
        log_error(user_id, "router_error", str(e))
        # Safe fallback if classification fails
        return {
            "intent": "general",
            "target_agent": "account_agent",
            "confidence": 0.0,
            "error": str(e),
        }

    # Get the classified intent
    intent = result.get("intent", "general").lower().strip()
    if intent not in INTENT_TO_AGENT:
        intent = "general"

    confidence = float(result.get("confidence", 0.5))
    target_agent = INTENT_TO_AGENT[intent]

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("Router: '%s' → %s (confidence=%.2f)", intent, target_agent, confidence)

    return {
        "intent":       intent,
        "target_agent": target_agent,
        "confidence":   confidence,
        "retrieval_metrics": {
            "router_latency_ms": round(elapsed_ms, 2),
            "router_reason":     result.get("reason", ""),
        },
    }


def route_to_agent(state: AgentState) -> str:
    """
    LangGraph edge function.
    After orchestrator runs, this tells LangGraph which node to go to next.
    """
    return state.get("target_agent", "account_agent")
