"""
Compliance Agent
================
Checks if transactions or actions follow banking rules and regulations.

Simple explanation:
- Looks at transaction data for unusual patterns
- Flags transactions that might need review (large amounts, unusual frequency)
- Answers questions about banking regulations and compliance
- Does NOT make legal decisions — only flags for human review
"""

import json
import logging
import time

from backend.agents.base import (
    AgentState,
    llm_chat,
    extract_text,
    build_sources,
    format_docs_for_prompt,
)
from backend.rag.reranker import retrieve_and_rerank
from backend.rag.sql_engine import run_safe_sql, get_all_schemas
from backend.rag.metrics import log_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a banking compliance assistant.
You help identify potential compliance issues in transactions and answer
questions about banking regulations.

Rules:
- Flag transactions above ₹10,00,000 as requiring review (RBI guideline)
- Flag if same amount is transferred multiple times in a day (structuring risk)
- Always say "requires human review" — never make final compliance decisions
- Be factual and reference the data, not assumptions
- Keep language simple and clear
"""

# ---------------------------------------------------------------------------
# Compliance rules (simple thresholds)
# ---------------------------------------------------------------------------
HIGH_VALUE_THRESHOLD = 1_000_000   # ₹10 lakh — requires review per RBI
FREQUENT_TRANSACTION_COUNT = 5     # 5+ same-amount transactions = flag


async def compliance_agent(state: AgentState) -> dict:
    """
    Compliance agent node.
    Can be called directly or triggered by other agents for compliance checks.
    """
    start_time = time.perf_counter()
    user_id = state["user_id"]

    # Get the user's question
    user_question = ""
    for msg in reversed(state.get("messages", [])):
        role = getattr(msg, "type", getattr(msg, "role", None))
        if role in ("human", "user"):
            user_question = msg.content
            break

    # Search for relevant compliance-related documents
    reranked_docs = []
    try:
        reranked_docs, _ = retrieve_and_rerank(
            query=user_question,
            user_id=user_id,
            top_n_retrieve=15,
            top_k_rerank=5,
        )
    except Exception as e:
        log_error(user_id, "retrieval_error", str(e), {"agent": "compliance"})

    # Run basic compliance checks on the data
    compliance_flags = []
    try:
        schemas = get_all_schemas(user_id)
        for table_name in schemas:
            # Check for high-value transactions
            high_value_result = run_safe_sql(
                user_id,
                f"SELECT * FROM {table_name} WHERE CAST(amount AS REAL) > {HIGH_VALUE_THRESHOLD} LIMIT 10"
            )
            if high_value_result.get("result") and high_value_result["rows"] > 0:
                compliance_flags.append(
                    f"⚠️ Found {high_value_result['rows']} high-value transaction(s) above ₹10,00,000 in '{table_name}'"
                )
    except Exception as e:
        logger.debug("Compliance SQL check skipped: %s", e)

    # Build prompt with flags and document context
    flags_text = "\n".join(compliance_flags) if compliance_flags else "No automatic flags raised."
    doc_context = format_docs_for_prompt(reranked_docs)

    full_system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Automated compliance flags:\n{flags_text}\n\n"
        f"Relevant data:\n{doc_context}"
    )

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": user_question},
    ]

    # Get GPT's compliance assessment
    try:
        response = await llm_chat(messages=messages, temperature=0.0, max_tokens=512)
        final_answer = extract_text(response)
    except Exception as e:
        log_error(user_id, "llm_error", str(e), {"agent": "compliance"})
        final_answer = "Compliance check encountered an error. Please try again."

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info("ComplianceAgent: done in %.0f ms for user %s", elapsed_ms, user_id)

    return {
        "agent_response": final_answer,
        "reranked_docs":  reranked_docs,
        "sql_results":    [{"flags": compliance_flags}],
        "sources":        build_sources(reranked_docs),
        "latency_ms":     round(elapsed_ms, 2),
        "retrieval_metrics": {},
    }


# ---------------------------------------------------------------------------
# Helper: can be called by other agents to do a quick compliance check
# ---------------------------------------------------------------------------
async def check_transaction_compliance(user_id: str, amount: float, description: str) -> dict:
    """
    Quick compliance check for a single transaction.
    Returns: {compliant: bool, flags: list, message: str}
    """
    flags = []

    if amount > HIGH_VALUE_THRESHOLD:
        flags.append(f"Amount ₹{amount:,.2f} exceeds ₹10,00,000 — requires RBI reporting")

    return {
        "compliant":  len(flags) == 0,
        "flags":      flags,
        "message":    "Passed basic compliance checks." if not flags else "Flagged for review.",
        "amount":     amount,
        "description": description,
    }
