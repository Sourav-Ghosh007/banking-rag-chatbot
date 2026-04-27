"""
Loan Agent
==========
Handles: loan applications, EMI computation, eligibility checks,
         outstanding balances, repayment schedules, loan product info.

Special capability: deterministic EMI calculator (no LLM estimation).
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Optional

from backend.agents.base import (
    AgentState,
    llm_chat,
    extract_text,
    extract_tool_calls,
    build_sources,
    format_docs_for_prompt,
)
from backend.rag.reranker import retrieve_and_rerank
from backend.rag.sql_engine import get_all_schemas, run_aggregation, run_safe_sql
from backend.rag.metrics import log_query, log_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM = """
You are a banking loan specialist assistant. You have access to:
- The customer's loan portfolio data (principal, rate, tenure, status)
- Loan product catalogue (types, eligibility criteria, interest rates)
- EMI calculation tool (always use this — never estimate EMI manually)

Guidelines:
- EMI = P × r × (1+r)^n / ((1+r)^n – 1) — but use the tool, not mental math.
- State loan amounts with tenure: "₹5,00,000 over 60 months at 8.5% p.a."
- For eligibility: always note that final approval depends on credit assessment.
- Do NOT invent loan products or rates not present in the data.
- Be clear about outstanding principal vs total amount payable.
""".strip()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_emi",
            "description": "Calculate monthly EMI for a loan using the standard reducing-balance formula.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal":        {"type": "number", "description": "Loan principal in INR"},
                    "annual_rate_pct":  {"type": "number", "description": "Annual interest rate as percentage (e.g. 8.5)"},
                    "tenure_months":    {"type": "integer", "description": "Loan tenure in months"},
                },
                "required": ["principal", "annual_rate_pct", "tenure_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loan_summary",
            "description": "Retrieve aggregated loan data for the user (total outstanding, count by type, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "agg_func":   {"type": "string", "enum": ["SUM", "AVG", "COUNT", "MIN", "MAX"]},
                    "column":     {"type": "string"},
                    "filters":    {"type": "object"},
                    "group_by":   {"type": "string"},
                },
                "required": ["table_name", "agg_func", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_loan_sql",
            "description": "Run a custom read-only SELECT on loan tables for complex queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_amortisation_schedule",
            "description": "Generate a full amortisation schedule (month-by-month breakdown).",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal":       {"type": "number"},
                    "annual_rate_pct": {"type": "number"},
                    "tenure_months":   {"type": "integer"},
                    "rows_to_show":    {"type": "integer", "description": "How many months to display (default 6)"},
                },
                "required": ["principal", "annual_rate_pct", "tenure_months"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------
def _emi(principal: float, annual_rate_pct: float, tenure_months: int) -> dict:
    """Reducing-balance EMI calculation."""
    if tenure_months <= 0 or principal <= 0:
        return {"error": "Principal and tenure must be positive."}
    if annual_rate_pct <= 0:
        # Zero-interest loan
        emi = principal / tenure_months
        return {
            "emi": round(emi, 2),
            "total_payable": round(emi * tenure_months, 2),
            "total_interest": 0.0,
            "principal": principal,
            "annual_rate_pct": 0.0,
            "tenure_months": tenure_months,
        }
    r = (annual_rate_pct / 100) / 12
    emi = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    total_payable = emi * tenure_months
    return {
        "emi": round(emi, 2),
        "total_payable": round(total_payable, 2),
        "total_interest": round(total_payable - principal, 2),
        "principal": principal,
        "annual_rate_pct": annual_rate_pct,
        "tenure_months": tenure_months,
    }


def _amortisation(principal: float, annual_rate_pct: float, tenure_months: int, rows: int = 6) -> list[dict]:
    """Generate amortisation schedule rows."""
    if annual_rate_pct <= 0:
        emi = principal / tenure_months
        schedule = []
        balance = principal
        for m in range(1, min(rows, tenure_months) + 1):
            balance -= emi
            schedule.append({
                "month": m, "emi": round(emi, 2),
                "principal_component": round(emi, 2),
                "interest_component": 0.0,
                "closing_balance": round(max(balance, 0), 2),
            })
        return schedule

    r = (annual_rate_pct / 100) / 12
    emi_calc = _emi(principal, annual_rate_pct, tenure_months)
    emi = emi_calc["emi"]
    schedule = []
    balance = principal
    for m in range(1, min(rows, tenure_months) + 1):
        interest = balance * r
        principal_component = emi - interest
        balance -= principal_component
        schedule.append({
            "month": m,
            "emi": round(emi, 2),
            "principal_component": round(principal_component, 2),
            "interest_component": round(interest, 2),
            "closing_balance": round(max(balance, 0), 2),
        })
    return schedule


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------
async def _execute_tool(tool_name: str, args: dict, user_id: str) -> str:
    try:
        if tool_name == "calculate_emi":
            result = _emi(
                float(args["principal"]),
                float(args["annual_rate_pct"]),
                int(args["tenure_months"]),
            )
            return json.dumps(result)

        elif tool_name == "generate_amortisation_schedule":
            rows = int(args.get("rows_to_show", 6))
            schedule = _amortisation(
                float(args["principal"]),
                float(args["annual_rate_pct"]),
                int(args["tenure_months"]),
                rows=rows,
            )
            emi_info = _emi(float(args["principal"]), float(args["annual_rate_pct"]), int(args["tenure_months"]))
            return json.dumps({"emi_info": emi_info, "schedule": schedule})

        elif tool_name == "get_loan_summary":
            result = run_aggregation(
                user_id=user_id,
                table_name=args["table_name"],
                agg_func=args["agg_func"],
                column=args["column"],
                filters=args.get("filters"),
                group_by=args.get("group_by"),
            )
            return json.dumps(result)

        elif tool_name == "run_loan_sql":
            result = run_safe_sql(user_id, args["sql"])
            return json.dumps(result)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as exc:
        logger.error("LoanAgent tool %s failed: %s", tool_name, exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------
async def loan_agent(state: AgentState) -> dict:
    t0 = time.perf_counter()
    user_id = state["user_id"]

    messages = state.get("messages", [])
    user_query = next(
        (m.content for m in reversed(messages)
         if getattr(m, "type", getattr(m, "role", None)) in ("human", "user")),
        "",
    )

    # RAG retrieval
    reranked_docs, rm = [], {}
    try:
        reranked_docs, rm_obj = retrieve_and_rerank(
            query=user_query, user_id=user_id,
            top_n_retrieve=20, top_k_rerank=5,
        )
        rm = rm_obj.to_dict()
    except Exception as exc:
        log_error(user_id, "retrieval_error", str(exc), {"agent": "loan"})

    # Schema context
    schema_info = ""
    try:
        schemas = get_all_schemas(user_id)
        if schemas:
            lines = [
                f"  {t}: {', '.join(c['name'] for c in cols if not c['name'].startswith('_'))}"
                for t, cols in schemas.items()
            ]
            schema_info = "Available tables:\n" + "\n".join(lines)
    except Exception:
        pass

    doc_context = format_docs_for_prompt(reranked_docs)
    system_msg = f"{_SYSTEM}\n\n## Schema\n{schema_info}\n\n## Document Context\n{doc_context}"

    llm_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_query},
    ]

    sql_results: list[dict] = []
    response = None
    for _round in range(4):   # up to 4 rounds for complex EMI + amort combos
        try:
            response = await llm_chat(messages=llm_messages, tools=_TOOLS, tool_choice="auto")
        except Exception as exc:
            log_error(user_id, "llm_error", str(exc), {"agent": "loan"})
            return {"agent_response": "Loan query failed. Please try again.", "error": str(exc)}

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            break

        llm_messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
            for tc in tool_calls
        ]})
        for tc in tool_calls:
            tool_result = await _execute_tool(tc["name"], tc["arguments"], user_id)
            try:
                sql_results.append(json.loads(tool_result))
            except Exception:
                pass
            llm_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

    final_answer = extract_text(response) if response else "Unable to process loan query."
    elapsed = (time.perf_counter() - t0) * 1000
    log_query(
        user_id=user_id, query=user_query,
        n_candidates=rm.get("n_candidates", 0), n_final=rm.get("n_final", 0),
        latency_retrieval_ms=rm.get("latency_retrieval_ms", 0),
        latency_rerank_ms=rm.get("latency_rerank_ms", 0),
        latency_total_ms=elapsed,
        cross_encoder_top1=rm.get("cross_encoder_top1", 0),
        cross_encoder_mean=rm.get("cross_encoder_mean", 0),
        agent_name="loan_agent",
    )
    logger.info("LoanAgent: done in %.0f ms for user %s", elapsed, user_id)

    return {
        "agent_response": final_answer,
        "reranked_docs": reranked_docs,
        "sql_results": sql_results,
        "sources": build_sources(reranked_docs),
        "latency_ms": round(elapsed, 2),
        "retrieval_metrics": rm,
    }
