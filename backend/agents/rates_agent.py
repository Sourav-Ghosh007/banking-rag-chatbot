"""
Rates Agent
===========
Handles: interest rates, forex / exchange rates, FD/RD rates,
         market benchmark rates, rate comparisons.

All rate data comes from the ingested rates.xlsx — never fabricated.
"""

from __future__ import annotations

import json
import logging
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
from backend.rag.sql_engine import (
    get_all_schemas,
    run_safe_sql,
    run_aggregation,
    get_user_tables,
)
from backend.rag.metrics import log_query, log_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM = """
You are a banking rates and market data specialist. You have access to:
- Interest rate sheets (savings, FD, RD, home loan, personal loan rates)
- Foreign exchange / currency rates
- Benchmark rates (repo, MCLR, base rate)

Guidelines:
- Always quote the source table and effective date when giving a rate.
- For rate comparisons, present as a clear table if multiple rates exist.
- Never round rates — show exact values from the data (e.g. 7.25% not "about 7%").
- If a rate is not in the data, say so explicitly — do not use general knowledge rates.
- Flag if rates may be outdated based on the data's effective date column.
- For forex: state both BUY and SELL rates when available.
""".strip()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_rates",
            "description": "Search rate tables with optional filters (currency, rate_type, tenure).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A read-only SELECT statement against rates tables.",
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_rates",
            "description": "Aggregate rate data — e.g. MAX rate for a given tenure.",
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
            "name": "fd_maturity_calculator",
            "description": (
                "Calculate FD maturity amount given principal, rate, and tenure. "
                "Supports simple and compound interest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "principal":       {"type": "number", "description": "Deposit amount in INR"},
                    "annual_rate_pct": {"type": "number", "description": "Annual interest rate %"},
                    "tenure_days":     {"type": "integer", "description": "Tenure in days"},
                    "compounding":     {
                        "type": "string",
                        "enum": ["quarterly", "monthly", "simple"],
                        "description": "Compounding frequency (default: quarterly)",
                    },
                },
                "required": ["principal", "annual_rate_pct", "tenure_days"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# FD calculator (deterministic)
# ---------------------------------------------------------------------------
def _fd_maturity(
    principal: float,
    annual_rate_pct: float,
    tenure_days: int,
    compounding: str = "quarterly",
) -> dict:
    r = annual_rate_pct / 100
    t = tenure_days / 365

    if compounding == "simple":
        interest = principal * r * t
        maturity = principal + interest
    elif compounding == "monthly":
        n = 12
        maturity = principal * (1 + r / n) ** (n * t)
        interest = maturity - principal
    else:  # quarterly (default)
        n = 4
        maturity = principal * (1 + r / n) ** (n * t)
        interest = maturity - principal

    return {
        "principal": principal,
        "annual_rate_pct": annual_rate_pct,
        "tenure_days": tenure_days,
        "compounding": compounding,
        "maturity_amount": round(maturity, 2),
        "interest_earned": round(interest, 2),
    }


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------
async def _execute_tool(tool_name: str, args: dict, user_id: str) -> str:
    try:
        if tool_name == "lookup_rates":
            result = run_safe_sql(user_id, args["sql"])
            return json.dumps(result)

        elif tool_name == "aggregate_rates":
            result = run_aggregation(
                user_id=user_id,
                table_name=args["table_name"],
                agg_func=args["agg_func"],
                column=args["column"],
                filters=args.get("filters"),
                group_by=args.get("group_by"),
            )
            return json.dumps(result)

        elif tool_name == "fd_maturity_calculator":
            result = _fd_maturity(
                float(args["principal"]),
                float(args["annual_rate_pct"]),
                int(args["tenure_days"]),
                args.get("compounding", "quarterly"),
            )
            return json.dumps(result)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as exc:
        logger.error("RatesAgent tool %s failed: %s", tool_name, exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------
async def rates_agent(state: AgentState) -> dict:
    t0 = time.perf_counter()
    user_id = state["user_id"]

    messages = state.get("messages", [])
    user_query = next(
        (m.content for m in reversed(messages)
         if getattr(m, "type", getattr(m, "role", None)) in ("human", "user")),
        "",
    )

    # RAG retrieval — rates are often best found by keyword (BM25 shines here)
    reranked_docs, rm = [], {}
    try:
        reranked_docs, rm_obj = retrieve_and_rerank(
            query=user_query, user_id=user_id,
            top_n_retrieve=20, top_k_rerank=5,
        )
        rm = rm_obj.to_dict()
    except Exception as exc:
        log_error(user_id, "retrieval_error", str(exc), {"agent": "rates"})

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

    for _round in range(3):
        try:
            response = await llm_chat(messages=llm_messages, tools=_TOOLS, tool_choice="auto")
        except Exception as exc:
            log_error(user_id, "llm_error", str(exc), {"agent": "rates"})
            return {"agent_response": "Rate lookup failed. Please try again.", "error": str(exc)}

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            break

        llm_messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
            for tc in tool_calls
        ]})
        for tc in tool_calls:
            result = await _execute_tool(tc["name"], tc["arguments"], user_id)
            try:
                sql_results.append(json.loads(result))
            except Exception:
                pass
            llm_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    final_answer = extract_text(response) if response else "Unable to retrieve rates."
    elapsed = (time.perf_counter() - t0) * 1000
    log_query(
        user_id=user_id, query=user_query,
        n_candidates=rm.get("n_candidates", 0), n_final=rm.get("n_final", 0),
        latency_retrieval_ms=rm.get("latency_retrieval_ms", 0),
        latency_rerank_ms=rm.get("latency_rerank_ms", 0),
        latency_total_ms=elapsed,
        cross_encoder_top1=rm.get("cross_encoder_top1", 0),
        cross_encoder_mean=rm.get("cross_encoder_mean", 0),
        agent_name="rates_agent",
    )
    logger.info("RatesAgent: done in %.0f ms for user %s", elapsed, user_id)

    return {
        "agent_response": final_answer,
        "reranked_docs": reranked_docs,
        "sql_results": sql_results,
        "sources": build_sources(reranked_docs),
        "latency_ms": round(elapsed, 2),
        "retrieval_metrics": rm,
    }
