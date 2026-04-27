"""
Account Agent
=============
Handles questions about transactions, account balance, spending history.

Simple explanation:
- User asks something like "What did I spend last month?"
- This agent searches the uploaded transaction data
- Uses SQL for numbers (total, average) and RAG for specific lookups
- Returns a clear answer
"""

import json
import logging
import time

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
# What this agent knows about
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful bank account assistant.
You help customers understand their transactions and spending.

Rules:
- Only use data from the customer's uploaded files
- For totals/averages, always use the SQL tool (never guess numbers)
- Format money as ₹X,XX,XXX
- If data is not available, say so clearly
- Keep answers short and simple
"""

# ---------------------------------------------------------------------------
# Tools the agent can use
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_total",
            "description": "Calculate SUM, AVG, COUNT, MIN or MAX on transaction data",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Name of the table"},
                    "agg_func":   {"type": "string", "enum": ["SUM", "AVG", "COUNT", "MIN", "MAX"]},
                    "column":     {"type": "string", "description": "Column to calculate on"},
                    "filters":    {"type": "object", "description": "Optional filters like {category: 'food'}"},
                    "group_by":   {"type": "string", "description": "Optional grouping column"},
                },
                "required": ["table_name", "agg_func", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transactions",
            "description": "Run a custom SQL query to find specific transactions",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A SELECT SQL query"},
                },
                "required": ["sql"],
            },
        },
    },
]


async def _run_tool(tool_name: str, args: dict, user_id: str) -> str:
    """Execute the tool the LLM requested and return result as string."""
    try:
        if tool_name == "calculate_total":
            result = run_aggregation(
                user_id=user_id,
                table_name=args["table_name"],
                agg_func=args["agg_func"],
                column=args["column"],
                filters=args.get("filters"),
                group_by=args.get("group_by"),
            )
            return json.dumps(result)

        elif tool_name == "search_transactions":
            result = run_safe_sql(user_id, args["sql"])
            return json.dumps(result)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.error("Tool error in account agent: %s", e)
        return json.dumps({"error": str(e)})


async def account_agent(state: AgentState) -> dict:
    """
    Main account agent function.
    Called by LangGraph when intent = 'transactions'.
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

    # Step 1: Search relevant documents using RAG
    reranked_docs, rm = [], {}
    try:
        reranked_docs, rm_obj = retrieve_and_rerank(
            query=user_question,
            user_id=user_id,
            top_n_retrieve=20,
            top_k_rerank=5,
        )
        rm = rm_obj.to_dict()
    except Exception as e:
        log_error(user_id, "retrieval_error", str(e))

    # Step 2: Get database table info so LLM knows what tables exist
    schema_text = ""
    try:
        schemas = get_all_schemas(user_id)
        if schemas:
            lines = [
                f"  Table '{t}': columns = {', '.join(c['name'] for c in cols if not c['name'].startswith('_'))}"
                for t, cols in schemas.items()
            ]
            schema_text = "Available database tables:\n" + "\n".join(lines)
    except Exception:
        pass

    # Step 3: Build the prompt for GPT
    doc_context = format_docs_for_prompt(reranked_docs)
    full_system = f"{SYSTEM_PROMPT}\n\n{schema_text}\n\nRelevant data found:\n{doc_context}"

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": user_question},
    ]

    # Step 4: LLM loop — GPT can call tools up to 3 times
    sql_results = []
    response = None

    for round_num in range(3):
        try:
            response = await llm_chat(messages=messages, tools=TOOLS, tool_choice="auto")
        except Exception as e:
            log_error(user_id, "llm_error", str(e))
            return {"agent_response": "Sorry, I could not process your request. Please try again.", "error": str(e)}

        tool_calls = extract_tool_calls(response)

        # If no tool calls, GPT has the final answer
        if not tool_calls:
            break

        # Add GPT's tool request to message history
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in tool_calls
            ],
        })

        # Run each tool and give results back to GPT
        for tc in tool_calls:
            result = await _run_tool(tc["name"], tc["arguments"], user_id)
            try:
                sql_results.append(json.loads(result))
            except Exception:
                pass
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    # Step 5: Return the final answer
    final_answer = extract_text(response) if response else "I could not find relevant transaction data."
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    log_query(
        user_id=user_id,
        query=user_question,
        n_candidates=rm.get("n_candidates", 0),
        n_final=rm.get("n_final", 0),
        latency_retrieval_ms=rm.get("latency_retrieval_ms", 0),
        latency_rerank_ms=rm.get("latency_rerank_ms", 0),
        latency_total_ms=elapsed_ms,
        cross_encoder_top1=rm.get("cross_encoder_top1", 0),
        cross_encoder_mean=rm.get("cross_encoder_mean", 0),
        agent_name="account_agent",
    )

    return {
        "agent_response": final_answer,
        "reranked_docs":  reranked_docs,
        "sql_results":    sql_results,
        "sources":        build_sources(reranked_docs),
        "latency_ms":     round(elapsed_ms, 2),
        "retrieval_metrics": rm,
    }
