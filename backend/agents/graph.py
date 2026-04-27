"""
LangGraph Graph
===============
Wires all agents together into one pipeline.

Simple explanation:
- Every user question enters at the orchestrator (router)
- The orchestrator decides which specialist agent to call
- The specialist runs and returns the answer
- LangGraph manages the flow between agents

Flow:
  User Question
      ↓
  orchestrator  ← classifies intent
      ↓
  ┌───┴────────────────┐
  │                    │
account_agent     loan_agent     rates_agent     comms_agent
  (transactions)  (loans)        (rates)         (email/slack)
  │
  └── Answer returned to user
"""

import logging
from functools import lru_cache

from langgraph.graph import StateGraph, END

from backend.agents.base         import AgentState
from backend.agents.orchestrator import orchestrator_agent, route_to_agent
from backend.agents.account      import account_agent
from backend.agents.loan_agent   import loan_agent
from backend.agents.rates_agent  import rates_agent
from backend.agents.notify       import comms_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node name constants
# ---------------------------------------------------------------------------
NODE_ORCHESTRATOR = "orchestrator"
NODE_ACCOUNT      = "account_agent"
NODE_LOAN         = "loan_agent"
NODE_RATES        = "rates_agent"
NODE_COMMS        = "comms_agent"


def build_graph():
    """Build and compile the LangGraph agent graph."""

    graph = StateGraph(AgentState)

    # Register all agent nodes
    graph.add_node(NODE_ORCHESTRATOR, orchestrator_agent)
    graph.add_node(NODE_ACCOUNT,      account_agent)
    graph.add_node(NODE_LOAN,         loan_agent)
    graph.add_node(NODE_RATES,        rates_agent)
    graph.add_node(NODE_COMMS,        comms_agent)

    # Start at the orchestrator
    graph.set_entry_point(NODE_ORCHESTRATOR)

    # After orchestrator, route to the correct agent
    graph.add_conditional_edges(
        NODE_ORCHESTRATOR,
        route_to_agent,
        {
            NODE_ACCOUNT: NODE_ACCOUNT,
            NODE_LOAN:    NODE_LOAN,
            NODE_RATES:   NODE_RATES,
            NODE_COMMS:   NODE_COMMS,
        },
    )

    # All specialist agents end the graph
    for node in [NODE_ACCOUNT, NODE_LOAN, NODE_RATES, NODE_COMMS]:
        graph.add_edge(node, END)

    compiled = graph.compile()
    logger.info("LangGraph compiled successfully with %d agent nodes", 4)
    return compiled


@lru_cache(maxsize=1)
def get_graph():
    """Return the compiled graph (created once, reused for all requests)."""
    return build_graph()


# ---------------------------------------------------------------------------
# Main entry point — called by FastAPI chat router
# ---------------------------------------------------------------------------
async def run_query(user_id: str, session_id: str, user_message: str) -> dict:
    """
    Run a user question through the full agent pipeline.

    Returns a dict with:
      - agent_response : the answer text
      - sources        : list of source documents used
      - intent         : what category the question was classified as
      - target_agent   : which agent answered
      - latency_ms     : total time taken
      - error          : any error message (None if successful)
    """
    from backend.agents.base import make_initial_state

    graph = get_graph()
    initial_state = make_initial_state(user_id, session_id, user_message)

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("Graph failed for user %s: %s", user_id, e)
        return {
            "agent_response": "Sorry, something went wrong. Please try again.",
            "sources":        [],
            "intent":         None,
            "target_agent":   None,
            "confidence":     0.0,
            "latency_ms":     0.0,
            "retrieval_metrics": {},
            "sql_results":    [],
            "error":          str(e),
        }

    return {
        "agent_response":    final_state.get("agent_response", ""),
        "sources":           final_state.get("sources", []),
        "intent":            final_state.get("intent"),
        "target_agent":      final_state.get("target_agent"),
        "confidence":        final_state.get("confidence", 0.0),
        "latency_ms":        final_state.get("latency_ms", 0.0),
        "retrieval_metrics": final_state.get("retrieval_metrics", {}),
        "sql_results":       final_state.get("sql_results", []),
        "error":             final_state.get("error"),
    }
