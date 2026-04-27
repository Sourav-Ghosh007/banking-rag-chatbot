"""
Notify Agent (Communications)
==============================
Handles: sending emails, scheduling calendar events, posting Slack messages.

Simple explanation:
- User says "Send an email to John about my loan approval"
- This agent figures out what action to take
- It calls Gmail / Google Calendar / Slack via their APIs
- OAuth2 tokens are stored securely (encrypted) in the database per user
"""

import json
import logging
import os
import time

import httpx

from backend.agents.base import AgentState, llm_chat, extract_text, extract_tool_calls
from backend.rag.metrics import log_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API endpoints (set in your .env file)
# ---------------------------------------------------------------------------
GMAIL_URL    = os.environ.get("GMAIL_MCP_URL",    "https://gmailmcp.googleapis.com/mcp/v1")
CALENDAR_URL = os.environ.get("CALENDAR_MCP_URL", "https://calendarmcp.googleapis.com/mcp/v1")
SLACK_URL    = os.environ.get("SLACK_MCP_URL",    "")

# ---------------------------------------------------------------------------
# System prompt — keep it simple
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a banking communication assistant.
You help users send emails, schedule meetings, and post Slack messages.

Rules:
- Always confirm what you are about to do before doing it
- For emails: confirm recipient, subject, and message
- For calendar: confirm date, time, and attendees
- For Slack: confirm channel and message
- Never make up email addresses or contacts
"""

# ---------------------------------------------------------------------------
# Tools available to the agent
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email using the user's Gmail account",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body":    {"type": "string", "description": "Email body text"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a meeting or event in Google Calendar",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":      {"type": "string"},
                    "start_time": {"type": "string", "description": "ISO format: 2025-06-01T10:00:00"},
                    "end_time":   {"type": "string", "description": "ISO format: 2025-06-01T11:00:00"},
                    "attendees":  {"type": "array", "items": {"type": "string"}, "description": "List of email addresses"},
                },
                "required": ["title", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_slack_message",
            "description": "Post a message to a Slack channel",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Channel name like #general"},
                    "message": {"type": "string", "description": "The message to post"},
                },
                "required": ["channel", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": "List upcoming calendar events for the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead": {"type": "integer", "description": "How many days ahead to look (default 7)"},
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Get the user's stored OAuth token from the database
# ---------------------------------------------------------------------------
async def _get_token(user_id: str, provider: str) -> str | None:
    """
    Fetch and decrypt the OAuth token for a provider (gmail, google_calendar, slack).
    Returns None if user has not connected that service.
    """
    try:
        from backend.db.session import AsyncSessionLocal
        from sqlalchemy import text
        from cryptography.fernet import Fernet

        fernet_key = os.environ.get("FERNET_KEY", "")
        if not fernet_key:
            logger.error("FERNET_KEY not configured")
            return None

        f = Fernet(fernet_key.encode())

        async with AsyncSessionLocal() as session:
            row = await session.execute(
                text("SELECT encrypted_token FROM user_oauth_tokens WHERE user_id=:uid AND provider=:prov LIMIT 1"),
                {"uid": user_id, "prov": provider},
            )
            record = row.fetchone()
            if not record:
                return None

            token_data = json.loads(f.decrypt(record[0].encode()).decode())
            return token_data.get("access_token")

    except Exception as e:
        logger.error("Token fetch failed for %s/%s: %s", user_id, provider, e)
        return None


# ---------------------------------------------------------------------------
# Call an external service (Gmail/Calendar/Slack) via MCP
# ---------------------------------------------------------------------------
async def _call_service(url: str, token: str, tool: str, params: dict) -> dict:
    """Send a request to an MCP server and return the result."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": params},
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {})


# ---------------------------------------------------------------------------
# Execute each tool call
# ---------------------------------------------------------------------------
async def _run_tool(tool_name: str, args: dict, user_id: str) -> str:
    try:
        if tool_name == "send_email":
            token = await _get_token(user_id, "gmail")
            if not token:
                return json.dumps({"error": "Gmail not connected. Please connect Gmail in your profile settings."})
            result = await _call_service(GMAIL_URL, token, "send_email", {
                "to": args["to"], "subject": args["subject"], "body": args["body"],
            })
            return json.dumps(result)

        elif tool_name == "create_calendar_event":
            token = await _get_token(user_id, "google_calendar")
            if not token:
                return json.dumps({"error": "Google Calendar not connected. Please connect it in profile settings."})
            result = await _call_service(CALENDAR_URL, token, "create_event", {
                "summary":   args["title"],
                "start":     {"dateTime": args["start_time"]},
                "end":       {"dateTime": args["end_time"]},
                "attendees": [{"email": e} for e in args.get("attendees", [])],
            })
            return json.dumps(result)

        elif tool_name == "list_calendar_events":
            token = await _get_token(user_id, "google_calendar")
            if not token:
                return json.dumps({"error": "Google Calendar not connected."})
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc).isoformat()
            days = int(args.get("days_ahead", 7))
            future = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
            result = await _call_service(CALENDAR_URL, token, "list_events", {
                "timeMin": now, "timeMax": future, "maxResults": 10,
            })
            return json.dumps(result)

        elif tool_name == "post_slack_message":
            token = await _get_token(user_id, "slack")
            if not token:
                return json.dumps({"error": "Slack not connected. Please connect Slack in profile settings."})
            if not SLACK_URL:
                return json.dumps({"error": "Slack MCP URL not configured on server."})
            result = await _call_service(SLACK_URL, token, "chat.postMessage", {
                "channel": args["channel"], "text": args["message"],
            })
            return json.dumps(result)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except httpx.HTTPError as e:
        return json.dumps({"error": f"Service unavailable: {e}"})
    except Exception as e:
        logger.error("Notify agent tool error: %s", e)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------
async def comms_agent(state: AgentState) -> dict:
    """
    Notify agent node — called by LangGraph when intent = 'comms'.
    """
    start_time = time.perf_counter()
    user_id = state["user_id"]

    # Get user's question
    user_question = ""
    for msg in reversed(state.get("messages", [])):
        role = getattr(msg, "type", getattr(msg, "role", None))
        if role in ("human", "user"):
            user_question = msg.content
            break

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_question},
    ]

    actions_taken = []
    response = None

    # LLM loop — up to 5 rounds for multi-step comms tasks
    for _ in range(5):
        try:
            response = await llm_chat(messages=messages, tools=TOOLS, tool_choice="auto")
        except Exception as e:
            log_error(user_id, "llm_error", str(e), {"agent": "notify"})
            return {"agent_response": "Communication service error. Please try again.", "error": str(e)}

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            break

        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
            for tc in tool_calls
        ]})

        for tc in tool_calls:
            result = await _run_tool(tc["name"], tc["arguments"], user_id)
            actions_taken.append({"tool": tc["name"], "result": result[:300]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    final_answer = extract_text(response) if response else "Could not complete the communication task."
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info("NotifyAgent: done in %.0f ms, %d actions for user %s", elapsed_ms, len(actions_taken), user_id)

    return {
        "agent_response":    final_answer,
        "sql_results":       actions_taken,
        "sources":           [],
        "latency_ms":        round(elapsed_ms, 2),
        "retrieval_metrics": {"actions_taken": len(actions_taken)},
    }
