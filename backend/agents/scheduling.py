"""
Scheduling Agent
================
Handles scheduling of banking appointments, EMI reminders, and payment due dates.

Simple explanation:
- User says "Remind me about my loan EMI on the 5th of every month"
- This agent creates calendar events or reminders
- Also helps find available appointment slots with bank staff
- Works with the notify agent (Google Calendar) under the hood
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta

from backend.agents.base import (
    AgentState,
    llm_chat,
    extract_text,
    extract_tool_calls,
)
from backend.rag.metrics import log_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a banking scheduling assistant.
You help customers:
1. Schedule appointments with bank staff
2. Set reminders for loan EMI payments
3. Set reminders for FD/RD maturity dates
4. Find available time slots

Rules:
- Always confirm the date, time, and timezone before creating anything
- Default timezone is IST (UTC+5:30) unless user specifies otherwise
- For EMI reminders: suggest 3 days before the due date
- Keep scheduling simple and confirm with the user before creating
"""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Create a calendar reminder or appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":       {"type": "string", "description": "Reminder title"},
                    "date":        {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time":        {"type": "string", "description": "Time in HH:MM format (24hr)"},
                    "description": {"type": "string", "description": "Details about the reminder"},
                    "attendees":   {"type": "array", "items": {"type": "string"}, "description": "Email addresses to invite"},
                },
                "required": ["title", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_emi_reminder_series",
            "description": "Create monthly EMI payment reminders for the next N months",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_name":   {"type": "string", "description": "Name of the loan (e.g. Home Loan)"},
                    "emi_amount":  {"type": "number", "description": "Monthly EMI amount"},
                    "due_day":     {"type": "integer", "description": "Day of month the EMI is due (1-28)"},
                    "months":      {"type": "integer", "description": "How many months to set reminders for"},
                    "remind_days_before": {"type": "integer", "description": "Days before due date to remind (default 3)"},
                },
                "required": ["loan_name", "emi_amount", "due_day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_reminders",
            "description": "List upcoming scheduled reminders and appointments",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead": {"type": "integer", "description": "Days ahead to look (default 30)"},
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
async def _run_tool(tool_name: str, args: dict, user_id: str) -> str:
    try:
        if tool_name == "create_reminder":
            # Build datetime string
            date_str = args["date"]
            time_str = args.get("time", "09:00")
            start_dt = f"{date_str}T{time_str}:00+05:30"   # IST timezone

            # Calculate end time (1 hour later by default)
            start = datetime.fromisoformat(start_dt)
            end_dt = (start + timedelta(hours=1)).isoformat()

            # Delegate to notify agent's calendar tool
            from backend.agents.notify import _get_token, _call_service, CALENDAR_URL
            token = await _get_token(user_id, "google_calendar")
            if not token:
                return json.dumps({"error": "Google Calendar not connected. Please connect it in profile settings."})

            result = await _call_service(CALENDAR_URL, token, "create_event", {
                "summary":     args["title"],
                "description": args.get("description", ""),
                "start":       {"dateTime": start_dt},
                "end":         {"dateTime": end_dt},
                "attendees":   [{"email": e} for e in args.get("attendees", [])],
            })
            return json.dumps({"status": "created", "event": result})

        elif tool_name == "create_emi_reminder_series":
            loan_name  = args["loan_name"]
            emi_amount = args["emi_amount"]
            due_day    = int(args["due_day"])
            months     = int(args.get("months", 3))
            remind_before = int(args.get("remind_days_before", 3))

            reminders_created = []
            today = datetime.now(timezone.utc)

            for i in range(months):
                # Calculate the due date for this month
                month = (today.month + i - 1) % 12 + 1
                year  = today.year + (today.month + i - 1) // 12
                try:
                    due_date = datetime(year, month, due_day, tzinfo=timezone.utc)
                except ValueError:
                    continue  # skip invalid dates (e.g. Feb 30)

                # Reminder is X days before due date
                reminder_date = due_date - timedelta(days=remind_before)
                reminder_str  = reminder_date.strftime("%Y-%m-%d")
                due_str       = due_date.strftime("%d %b %Y")

                reminders_created.append({
                    "reminder_date": reminder_str,
                    "title": f"EMI Reminder: {loan_name} — ₹{emi_amount:,.0f} due on {due_str}",
                })

            return json.dumps({
                "status": "planned",
                "reminders": reminders_created,
                "note": "Connect Google Calendar to auto-create these reminders.",
            })

        elif tool_name == "get_upcoming_reminders":
            days = int(args.get("days_ahead", 30))
            # Return upcoming reminders from calendar
            from backend.agents.notify import _get_token, _call_service, CALENDAR_URL
            token = await _get_token(user_id, "google_calendar")
            if not token:
                return json.dumps({"error": "Google Calendar not connected."})

            now    = datetime.now(timezone.utc).isoformat()
            future = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
            result = await _call_service(CALENDAR_URL, token, "list_events", {
                "timeMin": now, "timeMax": future, "maxResults": 20,
            })
            return json.dumps(result)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.error("Scheduling tool error: %s", e)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------
async def scheduling_agent(state: AgentState) -> dict:
    """
    Scheduling agent node.
    Handles appointment and reminder scheduling.
    """
    start_time = time.perf_counter()
    user_id = state["user_id"]

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

    response = None
    actions = []

    for _ in range(4):
        try:
            response = await llm_chat(messages=messages, tools=TOOLS, tool_choice="auto")
        except Exception as e:
            log_error(user_id, "llm_error", str(e), {"agent": "scheduling"})
            return {"agent_response": "Scheduling service error. Please try again.", "error": str(e)}

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
            actions.append({"tool": tc["name"], "result": result[:300]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    final_answer = extract_text(response) if response else "Could not complete scheduling task."
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info("SchedulingAgent: done in %.0f ms for user %s", elapsed_ms, user_id)

    return {
        "agent_response":    final_answer,
        "sql_results":       actions,
        "sources":           [],
        "latency_ms":        round(elapsed_ms, 2),
        "retrieval_metrics": {"scheduling_actions": len(actions)},
    }
