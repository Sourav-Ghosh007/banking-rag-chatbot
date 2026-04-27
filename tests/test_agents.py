"""
Agent Tests
===========
Tests for the LangGraph agents.
Run with: pytest tests/test_agents.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Test: Orchestrator classifies intents correctly
# ---------------------------------------------------------------------------
class TestOrchestratorAgent:

    @pytest.mark.asyncio
    async def test_routes_loan_query(self):
        """Loan question should be routed to loan_agent."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"intent": "loans", "confidence": 0.95, "reason": "EMI question"}'
        mock_response.choices[0].message.tool_calls = None

        with patch("backend.agents.base.get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

            from backend.agents.orchestrator import orchestrator_agent
            state = {
                "messages": [MagicMock(type="human", content="What is my loan EMI?")],
                "user_id": "test_user",
                "session_id": "test_session",
            }
            result = await orchestrator_agent(state)

        assert result["intent"] == "loans"
        assert result["target_agent"] == "loan_agent"
        assert result["confidence"] >= 0.9

    @pytest.mark.asyncio
    async def test_routes_transaction_query(self):
        """Transaction question should be routed to account_agent."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"intent": "transactions", "confidence": 0.9, "reason": "spending query"}'
        mock_response.choices[0].message.tool_calls = None

        with patch("backend.agents.base.get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

            from backend.agents.orchestrator import orchestrator_agent
            state = {
                "messages": [MagicMock(type="human", content="What did I spend last month?")],
                "user_id": "test_user",
                "session_id": "test_session",
            }
            result = await orchestrator_agent(state)

        assert result["intent"] == "transactions"
        assert result["target_agent"] == "account_agent"

    @pytest.mark.asyncio
    async def test_empty_query_fallback(self):
        """Empty query should fallback to general intent without crashing."""
        from backend.agents.orchestrator import orchestrator_agent
        state = {
            "messages": [MagicMock(type="human", content="   ")],
            "user_id": "test_user",
            "session_id": "test_session",
        }
        result = await orchestrator_agent(state)
        assert result["intent"] == "general"
        assert result["target_agent"] == "account_agent"

    def test_route_to_agent_function(self):
        """route_to_agent should return the target_agent from state."""
        from backend.agents.orchestrator import route_to_agent
        state = {"target_agent": "loan_agent"}
        assert route_to_agent(state) == "loan_agent"

    def test_route_to_agent_default(self):
        """route_to_agent should fallback if target_agent is missing."""
        from backend.agents.orchestrator import route_to_agent
        state = {}
        assert route_to_agent(state) == "account_agent"


# ---------------------------------------------------------------------------
# Test: Loan agent EMI calculations
# ---------------------------------------------------------------------------
class TestLoanAgentMath:

    def test_emi_standard(self):
        """Standard EMI calculation — reducing balance."""
        from backend.agents.loan_agent import _emi
        result = _emi(principal=500000, annual_rate_pct=8.5, tenure_months=60)

        assert result["emi"] == pytest.approx(10224.49, abs=1.0)
        assert result["total_payable"] == pytest.approx(613469.4, abs=10)
        assert result["total_interest"] == pytest.approx(113469.4, abs=10)

    def test_emi_zero_interest(self):
        """Zero interest loan — EMI = principal / months."""
        from backend.agents.loan_agent import _emi
        result = _emi(principal=120000, annual_rate_pct=0, tenure_months=12)
        assert result["emi"] == pytest.approx(10000.0, abs=0.01)
        assert result["total_interest"] == 0.0

    def test_amortisation_first_month(self):
        """First month: interest = principal * monthly_rate."""
        from backend.agents.loan_agent import _amortisation
        schedule = _amortisation(500000, 8.5, 60, rows=1)
        assert len(schedule) == 1
        assert schedule[0]["month"] == 1
        # First month interest ≈ 500000 * (8.5/100/12) ≈ 3541.67
        assert schedule[0]["interest_component"] == pytest.approx(3541.67, abs=1.0)

    def test_emi_invalid_inputs(self):
        """Negative tenure should return an error dict."""
        from backend.agents.loan_agent import _emi
        result = _emi(principal=100000, annual_rate_pct=10, tenure_months=-1)
        assert "error" in result


# ---------------------------------------------------------------------------
# Test: Rates agent FD calculator
# ---------------------------------------------------------------------------
class TestRatesAgent:

    def test_fd_quarterly_compounding(self):
        """FD maturity with quarterly compounding."""
        from backend.agents.rates_agent import _fd_maturity
        result = _fd_maturity(100000, 7.0, 365, compounding="quarterly")
        assert result["maturity_amount"] == pytest.approx(107186.0, abs=10)
        assert result["interest_earned"]  == pytest.approx(7186.0,  abs=10)

    def test_fd_simple_interest(self):
        """FD with simple interest for 1 year."""
        from backend.agents.rates_agent import _fd_maturity
        result = _fd_maturity(100000, 7.0, 365, compounding="simple")
        assert result["maturity_amount"] == pytest.approx(107000.0, abs=1)
        assert result["interest_earned"]  == pytest.approx(7000.0,  abs=1)

    def test_fd_returns_correct_keys(self):
        """FD result should have all required keys."""
        from backend.agents.rates_agent import _fd_maturity
        result = _fd_maturity(50000, 6.5, 180)
        required_keys = ["principal", "annual_rate_pct", "tenure_days", "maturity_amount", "interest_earned"]
        for key in required_keys:
            assert key in result


# ---------------------------------------------------------------------------
# Test: Graph routing
# ---------------------------------------------------------------------------
class TestGraph:

    def test_graph_builds_without_error(self):
        """The LangGraph graph should compile without throwing exceptions."""
        from backend.agents.graph import build_graph
        graph = build_graph()
        assert graph is not None

    def test_get_graph_is_cached(self):
        """get_graph() should return the same object on repeated calls."""
        from backend.agents.graph import get_graph
        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2   # same object — cached via lru_cache
