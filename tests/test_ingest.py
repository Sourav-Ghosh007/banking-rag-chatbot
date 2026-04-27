"""
Ingest & RAG Tests
==================
Tests for file validation, parsing, SQL engine, and retriever.
Run with: pytest tests/test_ingest.py -v
"""

import io
import os
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Test: File validation
# ---------------------------------------------------------------------------
class TestFileValidation:

    def test_csv_accepted(self):
        """CSV files should pass validation."""
        from backend.rag.ingest import validate_file_extension
        validate_file_extension("transactions.csv")   # no error

    def test_xlsx_accepted(self):
        """Excel files should pass validation."""
        from backend.rag.ingest import validate_file_extension
        validate_file_extension("loans.xlsx")   # no error

    def test_pdf_rejected(self):
        """PDF files must be rejected — by design."""
        from backend.rag.ingest import validate_file_extension
        with pytest.raises(ValueError, match="not supported"):
            validate_file_extension("statement.pdf")

    def test_docx_rejected(self):
        """Word documents should be rejected."""
        from backend.rag.ingest import validate_file_extension
        with pytest.raises(ValueError):
            validate_file_extension("report.docx")

    def test_png_rejected(self):
        """Image files should be rejected."""
        from backend.rag.ingest import validate_file_extension
        with pytest.raises(ValueError):
            validate_file_extension("screenshot.png")

    def test_case_insensitive(self):
        """Extension check should be case-insensitive (.CSV = .csv)."""
        from backend.rag.ingest import validate_file_extension
        validate_file_extension("DATA.CSV")    # no error
        validate_file_extension("report.XLSX") # no error


# ---------------------------------------------------------------------------
# Test: File parsing
# ---------------------------------------------------------------------------
class TestFileParsing:

    def _make_csv_bytes(self) -> bytes:
        """Create a simple CSV in memory for testing."""
        df = pd.DataFrame({
            "date":   ["2025-01-01", "2025-01-02"],
            "amount": [1000.0, 2500.0],
            "type":   ["credit", "debit"],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    def _make_xlsx_bytes(self) -> bytes:
        """Create a simple Excel file in memory for testing."""
        df = pd.DataFrame({
            "loan_id": ["L001", "L002"],
            "amount":  [500000, 1000000],
            "rate":    [8.5, 9.0],
        })
        buf = io.BytesIO()
        df.to_excel(buf, index=False, sheet_name="Loans")
        return buf.getvalue()

    def test_csv_parsing_returns_dataframe(self):
        """CSV file should parse into a dict with one DataFrame."""
        from backend.rag.ingest import _parse_file
        csv_bytes = self._make_csv_bytes()
        result = _parse_file(csv_bytes, "test.csv")
        assert len(result) == 1
        sheet_name = list(result.keys())[0]
        df = result[sheet_name]
        assert len(df) == 2
        assert "amount" in df.columns

    def test_xlsx_parsing_returns_sheets(self):
        """Excel file should parse into a dict with one DataFrame per sheet."""
        from backend.rag.ingest import _parse_file
        xlsx_bytes = self._make_xlsx_bytes()
        result = _parse_file(xlsx_bytes, "loans.xlsx")
        assert "Loans" in result or "loans" in [k.lower() for k in result]

    def test_column_names_lowercased(self):
        """Column names should be lowercased and spaces replaced with underscores."""
        from backend.rag.ingest import _parse_file
        df_raw = pd.DataFrame({"Transaction Date": ["2025-01-01"], "Total Amount": [999]})
        buf = io.BytesIO()
        df_raw.to_csv(buf, index=False)
        result = _parse_file(buf.getvalue(), "test.csv")
        df = list(result.values())[0]
        assert "transaction_date" in df.columns
        assert "total_amount"     in df.columns


# ---------------------------------------------------------------------------
# Test: Text chunking
# ---------------------------------------------------------------------------
class TestTextChunking:

    def test_short_text_not_split(self):
        """Text shorter than CHUNK_SIZE should return as single chunk."""
        from backend.rag.ingest import _chunk_text
        text   = "date: 2025-01-01 | amount: 1000 | type: credit"
        chunks = _chunk_text(text, size=300)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split(self):
        """Text longer than CHUNK_SIZE should be split into multiple chunks."""
        from backend.rag.ingest import _chunk_text
        text   = "x" * 700
        chunks = _chunk_text(text, size=300, overlap=50)
        assert len(chunks) > 1

    def test_row_to_text_format(self):
        """Row should format as [sheet] | col: val | col: val."""
        from backend.rag.ingest import _row_to_text
        row  = pd.Series({"amount": "1000", "type": "credit"})
        text = _row_to_text(row, "transactions")
        assert "[transactions]" in text
        assert "amount: 1000"   in text
        assert "type: credit"   in text


# ---------------------------------------------------------------------------
# Test: SQL Engine
# ---------------------------------------------------------------------------
class TestSQLEngine:

    def _setup_test_db(self, user_id: str) -> None:
        """Insert test data into the user's in-memory SQLite."""
        from backend.rag.ingest import _get_sqlite_conn
        conn = _get_sqlite_conn(user_id)
        conn.execute("CREATE TABLE IF NOT EXISTS test_txn (amount REAL, category TEXT)")
        conn.execute("INSERT INTO test_txn VALUES (1000, 'food')")
        conn.execute("INSERT INTO test_txn VALUES (2500, 'travel')")
        conn.execute("INSERT INTO test_txn VALUES (500,  'food')")
        conn.commit()

    def test_sum_column(self):
        """SUM should return correct total."""
        from backend.rag.sql_engine import run_aggregation
        user_id = "sql_test_user_sum"
        self._setup_test_db(user_id)
        result = run_aggregation(user_id, "test_txn", "SUM", "amount")
        assert result["error"]  is None
        assert result["result"] == pytest.approx(4000.0)

    def test_avg_column(self):
        """AVG should return correct average."""
        from backend.rag.sql_engine import run_aggregation
        user_id = "sql_test_user_avg"
        self._setup_test_db(user_id)
        result = run_aggregation(user_id, "test_txn", "AVG", "amount")
        assert result["error"]  is None
        assert result["result"] == pytest.approx(4000 / 3, abs=0.01)

    def test_count_rows(self):
        """COUNT should return correct row count."""
        from backend.rag.sql_engine import run_aggregation
        user_id = "sql_test_user_count"
        self._setup_test_db(user_id)
        result = run_aggregation(user_id, "test_txn", "COUNT", "*")
        assert result["result"] == 3

    def test_group_by(self):
        """GROUP BY should return one row per group."""
        from backend.rag.sql_engine import run_aggregation
        user_id = "sql_test_user_grp"
        self._setup_test_db(user_id)
        result = run_aggregation(user_id, "test_txn", "SUM", "amount", group_by="category")
        assert result["error"] is None
        assert len(result["result"]) == 2   # food and travel

    def test_sql_injection_blocked(self):
        """Malicious table names should be rejected."""
        from backend.rag.sql_engine import run_aggregation
        result = run_aggregation("any_user", "users; DROP TABLE users--", "COUNT", "*")
        assert result["error"] is not None

    def test_non_select_blocked(self):
        """Only SELECT statements should be allowed in run_safe_sql."""
        from backend.rag.sql_engine import run_safe_sql
        result = run_safe_sql("any_user", "DELETE FROM test_txn")
        assert result["error"] is not None

    def test_aggregation_intent_detection(self):
        """Queries with 'total' should be detected as SUM intent."""
        from backend.rag.sql_engine import parse_aggregation_intent
        result = parse_aggregation_intent("What is the total spending this month?")
        assert result is not None
        assert result["agg_func"] == "SUM"

    def test_no_aggregation_intent(self):
        """Queries without aggregation words should return None."""
        from backend.rag.sql_engine import parse_aggregation_intent
        result = parse_aggregation_intent("Show me my recent transactions")
        assert result is None
