"""
SQL Aggregation Engine
======================
Executes structured aggregations (SUM, AVG, COUNT, MIN, MAX, GROUP BY)
against per-user in-memory SQLite tables loaded by ingest.py.

Design principle: all numeric answers come from deterministic SQL — never
from LLM estimation.
"""

import re
import logging
import sqlite3
from typing import Any, Optional

from backend.rag.ingest import _get_sqlite_conn, list_user_documents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe identifier regex (prevent SQL injection via table/column names)
# ---------------------------------------------------------------------------
_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_]+$")


'''def _safe_id(name: str) -> str:
    """Validate and return a safe SQL identifier, raise if invalid."""
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: '{name}'")
    return name
'''

def _safe_id(name: str) -> str:
    """Validate and return a safe SQL identifier."""
    # Allow alphanumeric and underscores only
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    return cleaned

# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------
def get_user_tables(user_id: str) -> list[str]:
    """Return all user-owned table names (excludes internal _* tables)."""
    conn = _get_sqlite_conn(user_id)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return [r[0] for r in rows]


def get_table_schema(user_id: str, table_name: str) -> list[dict]:
    """
    Return column info for a table: [{name, type, numeric}].
    Raises ValueError for unknown tables.
    """
    available = get_user_tables(user_id)
    if table_name not in available:
        raise ValueError(
            f"Table '{table_name}' not found. Available: {available}"
        )
    conn = _get_sqlite_conn(user_id)
    #rows = conn.execute(f"PRAGMA table_info({_safe_id(table_name)})").fetchall()
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [
        {
            "name": r["name"],
            "type": r["type"],
            "numeric": r["type"].upper() in ("INTEGER", "REAL", "NUMERIC", "FLOAT"),
        }
        for r in rows
    ]


def get_all_schemas(user_id: str) -> dict[str, list[dict]]:
    """Return schemas for every table the user has — used by agents for planning."""
    result = {}
    for table in get_user_tables(user_id):
        try:
            result[table] = get_table_schema(user_id, table)
        except Exception as exc:
            logger.warning("Schema fetch failed for table %s: %s", table, exc)
    return result


# ---------------------------------------------------------------------------
# Core aggregation builder
# ---------------------------------------------------------------------------
def run_aggregation(
    user_id: str,
    table_name: str,
    agg_func: str,                      # SUM | AVG | COUNT | MIN | MAX
    column: str,                         # target column (ignored for COUNT *)
    filters: Optional[dict] = None,      # {col: value} equality filters
    group_by: Optional[str] = None,      # column to group results by
    order_by: Optional[str] = None,      # column to order results by
    order_dir: str = "DESC",
    limit: int = 100,
) -> dict[str, Any]:
    """
    Execute a single aggregation query and return structured results.

    Parameters
    ----------
    agg_func  : one of SUM, AVG, COUNT, MIN, MAX
    column    : column to aggregate (use '*' for COUNT)
    filters   : simple equality WHERE conditions
    group_by  : optional GROUP BY column
    order_by  : optional ORDER BY column
    order_dir : ASC or DESC
    limit     : max rows returned

    Returns
    -------
    {
        "query"  : the SQL executed,
        "result" : scalar value OR list[dict] for GROUP BY,
        "rows"   : int,
        "error"  : str or None,
    }
    """
    ALLOWED_AGG = {"SUM", "AVG", "COUNT", "MIN", "MAX"}
    ALLOWED_DIR = {"ASC", "DESC"}

    agg_func = agg_func.upper()
    order_dir = order_dir.upper()

    if agg_func not in ALLOWED_AGG:
        return _error(f"agg_func must be one of {ALLOWED_AGG}, got '{agg_func}'")
    if order_dir not in ALLOWED_DIR:
        return _error(f"order_dir must be ASC or DESC")

    try:
        _safe_id(table_name)
    except ValueError as e:
        return _error(str(e))

    # Build SELECT clause
    if column == "*":
        select_expr = "COUNT(*)"
        result_alias = "count"
    else:
        try:
            _safe_id(column)
        except ValueError as e:
            return _error(str(e))
        select_expr = f"{agg_func}({column})"
        result_alias = f"{agg_func.lower()}_{column}"

    if group_by:
        try:
            _safe_id(group_by)
        except ValueError as e:
            return _error(str(e))

    # WHERE clause (parameterised)
    where_parts: list[str] = []
    params: list[Any] = []
    if filters:
        for col, val in filters.items():
            try:
                _safe_id(col)
            except ValueError as e:
                return _error(str(e))
            where_parts.append(f"{col} = ?")
            params.append(val)

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    # GROUP BY / ORDER BY
    group_sql = f"GROUP BY {group_by}" if group_by else ""

    if group_by:
        select_full = f"{group_by}, {select_expr} AS {result_alias}"
        order_col = order_by or result_alias
        try:
            _safe_id(order_col)
        except ValueError:
            order_col = result_alias
        order_sql = f"ORDER BY {order_col} {order_dir}"
    else:
        select_full = f"{select_expr} AS {result_alias}"
        order_sql = ""

    limit_sql = f"LIMIT {int(limit)}"

    sql = f"""
        SELECT {select_full}
        FROM   "{table_name}"
        {where_sql}
        {group_sql}
        {order_sql}
        {limit_sql}
    """.strip()

    return _execute(user_id, sql, params, result_alias, grouped=bool(group_by))


# ---------------------------------------------------------------------------
# Free-form safe SQL execution (read-only)
# ---------------------------------------------------------------------------
def run_safe_sql(user_id: str, sql: str, params: Optional[list] = None) -> dict[str, Any]:
    """
    Execute an arbitrary read-only SQL statement supplied by an agent.
    Rejects any statement that isn't a SELECT.
    """
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return _error("Only SELECT statements are permitted in run_safe_sql.")

    # Basic injection guard — no stacked statements
    if ";" in sql.rstrip(";"):
        return _error("Multiple statements detected — only single SELECT allowed.")

    return _execute(user_id, sql, params or [], result_alias=None, grouped=False)


# ---------------------------------------------------------------------------
# Convenience wrappers — used by agents
# ---------------------------------------------------------------------------
def sum_column(user_id: str, table: str, column: str, filters: Optional[dict] = None) -> dict:
    return run_aggregation(user_id, table, "SUM", column, filters=filters)


def avg_column(user_id: str, table: str, column: str, filters: Optional[dict] = None) -> dict:
    return run_aggregation(user_id, table, "AVG", column, filters=filters)


def count_rows(user_id: str, table: str, filters: Optional[dict] = None) -> dict:
    return run_aggregation(user_id, table, "COUNT", "*", filters=filters)


def min_column(user_id: str, table: str, column: str, filters: Optional[dict] = None) -> dict:
    return run_aggregation(user_id, table, "MIN", column, filters=filters)


def max_column(user_id: str, table: str, column: str, filters: Optional[dict] = None) -> dict:
    return run_aggregation(user_id, table, "MAX", column, filters=filters)


def group_sum(user_id: str, table: str, column: str, group_by: str, filters: Optional[dict] = None) -> dict:
    return run_aggregation(user_id, table, "SUM", column, filters=filters, group_by=group_by)


def group_avg(user_id: str, table: str, column: str, group_by: str, filters: Optional[dict] = None) -> dict:
    return run_aggregation(user_id, table, "AVG", column, filters=filters, group_by=group_by)


def group_count(user_id: str, table: str, group_by: str, filters: Optional[dict] = None) -> dict:
    return run_aggregation(user_id, table, "COUNT", "*", filters=filters, group_by=group_by)


# ---------------------------------------------------------------------------
# Natural-language to SQL helper (intent parsing)
# ---------------------------------------------------------------------------
_AGG_KEYWORDS = {
    "total": "SUM", "sum": "SUM",
    "average": "AVG", "avg": "AVG", "mean": "AVG",
    "count": "COUNT", "how many": "COUNT", "number of": "COUNT",
    "minimum": "MIN", "min": "MIN", "lowest": "MIN", "smallest": "MIN",
    "maximum": "MAX", "max": "MAX", "highest": "MAX", "largest": "MAX",
}


def parse_aggregation_intent(query: str) -> Optional[dict]:
    """
    Heuristic parser: extract {agg_func, hint_column} from a natural language
    query string. Returns None if no aggregation intent detected.
    Used by the router agent to decide whether to invoke sql_engine or retriever.
    """
    q = query.lower()
    detected_agg = None
    for keyword, func in _AGG_KEYWORDS.items():
        if keyword in q:
            detected_agg = func
            break

    if detected_agg is None:
        return None

    return {"agg_func": detected_agg, "raw_query": query}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _execute(
    user_id: str,
    sql: str,
    params: list,
    result_alias: Optional[str],
    grouped: bool,
) -> dict[str, Any]:
    conn = _get_sqlite_conn(user_id)
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        logger.debug("SQL executed for user %s:\n%s\nParams: %s", user_id, sql, params)

        if grouped or result_alias is None:
            # Return list of dicts
            cols = [d[0] for d in cursor.description]
            data = [dict(zip(cols, r)) for r in rows]
            return {"query": sql, "result": data, "rows": len(data), "error": None}
        else:
            # Scalar
            scalar = rows[0][0] if rows else None
            # Round floats for readability
            if isinstance(scalar, float):
                scalar = round(scalar, 4)
            return {"query": sql, "result": scalar, "rows": 1, "error": None}

    except sqlite3.Error as exc:
        logger.error("SQLite error for user %s: %s | SQL: %s", user_id, exc, sql)
        return _error(str(exc), sql)


def _error(msg: str, sql: str = "") -> dict[str, Any]:
    return {"query": sql, "result": None, "rows": 0, "error": msg}
