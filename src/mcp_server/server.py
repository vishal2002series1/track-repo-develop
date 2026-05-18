# src/mcp_server/server.py
import os

# 🛑 CRITICAL: Suppress aggressive C++ gRPC logging in the background MCP process
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import re
# import sqlite3  # 🔴 DEPRECATED: SQLite - Replaced with Azure PostgreSQL (kept for reference)
import chromadb
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import json

import math as _math
from decimal import Decimal
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# 🟢 Ensure project root is on sys.path so 'src.*' imports resolve when this
# script is launched as a subprocess by the MCP client (python server.py).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 🟢 AZURE POSTGRESQL MIGRATION: Import shared connection module
from src.db.pg_connection import get_pg_connection, AZURE_PG_SCHEMA

# 🟢 LOCAL EMBEDDING MIGRATION: Swap AWS Bedrock for local HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
mcp = FastMCP("AeonWealthMCP")

# --- Bulletproof Pathing ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
LOCAL_DATA_DIR = os.path.join(BASE_DIR, 'data_local')
# SQLITE_DB_PATH = os.path.join(LOCAL_DATA_DIR, 'aeon_db.sqlite')  # 🔴 DEPRECATED: SQLite path (kept for reference)
CHROMA_DB_PATH = os.path.join(LOCAL_DATA_DIR, 'chroma_db')
MODEL_PATH = os.path.join(BASE_DIR, 'local_embedding_model')

# 🔌 Initialize the Local Offline Embedder
embedder = HuggingFaceEmbeddings(model_name=MODEL_PATH)

# ⚡ GLOBAL CACHE STATE
QUERY_CACHE = {}
SCHEMA_CACHE = {}
# LAST_DB_MTIME = 0  # 🔴 DEPRECATED: SQLite file mtime tracking (kept for reference)
CACHE_TTL_SECONDS = 300  # 🟢 Cache TTL for PostgreSQL (5 minutes)
LAST_CACHE_CLEAR = time.time()

def check_cache_invalidation():
    """Clears cache periodically for PostgreSQL (no file-based mtime check)."""
    global LAST_CACHE_CLEAR, QUERY_CACHE, SCHEMA_CACHE
    current_time = time.time()
    if current_time - LAST_CACHE_CLEAR > CACHE_TTL_SECONDS:
        QUERY_CACHE.clear()
        SCHEMA_CACHE.clear()
        LAST_CACHE_CLEAR = current_time
    # 🔴 DEPRECATED: SQLite file modification time check (kept for reference)
    # try:
    #     current_mtime = os.path.getmtime(SQLITE_DB_PATH)
    #     if current_mtime != LAST_DB_MTIME:
    #         QUERY_CACHE.clear()
    #         SCHEMA_CACHE.clear()
    #         LAST_DB_MTIME = current_mtime
    # except OSError:
    #     pass


def _rewrite_to_aeon_schema(query: str) -> str:
    """Force all SQL schema references to AZURE_PG_SCHEMA (e.g., aeon2)."""
    updated = query
    # Rewrite public."Table" -> aeon2."Table"
    updated = re.sub(r'(?i)\bpublic\s*\.', f'{AZURE_PG_SCHEMA}.', updated)
    # Rewrite "public"."Table" -> "aeon2"."Table"
    updated = re.sub(r'(?i)"public"\s*\.', f'"{AZURE_PG_SCHEMA}".', updated)
    return updated

###------------------MATH TOOL RELATED FUNCTIONS------------------###
# ── Module-level constants ─────────────────────────────────────────────────────
# Defined once at module load — never redefined per call.

# F12: Equity detection — keyword-in-string (substring match, lowercase).
# Handles: "Stocks", "US Equities", "Large Cap Growth Fund", "Domestic Equity" etc.
_EQUITY_KEYWORDS: frozenset = frozenset([
    "stock", "equit", "large cap", "small cap", "mid cap",
    "growth", "value fund", "domestic fund", "international fund",
])

# F12: Risk profiles that trigger mismatch check when equity > 50%
_CONSERVATIVE_PROFILES: frozenset = frozenset(["conservative", "moderate"])

# Tax sensitivity indicator values (lowercase match)
_TAX_SENSITIVE_VALUES: frozenset = frozenset(["high", "yes", "true", "1", "sensitive"])

# F14: Fee tiers by asset class (decimal annual rate).
# REVIEWER NOTE: Hard-coded per WF_002 spec. Future: move to FeeTier DB table.
_FEE_TIERS: Dict[str, float] = {
    "stocks":       0.0075,
    "equities":     0.0075,
    "bonds":        0.0050,
    "fixed income": 0.0050,
    "cash":         0.0010,
    "real estate":  0.0065,
    "alternatives": 0.0080,
    "commodities":  0.0060,
}
_DEFAULT_FEE: float = 0.0050  # Fallback for unrecognised asset class

# F15/F16: Long-run capital market assumptions.
# REVIEWER NOTE: Illustrative figures — always disclosed in tool output.
# Not a forecast. Must be presented to clients with full disclaimer.
_ASSET_PARAMS: Dict[str, Dict[str, float]] = {
    "stocks":       {"mu": 0.090, "sigma": 0.170},
    "equities":     {"mu": 0.090, "sigma": 0.170},
    "bonds":        {"mu": 0.040, "sigma": 0.060},
    "fixed income": {"mu": 0.040, "sigma": 0.060},
    "cash":         {"mu": 0.020, "sigma": 0.010},
    "real estate":  {"mu": 0.070, "sigma": 0.140},
    "alternatives": {"mu": 0.060, "sigma": 0.120},
    "commodities":  {"mu": 0.050, "sigma": 0.150},
}
_DEFAULT_PARAMS: Dict[str, float] = {"mu": 0.050, "sigma": 0.100}

# F13: Consolidation opportunity threshold (outside assets must exceed this)
_CONSOLIDATION_THRESHOLD: float = 250_000.0


# ── Private helpers ────────────────────────────────────────────────────────────
# Shared across all 4 math tools. Prefixed with _ — not exposed as MCP tools.

def _sf(value: Any, field: str = "field", fallback: float = 0.0) -> Tuple[float, Optional[str]]:
    """
    Safe float conversion. Handles all types returned by psycopg2 + Azure PostgreSQL:
      - None           → fallback + warning
      - int / float    → direct cast
      - Decimal        → float() cast (psycopg2 returns Decimal for NUMERIC columns)
      - datetime       → fallback + warning (wrong type for numeric field)
      - str "$1,200.50"→ strip $ and , then cast
      - empty string   → fallback + warning

    Returns: (converted_value, warning_message_or_None)
    """
    if value is None:
        return fallback, f"'{field}' is None — defaulted to {fallback}"
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, Decimal):
        # psycopg2 returns Decimal for NUMERIC/DECIMAL PostgreSQL columns
        return float(value), None
    try:
        cleaned = str(value).replace("$", "").replace(",", "").strip()
        if not cleaned:
            return fallback, f"'{field}' is empty string — defaulted to {fallback}"
        return float(cleaned), None
    except (ValueError, TypeError):
        return fallback, f"'{field}' value '{value}' ({type(value).__name__}) is not numeric — defaulted to {fallback}"


def _ss(value: Any, fallback: str = "Unknown") -> str:
    """Safe string: strips and returns value, or fallback if None/empty."""
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _norm(value: Any) -> str:
    """
    Display normalisation: strip + collapse internal whitespace.
    Preserves original casing — does NOT call .title() or .lower().
    Use for output keys and display values.
    NOTE: datetime objects from PostgreSQL become "2026-04-20 00:00:00" via str().
    """
    return " ".join(_ss(value).split())


def _nk(value: Any) -> str:
    """
    Key normalisation: lowercase + stripped.
    Use for comparisons and dict lookups ONLY — never for display output.
    """
    return _ss(value).lower().strip()


def _is_equity(asset_class: str) -> bool:
    """
    F12: Equity detection via keyword-in-string matching (not exact match).
    Correctly handles: "Stocks", "US Equities", "Large Cap Growth Fund",
    "Domestic Equity", "International Equities", "Equity" etc.
    """
    s = _nk(asset_class)
    return any(kw in s for kw in _EQUITY_KEYWORDS)


def _fee(asset_class: str) -> float:
    """F14: Fee tier lookup by asset class (normalised to lowercase for lookup)."""
    return _FEE_TIERS.get(_nk(asset_class), _DEFAULT_FEE)


def _params(asset_class: str) -> Dict[str, float]:
    """F15/F16: mu (expected return) and sigma (volatility) lookup by asset class."""
    return _ASSET_PARAMS.get(_nk(asset_class), _DEFAULT_PARAMS)


def _check_type(v: Any, t: type, field: str) -> Tuple[Any, Optional[str]]:
    """
    Type validation guard.
    Catches the common LLM mistake of passing stringified JSON instead of
    parsed Python objects (e.g. '{"key": "val"}' instead of {"key": "val"}).
    Returns: (value, error_string_or_None)
    """
    if v is None:
        return None, None  # Handled downstream with safe defaults
    if not isinstance(v, t):
        return None, (
            f"'{field}' must be {t.__name__}, got {type(v).__name__}. "
            "Pass parsed Python objects, not JSON strings."
        )
    return v, None


class _Warnings:
    """
    Severity-aware, deduplicated warning collector.

    Why deduplication?
    A portfolio with 50 holdings where all have CostBasis=0 would otherwise
    produce 50 identical warnings. Deduplication collapses these to one.

    data_quality() return values:
      "ok"       — no critical warnings
      "degraded" — at least one critical warning (agent should surface it)
      "blocked"  — returned directly in tool error responses (not via this class)
    """
    def __init__(self) -> None:
        self._msgs: List[str] = []
        self._seen: set = set()
        self.has_critical: bool = False

    def add(self, msg: str, critical: bool = False) -> None:
        # Deduplicate by first 100 chars of message
        key = msg[:100]
        if key in self._seen:
            return
        self._seen.add(key)
        prefix = "⚠️ CRITICAL" if critical else "⚠️"
        self._msgs.append(f"{prefix}: {msg}")
        if critical:
            self.has_critical = True

    def list(self) -> List[str]:
        return self._msgs

    def data_quality(self) -> str:
        return "degraded" if self.has_critical else "ok"


def _enrich(holdings: List[Dict], V: float) -> List[Dict]:
    """
    Build a local enriched copy of holdings rows.
    IMPORTANT: Never mutates the caller's input list or dicts.
    Each output dict adds computed fields prefixed with '_'.

    Expected input keys (from Holdings table):
      MarketValue, CostBasis, AssetClass, Sector, SecurityName, Ticker

    PostgreSQL type notes:
      MarketValue and CostBasis arrive as Decimal — handled by _sf().
      AsOfDate arrives as datetime — handled by str() in _norm().
    """
    out = []
    for h in holdings:
        mv, _ = _sf(h.get("MarketValue"), "MarketValue")
        cb, _ = _sf(h.get("CostBasis"),   "CostBasis")
        out.append({
            "_mv":  mv,
            "_cb":  cb,
            "_w":   mv / V if V > 0 else 0.0,
            "_ac":  _norm(h.get("AssetClass")),
            "_sec": _norm(h.get("Sector")),
            # SecurityName preferred; fall back to Ticker; then N/A
            "_nm":  _norm(h.get("SecurityName") or h.get("Ticker") or "N/A"),
        })
    return out


def _agg_pct(enriched: List[Dict], key: str, V: float) -> Dict[str, float]:
    """
    Aggregate MarketValue by a grouping key and express as % of total V.
    Used for asset-class allocation (key="_ac") and sector breakdown (key="_sec").
    """
    m: Dict[str, float] = {}
    for h in enriched:
        k = h[key]
        m[k] = m.get(k, 0.0) + h["_mv"]
    return {k: round((v / V) * 100, 2) for k, v in m.items()}


def _hhi(weights: List[float]) -> float:
    """
    Herfindahl-Hirschman Index.
    Formula: H = 10000 × Σ(wᵢ²)
    Range: 0 (perfectly diversified) → 10000 (single position).
    Input: list of weights each in [0, 1].
    """
    return round(10000 * sum(w ** 2 for w in weights), 2)


def _hhi_label(v: float) -> str:
    """Human-readable HHI interpretation for advisor output."""
    if v < 1500:
        return "Low — well diversified"
    if v < 2500:
        return "Moderate — some concentration"
    return "High — concentrated, review recommended"

#### MATH TOOLS (F12-F16) BELOW — USE THE ABOVE HELPERS FOR CALCULATIONS AND DATA QUALITY CHECKS ####



@mcp.tool()
def execute_sql(query: str) -> str:
    """Execute a read-only SQL query against the Aeon Wealth database."""
    global QUERY_CACHE
    check_cache_invalidation()

    normalized_query = _rewrite_to_aeon_schema(query)

    if normalized_query != query:
        print(
            f"\n      🧭 [SCHEMA REWRITE]: replaced 'public' with '{AZURE_PG_SCHEMA}'\n"
            f"      Original: {query}\n"
            f"      Rewritten: {normalized_query}\n",
            file=sys.stderr,
        )

    if normalized_query in QUERY_CACHE:
        print(f"\n      ⚡ [CACHE HIT - Instant Return]:\n{normalized_query}\n", file=sys.stderr)
        return QUERY_CACHE[normalized_query]

    print(f"\n      🟦 [MCP SQL Executing - Azure PostgreSQL]:\n{normalized_query}\n", file=sys.stderr)
    try:
        if normalized_query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE")):
            return "Error: Only SELECT queries are allowed."

        # 🟢 AZURE POSTGRESQL CONNECTION
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute(normalized_query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        # 🔴 DEPRECATED: SQLite connection (kept for reference)
        # conn = sqlite3.connect(SQLITE_DB_PATH)
        # cursor = conn.cursor()
        # cursor.execute(query)
        # columns = [description[0] for description in cursor.description]
        # rows = cursor.fetchall()
        # conn.close()

        if not rows:
            res = "No results found."
        else:
            res = " | ".join(columns) + "\n"
            res += "-" * len(res) + "\n"
            for row in rows:
                res += " | ".join(str(val) if val is not None else "NULL" for val in row) + "\n"
                
        QUERY_CACHE[normalized_query] = res
        return res
    except Exception as e:
        print(f"\n      ❌ [MCP SQL ERROR]: {str(e)}\n", file=sys.stderr)
        return f"SQL Error: {str(e)}"

@mcp.tool()
def get_database_schema(table_names: list[str] = None) -> str:
    """Returns the column information for requested tables from Azure PostgreSQL."""
    global SCHEMA_CACHE
    check_cache_invalidation()
    
    cache_key = str(table_names)
    if cache_key in SCHEMA_CACHE:
        print(f"\n      ⚡ [SCHEMA CACHE HIT]: {cache_key}\n", file=sys.stderr)
        return SCHEMA_CACHE[cache_key]

    print(f"\n      🗄️ [FETCHING SCHEMA - Azure PostgreSQL]: {cache_key}\n", file=sys.stderr)
    try:
        # 🟢 AZURE POSTGRESQL CONNECTION
        conn = get_pg_connection()
        cursor = conn.cursor()

        # 🟢 PostgreSQL: Use information_schema instead of sqlite_master
        if table_names and len(table_names) > 0:
            placeholders = ','.join(['%s'] * len(table_names))
            pg_query = f"""
                SELECT table_name, column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name IN ({placeholders})
                ORDER BY table_name, ordinal_position
            """
            cursor.execute(pg_query, (AZURE_PG_SCHEMA, *table_names))
        else:
            cursor.execute("""
                SELECT table_name, column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = %s
                ORDER BY table_name, ordinal_position
            """, (AZURE_PG_SCHEMA,))

        rows = cursor.fetchall()
        conn.close()

        # 🔴 DEPRECATED: SQLite schema query (kept for reference)
        # conn = sqlite3.connect(SQLITE_DB_PATH)
        # cursor = conn.cursor()
        # if table_names and len(table_names) > 0:
        #     placeholders = ','.join(['?'] * len(table_names))
        #     cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({placeholders})", tuple(table_names))
        # else:
        #     cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        # rows = cursor.fetchall()
        # conn.close()

        if not rows:
            return "No schema found for the requested tables."

        # 🟢 Format PostgreSQL schema as CREATE TABLE-style output
        res = "--- Database Schema (Azure PostgreSQL) ---\n"
        current_table = None
        for row in rows:
            table_name, column_name, data_type, is_nullable, column_default = row
            if table_name != current_table:
                if current_table is not None:
                    res += ");\n\n"
                res += f"CREATE TABLE {table_name} (\n"
                current_table = table_name
            nullable = "" if is_nullable == "YES" else " NOT NULL"
            default = f" DEFAULT {column_default}" if column_default else ""
            res += f"    {column_name} {data_type.upper()}{nullable}{default},\n"
        if current_table is not None:
            res += ");\n\n"

        # 🔴 DEPRECATED: SQLite formatting (kept for reference)
        # res = "--- Database Schema ---\n"
        # for row in rows:
        #     sql_str = row[1] if len(row) > 1 else row[0]
        #     if sql_str:
        #         res += sql_str + ";\n\n"

        SCHEMA_CACHE[cache_key] = res
        return res
    except Exception as e:
        return f"Schema Error: {str(e)}"


@mcp.tool()
def compute_portfolio_concentration(client_id: int) -> str:
    """Calculates portfolio concentration metrics based on the JSON Breakdown column."""
    global QUERY_CACHE
    check_cache_invalidation()
    
    try:
        # 🟢 AZURE POSTGRESQL CONNECTION (uses %s and quoted identifiers for case-sensitivity)
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT "Breakdown" FROM "PortfolioData" WHERE "ClientId" = %s', (client_id,))
        row = cursor.fetchone()
        conn.close()

        # 🔴 DEPRECATED: SQLite connection (kept for reference)
        # conn = sqlite3.connect(SQLITE_DB_PATH)
        # cursor = conn.cursor()
        # cursor.execute("SELECT Breakdown FROM PortfolioData WHERE ClientId = ?", (client_id,))
        # row = cursor.fetchone()
        # conn.close()

        if not row or not row[0]:
            return f"No portfolio breakdown data found for Client {client_id}."
            
        try:
            # Safely parse the JSON string (e.g., "[{'Cash': 5, 'Stocks': 60...}]")
            raw_str = row[0].replace("'", '"') 
            breakdown_list = json.loads(raw_str)
            
            breakdown = breakdown_list[0] if isinstance(breakdown_list, list) else breakdown_list
                
            res = f"--- Portfolio Concentration for Client {client_id} ---\n"
            for asset_class, percentage in breakdown.items():
                res += f"- {asset_class}: {percentage}%\n"
                if float(percentage) >= 40:
                    res += f"  ⚠️ HIGH CONCENTRATION WARNING: {asset_class} represents a significant risk.\n"
            return res
            
        except json.JSONDecodeError:
            return f"Error: Could not parse Breakdown JSON for Client {client_id}. Raw data: {row[0]}"
    except Exception as e:
        return f"Database Error: {str(e)}"

@mcp.tool()
def search_transcripts(client_id: int, query: str) -> str:
    """Semantic search over a specific client's past meeting transcripts and summaries."""
    try:
        path = os.path.join(CHROMA_DB_PATH, 'transcripts')
        if not os.path.exists(path):
            return "Error: Vector DB not initialized for transcripts."
            
        vector_store = Chroma(persist_directory=path, embedding_function=embedder)
        results = vector_store.similarity_search(query, k=3, filter={"client_id": client_id})
        
        if not results:
            return f"No transcript results found for Client {client_id} regarding '{query}'."
        return "\n".join([f"- [{res.metadata.get('source_type', 'unknown')}]: {res.page_content}" for res in results])
    except Exception as e:
        return f"Transcript Search Error: {str(e)}"

@mcp.tool()
def search_client_emails(client_id: int, query: str) -> str:
    """Semantic search over a specific client's email history."""
    try:
        path = os.path.join(CHROMA_DB_PATH, 'emails')
        if not os.path.exists(path):
            return "Error: Vector DB not initialized for emails."
            
        vector_store = Chroma(persist_directory=path, embedding_function=embedder)
        results = vector_store.similarity_search(query, k=3, filter={"client_id": client_id})
        
        if not results:
            return f"No email results found for Client {client_id} regarding '{query}'."
        return "\n".join([f"- [{res.metadata.get('source_type', 'unknown')}]: {res.page_content}" for res in results])
    except Exception as e:
        return f"Email Search Error: {str(e)}"

### MATH TOOLS ###
# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1 — compute_portfolio_metrics
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def compute_portfolio_metrics(
    holdings:       List[Dict],
    portfolio_data: Dict,
    client_details: Dict,
) -> Dict:
    """
    Compute portfolio snapshot metrics for a single client. Pure math — no database calls.

    WHEN TO CALL THIS TOOL:
      Use for any question about a client's current portfolio state:
      - Asset allocation or sector breakdown  ("What % is in bonds?")
      - Portfolio concentration / HHI         ("Is the client over-concentrated?")
      - Risk profile vs holdings mismatch     ("Does allocation match risk tolerance?")
      - Unrealized gains and losses           ("What are the P&L figures?")
      - Tax-loss harvesting opportunity       ("How much can be harvested?")
      - Wallet share / outside assets         ("What share of wealth do we manage?")

    DO NOT USE THIS TOOL FOR:
      - Portfolio return vs benchmark  → use compute_portfolio_return
      - Advisor-level aggregation      → use analyze_advisor_book
      - What-if reallocation           → use simulate_reallocation
      - Fetching data from DB          → use execute_sql

    PRE-FETCH PATTERN — always fetch data before calling this tool:
      holdings:
        SELECT "Id", "AssetClass", "Sector", "SecurityName", "Ticker",
               "MarketValue", "CostBasis", "UnrealizedPnL", "Weight", "AsOfDate"
        FROM "Holdings"
        WHERE "ClientId" = 'CLI-001'
        ORDER BY "AsOfDate" DESC

      portfolio_data:
        SELECT "TotalValue", "TotalOutsideAssetsValue", "WalletShare",
               "RiskTolerance", "RiskCategory", "BenchmarkReturn"
        FROM "PortfolioData"
        WHERE "ClientId" = 'CLI-001'

      client_details:
        SELECT "Name", "TaxSensitivity", "Persona", "AttritionRisk", "ClientSentiment"
        FROM "ClientDetails"
        WHERE "ClientId" = 'CLI-001'

    RETURNS dict with keys:
      total_aum, computed_total_cross_check, as_of_date,
      allocation_pct, sector_concentration, top_positions,
      hhi_position_level, hhi_asset_class_level, hhi_interpretation,
      wallet_share_pct, outside_assets,
      unrealized_pnl_total, pnl_breakdown, harvestable_loss,
      risk_tolerance, risk_category, mismatch_flag, mismatch_reason, equity_pct,
      tax_sensitive, client_persona,
      data_quality ("ok" | "degraded" | "blocked"),
      warnings (list[str])
    """

    W = _Warnings()

    # ── 0. Input type validation ───────────────────────────────────────────────
    # Catches LLM passing stringified JSON instead of parsed objects
    holdings, err = _check_type(holdings, list, "holdings")
    if err:
        return {"error": err, "data_quality": "blocked", "warnings": []}

    portfolio_data, err = _check_type(portfolio_data, dict, "portfolio_data")
    if err:
        W.add(err, critical=True)
    pd_ = portfolio_data or {}

    client_details, err = _check_type(client_details, dict, "client_details")
    if err:
        W.add(err)
    cd_ = client_details or {}

    # ── 0b. Empty holdings guard ───────────────────────────────────────────────
    if not holdings:
        return {
            "error": "No holdings provided. Call execute_sql on Holdings table first.",
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    # ── F1: Compute total portfolio value ──────────────────────────────────────
    computed_total = 0.0
    null_mv_count  = 0
    for h in holdings:
        mv, w = _sf(h.get("MarketValue"), "MarketValue")
        if w:
            null_mv_count += 1
        computed_total += mv

    # Batch-report MarketValue parse failures (not per-row to avoid warning floods)
    if null_mv_count:
        W.add(
            f"{null_mv_count}/{len(holdings)} holding(s) had non-numeric MarketValue — "
            "defaulted to 0. Total AUM may be understated.",
            critical=(null_mv_count == len(holdings))
        )

    if computed_total == 0.0:
        return {
            "error": "All MarketValue entries resolved to zero. Cannot compute metrics.",
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    # Use PortfolioData.TotalValue if available, otherwise use computed sum
    reported_total, w = _sf(pd_.get("TotalValue"), "TotalValue", fallback=0.0)
    if w:
        W.add(w)
    V = reported_total if reported_total > 0 else computed_total

    # Cross-check: warn if computed total drifts >5% from reported total
    if reported_total > 0:
        drift = abs(computed_total - reported_total) / reported_total * 100
        if drift > 5.0:
            W.add(
                f"Cross-check: computed ${computed_total:,.2f} vs PortfolioData.TotalValue "
                f"${reported_total:,.2f} — {drift:.1f}% drift. Holdings snapshot may be incomplete.",
                critical=True
            )

    # ── F2: Enrich holdings (local copy — never mutates input) ────────────────
    enriched = _enrich(holdings, V)

    # Warn if CostBasis is missing (affects P&L and harvestable loss accuracy)
    cb_zero = sum(1 for h in enriched if h["_cb"] == 0.0)
    if cb_zero:
        W.add(
            f"{cb_zero}/{len(enriched)} holding(s) have CostBasis=0. "
            "P&L and harvestable-loss figures may be understated.",
            critical=(cb_zero == len(enriched))
        )

    # ── F3: Asset-class allocation (% of total AUM) ────────────────────────────
    allocation_pct = _agg_pct(enriched, "_ac", V)

    # ── F4: Sector concentration (% of total AUM) ─────────────────────────────
    sector_concentration = _agg_pct(enriched, "_sec", V)

    # ── F5: HHI at two levels ──────────────────────────────────────────────────
    # Position-level HHI: individual holding weights (granular view)
    hhi_pos = _hhi([h["_w"] for h in enriched])
    # Asset-class-level HHI: aggregated weights (advisor-relevant view)
    # A portfolio of 50 tiny positions in one asset class would show low
    # position HHI but high asset-class HHI — both are reported.
    ac_weights = {ac: pct / 100 for ac, pct in allocation_pct.items()}
    hhi_ac = _hhi(list(ac_weights.values()))

    # ── Top 5 positions by market value ───────────────────────────────────────
    top_positions = sorted(
        [
            {
                "security":    h["_nm"],
                "asset_class": h["_ac"],
                "market_value": round(h["_mv"], 2),
                "weight_pct":  round(h["_w"] * 100, 2),
            }
            for h in enriched
        ],
        key=lambda x: x["market_value"],
        reverse=True,
    )[:5]

    # ── F6: Wallet share ───────────────────────────────────────────────────────
    # WalletShare = managed AUM / (managed AUM + outside assets)
    V_out, w = _sf(pd_.get("TotalOutsideAssetsValue"), "TotalOutsideAssetsValue")
    if w:
        W.add(w)

    if (V + V_out) > 0:
        wallet_share: Optional[float] = round((V / (V + V_out)) * 100, 2)
    else:
        wallet_share = None
        W.add("Wallet share is undefined: V + V_out = 0.")

    # ── F7: Unrealized P&L ─────────────────────────────────────────────────────
    # Note: PostgreSQL returns UnrealizedPnL directly but we recompute from
    # MarketValue - CostBasis to ensure consistency with our own CostBasis values.
    pnl_breakdown = []
    total_pnl = 0.0
    for h in enriched:
        pnl = h["_mv"] - h["_cb"]
        total_pnl += pnl
        pnl_breakdown.append({
            "security":       h["_nm"],
            "asset_class":    h["_ac"],
            "market_value":   round(h["_mv"], 2),
            "cost_basis":     round(h["_cb"], 2),
            "unrealized_pnl": round(pnl, 2),
            # pnl_pct is None-safe — CostBasis=0 is a known data quality issue
            "pnl_pct": (
                round((pnl / h["_cb"]) * 100, 2)
                if h["_cb"] != 0.0
                else "N/A — cost basis unavailable"
            ),
        })

    # ── F8: Harvestable loss ───────────────────────────────────────────────────
    # Only positions where MarketValue < CostBasis qualify for tax-loss harvesting
    harvestable_loss = round(
        sum(h["_cb"] - h["_mv"] for h in enriched if h["_mv"] < h["_cb"]),
        2,
    )

    # ── F12: Risk profile mismatch ─────────────────────────────────────────────
    # Flag: conservative/moderate profile + >50% equity allocation
    risk_tolerance = _norm(pd_.get("RiskTolerance"))
    risk_category  = _norm(pd_.get("RiskCategory"))

    equity_pct = round(
        sum(pct for ac, pct in allocation_pct.items() if _is_equity(ac)), 2
    )

    mismatch_flag = (
        _nk(risk_tolerance) in _CONSERVATIVE_PROFILES
        and equity_pct > 50.0
    )
    mismatch_reason = (
        f"{risk_tolerance} profile but {equity_pct:.1f}% allocated to equities "
        "(threshold: >50%). Allocation review recommended."
        if mismatch_flag
        else None
    )

    # ── Tax sensitivity ────────────────────────────────────────────────────────
    tax_sensitive = _nk(cd_.get("TaxSensitivity") or "") in _TAX_SENSITIVE_VALUES

    # ── Snapshot date passthrough ──────────────────────────────────────────────
    # Extracts AsOfDate from first holding that has one.
    # PostgreSQL returns datetime objects — str() gives "2026-04-20 00:00:00".
    raw_date = next((h.get("AsOfDate") for h in holdings if h.get("AsOfDate")), None)
    if raw_date is not None:
        # Format datetime cleanly — strip time component if midnight
        as_of_date = str(raw_date)[:10] if hasattr(raw_date, "year") else _norm(raw_date)
    else:
        as_of_date = "Not provided"

    return {
        "total_aum":                  round(V, 2),
        "computed_total_cross_check": round(computed_total, 2),
        "as_of_date":                 as_of_date,
        "allocation_pct":             allocation_pct,
        "sector_concentration":       sector_concentration,
        "top_positions":              top_positions,
        "hhi_position_level":         hhi_pos,
        "hhi_asset_class_level":      hhi_ac,
        "hhi_interpretation":         _hhi_label(hhi_ac),
        "wallet_share_pct":           wallet_share,
        "outside_assets":             round(V_out, 2),
        "unrealized_pnl_total":       round(total_pnl, 2),
        "pnl_breakdown":              pnl_breakdown,
        "harvestable_loss":           harvestable_loss,
        "risk_tolerance":             risk_tolerance,
        "risk_category":              risk_category,
        "mismatch_flag":              mismatch_flag,
        "mismatch_reason":            mismatch_reason,
        "equity_pct":                 equity_pct,
        "tax_sensitive":              tax_sensitive,
        "client_persona":             _norm(cd_.get("Persona")),
        "data_quality":               W.data_quality(),
        "warnings":                   W.list(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2 — compute_portfolio_return
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def compute_portfolio_return(
    snapshot_t0:      List[Dict],
    snapshot_t1:      List[Dict],
    benchmark_return: float,
    window:           str          = "custom",
    date_t0:          Optional[str] = None,
    date_t1:          Optional[str] = None,
) -> Dict:
    """
    Compute portfolio return and benchmark comparison from two Holdings snapshots.
    Pure math — no database calls.

    WHEN TO CALL THIS TOOL:
      - "What is the portfolio return over the last 3 months?"
      - "How does the portfolio compare to its benchmark?"
      - "What is the alpha in basis points?"
      - "Which asset classes drove or dragged returns?"

    DO NOT USE THIS TOOL FOR:
      - Single-snapshot metrics (allocation, HHI, P&L) → use compute_portfolio_metrics
      - Advisor-level aggregated performance           → use analyze_advisor_book
      - What-if reallocation                           → use simulate_reallocation

    DATA REQUIREMENT — TWO SNAPSHOTS FROM DIFFERENT DATES:
      Both snapshot_t0 (start) and snapshot_t1 (end) must be non-empty and
      from different AsOfDate values. If only one date exists in the DB for
      the requested window, this tool returns data_blocked=True with a reason.

    PRE-FETCH PATTERN:
      snapshot_t0:
        SELECT "AssetClass", "MarketValue", "AsOfDate" FROM "Holdings"
        WHERE "ClientId" = 'CLI-001' AND "AsOfDate" = '<start_date>'

      snapshot_t1:
        SELECT "AssetClass", "MarketValue", "AsOfDate" FROM "Holdings"
        WHERE "ClientId" = 'CLI-001' AND "AsOfDate" = '<end_date>'

      benchmark_return:
        SELECT "BenchmarkReturn" FROM "PortfolioData" WHERE "ClientId" = 'CLI-001'

    INPUTS:
      snapshot_t0:      Holdings rows at START of window.
      snapshot_t1:      Holdings rows at END of window.
      benchmark_return: Decimal benchmark return (e.g. 0.072 = 7.2%). Pass 0.0 if unavailable.
      window:           Label only — "1M"|"3M"|"6M"|"YTD"|"1Y"|"custom". Does not affect calc.
      date_t0:          AsOfDate of t0 snapshot (for output labelling and same-date guard).
      date_t1:          AsOfDate of t1 snapshot (for output labelling and same-date guard).

    RETURNS dict with keys:
      portfolio_return, portfolio_return_pct, benchmark_return, benchmark_return_pct,
      delta_bps, outperformed, by_asset_class, window, date_t0, date_t1,
      v0_total, v1_total, snapshots_used,
      data_blocked (bool), reason (str|None),
      data_quality, warnings
    """

    W = _Warnings()

    # ── Input type validation ──────────────────────────────────────────────────
    snapshot_t0, err = _check_type(snapshot_t0, list, "snapshot_t0")
    if err:
        return {"error": err, "data_blocked": True, "data_quality": "blocked", "warnings": []}

    snapshot_t1, err = _check_type(snapshot_t1, list, "snapshot_t1")
    if err:
        return {"error": err, "data_blocked": True, "data_quality": "blocked", "warnings": []}

    bm, w = _sf(benchmark_return, "benchmark_return", fallback=0.0)
    if w:
        W.add(w)
        W.add("benchmark_return is missing — delta_bps will be relative to 0.", critical=True)

    # ── data_blocked guards ────────────────────────────────────────────────────
    if not snapshot_t0 or not snapshot_t1:
        return {
            "data_blocked": True,
            "reason": (
                "One or both snapshots are empty. "
                "At least two Holdings snapshots from different dates are required."
            ),
            "window": window,
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    # Guard against same-date snapshots (would produce return=0 silently)
    if date_t0 and date_t1 and date_t0 == date_t1:
        return {
            "data_blocked": True,
            "reason": (
                f"Both snapshots share the same date ({date_t0}). "
                "Return calculation requires two distinct dates."
            ),
            "window": window,
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    # ── F9: Portfolio values ───────────────────────────────────────────────────
    V0 = sum(_sf(h.get("MarketValue"), "MarketValue")[0] for h in snapshot_t0)
    V1 = sum(_sf(h.get("MarketValue"), "MarketValue")[0] for h in snapshot_t1)

    if V0 == 0.0:
        return {
            "data_blocked": True,
            "reason": "snapshot_t0 total MarketValue is zero. Cannot divide for return calculation.",
            "window": window,
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    if V1 == 0.0:
        W.add("snapshot_t1 total MarketValue is zero. Portfolio return will be -100%.", critical=True)

    # ── F9: Simple return ──────────────────────────────────────────────────────
    portfolio_return = round((V1 - V0) / V0, 6)

    # ── F10: Alpha vs benchmark (basis points) ─────────────────────────────────
    # 1 basis point = 0.01%. delta_bps > 0 means outperformance.
    delta_bps = round((portfolio_return - bm) * 10000, 2)

    # ── F11: By-asset-class breakdown ─────────────────────────────────────────
    # Aggregate V0 and V1 per asset class for attribution analysis
    ac_v0: Dict[str, float] = {}
    ac_v1: Dict[str, float] = {}

    for h in snapshot_t0:
        ac = _norm(h.get("AssetClass"))
        mv, _ = _sf(h.get("MarketValue"), "MarketValue")
        ac_v0[ac] = ac_v0.get(ac, 0.0) + mv

    for h in snapshot_t1:
        ac = _norm(h.get("AssetClass"))
        mv, _ = _sf(h.get("MarketValue"), "MarketValue")
        ac_v1[ac] = ac_v1.get(ac, 0.0) + mv

    all_classes = set(ac_v0) | set(ac_v1)
    by_asset_class: Dict[str, Any] = {}

    for ac in sorted(all_classes):
        v0_c = ac_v0.get(ac, 0.0)
        v1_c = ac_v1.get(ac, 0.0)

        # Asset class present at t1 but not t0 → return undefined
        if v0_c == 0.0:
            ac_return = None
            W.add(f"Asset class '{ac}' has no t0 value — return undefined for this class.")
        else:
            ac_return = round((v1_c - v0_c) / v0_c, 6)

        # Weight uses t1 value (end-of-period weight)
        ac_weight = round((v1_c / V1) * 100, 2) if V1 > 0 else None

        by_asset_class[ac] = {
            "return":           ac_return,
            "return_pct":       round(ac_return * 100, 4) if ac_return is not None else None,
            "v0":               round(v0_c, 2),
            "v1":               round(v1_c, 2),
            "weight_at_t1_pct": ac_weight,
        }

    return {
        "portfolio_return":      portfolio_return,
        "portfolio_return_pct":  round(portfolio_return * 100, 4),
        "benchmark_return":      bm,
        "benchmark_return_pct":  round(bm * 100, 4),
        "delta_bps":             delta_bps,
        "outperformed":          delta_bps > 0,
        "by_asset_class":        by_asset_class,
        "window":                window,
        "date_t0":               date_t0 or "Not provided",
        "date_t1":               date_t1 or "Not provided",
        "snapshots_used":        2,
        "v0_total":              round(V0, 2),
        "v1_total":              round(V1, 2),
        "data_blocked":          False,
        "reason":                None,
        "data_quality":          W.data_quality(),
        "warnings":              W.list(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3 — analyze_advisor_book
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def analyze_advisor_book(
    advisor_id:   str,
    mode:         str,
    clients_data: List[Dict],
    window:       str = "6M",
) -> Dict:
    """
    Advisor-scope aggregation across a book of clients. Pure math — no database calls.

    REVIEWER NOTE: advisor_id is str not int — AdvisorDetails.AdvisorId is TEXT in PostgreSQL.

    WHEN TO CALL THIS TOOL:
      Use for any advisor-level question spanning multiple clients:
      - "What is the total AUM for advisor ADV-001?"         (mode=aum)
      - "Which clients have consolidation opportunity?"       (mode=consolidation)
      - "What is the estimated revenue for this advisor?"    (mode=revenue)
      - "How did the advisor's book perform this quarter?"   (mode=performance)

    DO NOT USE THIS TOOL FOR:
      - Single-client metrics   → use compute_portfolio_metrics
      - Single-client return    → use compute_portfolio_return
      - What-if reallocation    → use simulate_reallocation

    MODE GUIDE + PRE-FETCH PATTERN:
      mode=aum:
        clients_data keys: client_id, total_value, segment (optional)
        fetch: SELECT "ClientId", "TotalValue", "Persona" FROM "PortfolioData"
               JOIN "ClientDetails" USING ("ClientId") WHERE "AdvisorId" = 'ADV-001'

      mode=consolidation:
        clients_data keys: client_id, total_value, outside_assets_value, segment
        fetch: SELECT "ClientId", "TotalValue", "TotalOutsideAssetsValue", "Persona"
               FROM "PortfolioData" JOIN "ClientDetails" USING ("ClientId")
               WHERE "AdvisorId" = 'ADV-001'

      mode=revenue:
        clients_data keys: client_id, asset_breakdown ({asset_class: market_value}), segment
        fetch: SELECT "ClientId", "AssetClass", SUM("MarketValue") as mv
               FROM "Holdings" JOIN "ClientDetails" USING ("ClientId")
               WHERE "AdvisorId" = 'ADV-001' GROUP BY "ClientId", "AssetClass"
        Then group into: {"client_id": "CLI-001", "asset_breakdown": {"Equity": 500000, ...}}

      mode=performance:
        clients_data keys: client_id, total_value, portfolio_return (decimal), segment
        NOTE: run compute_portfolio_return per client FIRST, then pass results here.

    RETURNS dict with keys:
      advisor_id, mode, window,
      summary (mode-specific aggregates),
      per_client (list[dict] sorted by primary metric descending),
      methodology_note (fee tiers / thresholds disclosed),
      data_quality, warnings
    """

    W = _Warnings()

    # ── Input validation ───────────────────────────────────────────────────────
    clients_data, err = _check_type(clients_data, list, "clients_data")
    if err:
        return {"error": err, "data_quality": "blocked", "warnings": []}

    if not clients_data:
        return {
            "error": "clients_data is empty. Fetch advisor's clients via execute_sql first.",
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    mode = _nk(mode)
    valid_modes = {"aum", "consolidation", "revenue", "performance"}
    if mode not in valid_modes:
        return {
            "error": f"Invalid mode '{mode}'. Must be one of: {sorted(valid_modes)}.",
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    per_client: List[Dict] = []
    summary:    Dict       = {}
    methodology: str       = ""

    # ── mode=aum ───────────────────────────────────────────────────────────────
    if mode == "aum":
        total_aum = 0.0
        segment_aum: Dict[str, float] = {}

        for c in clients_data:
            tv, w = _sf(c.get("total_value"), "total_value")
            if w:
                W.add(w)
            seg = _norm(c.get("segment"))
            cid = c.get("client_id", "unknown")
            total_aum += tv
            segment_aum[seg] = segment_aum.get(seg, 0.0) + tv
            per_client.append({"client_id": cid, "segment": seg, "total_value": round(tv, 2)})

        summary = {
            "total_aum":    round(total_aum, 2),
            "client_count": len(clients_data),
            "by_segment":   {s: round(v, 2) for s, v in segment_aum.items()},
        }
        methodology = "F1: total_aum = Σ(client total_value)."

    # ── mode=consolidation ─────────────────────────────────────────────────────
    elif mode == "consolidation":
        total_opportunity = 0.0
        flagged_clients   = 0

        for c in clients_data:
            tv,    w1 = _sf(c.get("total_value"),          "total_value")
            v_out, w2 = _sf(c.get("outside_assets_value"), "outside_assets_value")
            if w1: W.add(w1)
            if w2: W.add(w2)

            cid = c.get("client_id", "unknown")
            seg = _norm(c.get("segment"))

            # F13: score = V × (1 − WalletShare) where outside_assets > threshold
            # Mathematically equivalent to: score = V × V_out / (V + V_out)
            qualifies = v_out > _CONSOLIDATION_THRESHOLD
            if qualifies and (tv + v_out) > 0:
                ws    = tv / (tv + v_out)
                score = round(tv * (1 - ws), 2)
            else:
                ws    = tv / (tv + v_out) if (tv + v_out) > 0 else None
                score = 0.0

            total_opportunity += score
            if score > 0:
                flagged_clients += 1

            per_client.append({
                "client_id":           cid,
                "segment":             seg,
                "total_value":         round(tv, 2),
                "outside_assets":      round(v_out, 2),
                "wallet_share_pct":    round(ws * 100, 2) if ws is not None else None,
                "consolidation_score": score,
                "qualifies":           qualifies,
            })

        # Sort highest opportunity first
        per_client.sort(key=lambda x: x["consolidation_score"], reverse=True)

        summary = {
            "total_consolidation_opportunity": round(total_opportunity, 2),
            "flagged_clients":                 flagged_clients,
            "total_clients":                   len(clients_data),
            "outside_assets_threshold":        _CONSOLIDATION_THRESHOLD,
        }
        methodology = (
            f"F13: score = V × (1 − WalletShare) "
            f"where outside_assets_value > ${_CONSOLIDATION_THRESHOLD:,.0f}. "
            "Clients below threshold score 0."
        )

    # ── mode=revenue ───────────────────────────────────────────────────────────
    elif mode == "revenue":
        total_revenue = 0.0
        segment_rev: Dict[str, float] = {}

        for c in clients_data:
            ab, err = _check_type(c.get("asset_breakdown"), dict, "asset_breakdown")
            if err:
                W.add(f"Client {c.get('client_id')}: {err}")
                ab = {}
            ab  = ab or {}
            cid = c.get("client_id", "unknown")
            seg = _norm(c.get("segment"))

            # F14: revenue = Σ(MV_c × fee_c) per asset class
            client_rev = sum(
                _sf(mv, "asset_mv")[0] * _fee(ac)
                for ac, mv in ab.items()
            )
            total_revenue += client_rev
            segment_rev[seg] = segment_rev.get(seg, 0.0) + client_rev

            per_client.append({
                "client_id":        cid,
                "segment":          seg,
                "revenue_estimate": round(client_rev, 2),
                "asset_breakdown":  {ac: round(_sf(mv)[0], 2) for ac, mv in ab.items()},
            })

        per_client.sort(key=lambda x: x["revenue_estimate"], reverse=True)

        summary = {
            "total_revenue_estimate": round(total_revenue, 2),
            "by_segment":             {s: round(v, 2) for s, v in segment_rev.items()},
            "fee_tiers_applied":      _FEE_TIERS,
            "default_fee_tier":       _DEFAULT_FEE,
        }
        methodology = (
            "F14: revenue = Σ(MV_c × fee_c). "
            "Fee tiers are hard-coded (see fee_tiers_applied). "
            "REVIEWER NOTE: Move to FeeTier DB table in production."
        )

    # ── mode=performance ───────────────────────────────────────────────────────
    elif mode == "performance":
        total_aum     = 0.0
        weighted_sum  = 0.0
        segment_aum:  Dict[str, float] = {}
        segment_wsum: Dict[str, float] = {}

        for c in clients_data:
            tv, w1 = _sf(c.get("total_value"),     "total_value")
            pr, w2 = _sf(c.get("portfolio_return"), "portfolio_return")
            if w1: W.add(w1)
            if w2: W.add(f"Client {c.get('client_id')}: {w2}", critical=True)

            cid = c.get("client_id", "unknown")
            seg = _norm(c.get("segment"))

            total_aum    += tv
            weighted_sum += tv * pr
            segment_aum[seg]  = segment_aum.get(seg, 0.0) + tv
            segment_wsum[seg] = segment_wsum.get(seg, 0.0) + tv * pr

            per_client.append({
                "client_id":            cid,
                "segment":              seg,
                "total_value":          round(tv, 2),
                "portfolio_return":     pr,
                "portfolio_return_pct": round(pr * 100, 4),
            })

        # F11: AUM-weighted book return
        book_return = (weighted_sum / total_aum) if total_aum > 0 else None

        # F11: per-segment AUM-weighted return
        by_segment: Dict[str, Any] = {}
        for seg in segment_aum:
            seg_r = (
                segment_wsum[seg] / segment_aum[seg]
                if segment_aum[seg] > 0
                else None
            )
            by_segment[seg] = {
                "aum":                 round(segment_aum[seg], 2),
                "aum_weighted_return": seg_r,
                "return_pct":          round(seg_r * 100, 4) if seg_r is not None else None,
            }

        per_client.sort(key=lambda x: x["portfolio_return"], reverse=True)

        summary = {
            "book_aum_weighted_return": book_return,
            "book_return_pct":          round(book_return * 100, 4) if book_return is not None else None,
            "total_aum":                round(total_aum, 2),
            "client_count":             len(clients_data),
            "by_segment":               by_segment,
            "window":                   window,
        }
        methodology = (
            "F11: book_return = Σ(V_k × R_k) / Σ(V_k). "
            "portfolio_return per client must be pre-computed using "
            "compute_portfolio_return before passing to this tool."
        )

    return {
        "advisor_id":       advisor_id,
        "mode":             mode,
        "window":           window,
        "summary":          summary,
        "per_client":       per_client,
        "methodology_note": methodology,
        "data_quality":     W.data_quality(),
        "warnings":         W.list(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4 — simulate_reallocation
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def simulate_reallocation(
    current_holdings: List[Dict],
    target_weights:   Dict[str, float],
) -> Dict:
    """
    Simulate a portfolio reallocation and compare current vs target expected return and risk.
    Pure math — no database calls.

    ⚠️ SIMULATION ONLY — results are illustrative.
      Uses hard-coded long-run capital market assumptions (see _ASSET_PARAMS above).
      F16 uses uncorrelated variance model — actual correlated risk will differ.
      Always present simulation_disclaimer to the client.

    WHEN TO CALL THIS TOOL:
      - "What happens to expected return if we shift 20% from bonds to stocks?"
      - "Show the risk/return trade-off of a proposed new allocation."
      - "What would the portfolio look like with these target weights?"

    DO NOT USE THIS TOOL FOR:
      - Live portfolio metrics   → use compute_portfolio_metrics
      - Historical return calc   → use compute_portfolio_return
      - Advisor-book aggregation → use analyze_advisor_book

    PRE-FETCH PATTERN:
      current_holdings:
        SELECT "AssetClass", "MarketValue" FROM "Holdings"
        WHERE "ClientId" = 'CLI-001'
        AND "AsOfDate" = (SELECT MAX("AsOfDate") FROM "Holdings" WHERE "ClientId" = 'CLI-001')

    INPUTS:
      current_holdings: Holdings rows. Required keys: MarketValue, AssetClass.
      target_weights:   Proposed allocation as decimals summing to 1.0 (±0.02 tolerance).
                        Example: {"Equity": 0.60, "Fixed Income": 0.30, "Cash": 0.10}

    RETURNS dict with keys:
      current_allocation, target_allocation,
      current_expected_return_pct, target_expected_return_pct,
      current_risk_sigma_pct, target_risk_sigma_pct,
      return_delta_pct_points, risk_delta_pct_points,
      return_improves, risk_increases,
      assumptions (mu/sigma values fully disclosed),
      simulation_disclaimer,
      data_quality, warnings
    """

    W = _Warnings()

    # ── Input type validation ──────────────────────────────────────────────────
    current_holdings, err = _check_type(current_holdings, list, "current_holdings")
    if err:
        return {"error": err, "data_quality": "blocked", "warnings": []}

    target_weights, err = _check_type(target_weights, dict, "target_weights")
    if err:
        return {"error": err, "data_quality": "blocked", "warnings": []}

    if not current_holdings:
        return {
            "error": "current_holdings is empty. Fetch from Holdings table via execute_sql.",
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    if not target_weights:
        return {
            "error": "target_weights is empty. Provide proposed allocation as {asset_class: weight}.",
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    # ── Validate and normalise target weights ──────────────────────────────────
    tw_vals: Dict[str, float] = {}
    for ac, w in target_weights.items():
        val, warn_msg = _sf(w, f"target_weights[{ac}]")
        if warn_msg:
            W.add(warn_msg)
        if val < 0:
            W.add(f"Negative weight for '{ac}' ({val}). Setting to 0.", critical=True)
            val = 0.0
        tw_vals[_norm(ac)] = val

    weight_sum = sum(tw_vals.values())
    if abs(weight_sum - 1.0) > 0.02:
        return {
            "error": (
                f"target_weights sum to {weight_sum:.4f}, expected 1.0 (±0.02). "
                "Adjust weights before calling this tool."
            ),
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    # Renormalise to exactly 1.0 if within tolerance (e.g. 0.99 or 1.01)
    if weight_sum != 1.0 and weight_sum > 0:
        tw_vals = {ac: round(w / weight_sum, 6) for ac, w in tw_vals.items()}
        W.add(f"target_weights renormalised from {weight_sum:.4f} to 1.0.")

    # ── Current allocation from holdings ──────────────────────────────────────
    V_total = sum(_sf(h.get("MarketValue"), "MarketValue")[0] for h in current_holdings)

    if V_total == 0.0:
        return {
            "error": "All current_holdings MarketValue entries resolve to zero.",
            "data_quality": "blocked",
            "warnings": W.list(),
        }

    ac_mv: Dict[str, float] = {}
    for h in current_holdings:
        ac = _norm(h.get("AssetClass"))
        mv, _ = _sf(h.get("MarketValue"), "MarketValue")
        ac_mv[ac] = ac_mv.get(ac, 0.0) + mv

    current_weights = {ac: mv / V_total for ac, mv in ac_mv.items()}

    # Warn on asset classes not in the assumptions table (will use default mu/sigma)
    all_classes = set(current_weights) | set(tw_vals)
    for ac in all_classes:
        if _nk(ac) not in _ASSET_PARAMS:
            W.add(
                f"Asset class '{ac}' not in assumptions table — "
                f"using default mu={_DEFAULT_PARAMS['mu']}, sigma={_DEFAULT_PARAMS['sigma']}."
            )

    # ── F15: Expected return ───────────────────────────────────────────────────
    # E[R_p] = Σ(w_c × μ_c)
    def _expected_return(weights: Dict[str, float]) -> float:
        return sum(w * _params(ac)["mu"] for ac, w in weights.items())

    # ── F16: Portfolio volatility (uncorrelated approximation) ────────────────
    # σ_p = sqrt(Σ(w_c² × σ_c²))
    # REVIEWER NOTE: This ignores cross-asset correlations. Actual risk will differ.
    # Correlated model requires a covariance matrix — not implemented in this version.
    def _portfolio_sigma(weights: Dict[str, float]) -> float:
        return _math.sqrt(
            sum((w ** 2) * (_params(ac)["sigma"] ** 2) for ac, w in weights.items())
        )

    current_er    = round(_expected_return(current_weights), 6)
    current_sigma = round(_portfolio_sigma(current_weights), 6)
    target_er     = round(_expected_return(tw_vals), 6)
    target_sigma  = round(_portfolio_sigma(tw_vals), 6)

    return_delta = round((target_er    - current_er)    * 100, 4)  # percentage points
    risk_delta   = round((target_sigma - current_sigma) * 100, 4)  # percentage points

    # Disclose all assumptions used in this simulation
    assumptions_used = {ac: _params(ac) for ac in sorted(all_classes)}

    return {
        "current_allocation": {
            ac: {
                "weight_pct":   round(w * 100, 2),
                "market_value": round(ac_mv.get(ac, 0), 2),
            }
            for ac, w in current_weights.items()
        },
        "target_allocation": {
            ac: {"weight_pct": round(w * 100, 2)}
            for ac, w in tw_vals.items()
        },
        "current_expected_return":     current_er,
        "current_expected_return_pct": round(current_er    * 100, 4),
        "target_expected_return":      target_er,
        "target_expected_return_pct":  round(target_er     * 100, 4),
        "current_risk_sigma":          current_sigma,
        "current_risk_sigma_pct":      round(current_sigma * 100, 4),
        "target_risk_sigma":           target_sigma,
        "target_risk_sigma_pct":       round(target_sigma  * 100, 4),
        "return_delta_pct_points":     return_delta,
        "risk_delta_pct_points":       risk_delta,
        "return_improves":             return_delta > 0,
        "risk_increases":              risk_delta   > 0,
        "assumptions": {
            "note": (
                "μ = long-run expected annual return. "
                "σ = annual volatility. "
                "F16 uses uncorrelated variance approximation — "
                "actual correlated portfolio risk will differ."
            ),
            "values": assumptions_used,
        },
        "simulation_disclaimer": (
            "⚠️ SIMULATION ONLY. Results use hard-coded long-run capital market assumptions "
            "and an uncorrelated variance model. They are NOT a guarantee of future performance. "
            "This output must be presented with full disclosure to the client."
        ),
        "data_quality": W.data_quality(),
        "warnings":     W.list(),
    }

if __name__ == "__main__":
    print("Starting Aeon Wealth MCP Server...", file=sys.stderr)
    mcp.run(transport="stdio")