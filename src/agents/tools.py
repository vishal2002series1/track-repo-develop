# src/agents/tools.py
import asyncio
import atexit
import os
import sys
import threading
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _debug(message: str) -> None:
    print(f"[AEON MCP CLIENT] {message}", file=sys.stderr, flush=True)

# --- 🔍 Robust Path Resolution for MCP Server ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
path_option_1 = os.path.join(BASE_DIR, 'src', 'mcp', 'sqlite_server.py')
path_option_2 = os.path.join(BASE_DIR, 'src', 'mcp_server', 'server.py')

if os.path.exists(path_option_1):
    SERVER_SCRIPT_PATH = path_option_1
else:
    SERVER_SCRIPT_PATH = path_option_2

# --- Persistent MCP Client ---
class _PersistentMCPClient:
    """Keeps one FastMCP stdio session alive across repeated tool calls."""

    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self._lock = threading.RLock()
        self._loop = None
        self._thread = None
        self._session = None
        self._stdio_cm = None
        self._session_cm = None

    async def _connect(self):
        if self._session is not None:
            return

        _debug(f"Starting FastMCP server process: {self.server_script_path}")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script_path],
        )
        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        _debug("Initializing MCP session")
        await self._session.initialize()
        _debug("MCP session initialized")

    async def _disconnect(self):
        _debug("Closing MCP session")
        session_cm = self._session_cm
        stdio_cm = self._stdio_cm
        self._session = None
        self._session_cm = None
        self._stdio_cm = None

        if session_cm is not None:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if stdio_cm is not None:
            try:
                await stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass

    def _ensure_started(self):
        with self._lock:
            if self._loop is not None and self._thread is not None and self._thread.is_alive():
                return

            ready = threading.Event()
            error_holder = {}

            def _runner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                try:
                    _debug("Starting MCP client event loop")
                    loop.run_until_complete(self._connect())
                except Exception as exc:
                    error_holder["exc"] = exc
                    _debug(f"MCP client startup failed: {exc}")
                    ready.set()
                    return
                ready.set()
                loop.run_forever()
                loop.run_until_complete(self._disconnect())
                loop.close()
                _debug("MCP client event loop stopped")

            self._thread = threading.Thread(target=_runner, name="aeon-mcp-client", daemon=True)
            self._thread.start()
            ready.wait()
            if "exc" in error_holder:
                self._loop = None
                self._thread = None
                raise RuntimeError(f"Failed to initialize MCP client: {error_holder['exc']}")

    async def _call_tool(self, tool_name: str, args: dict) -> str:
        if self._session is None:
            await self._connect()

        _debug(f"Calling tool '{tool_name}' with args={args}")
        result = await self._session.call_tool(tool_name, arguments=args)
        content = getattr(result, "content", None) or []
        text_chunks = [item.text for item in content if hasattr(item, "text") and item.text]
        if text_chunks:
            _debug(f"Tool '{tool_name}' returned {len(text_chunks)} text chunk(s)")
            return "\n".join(text_chunks)
        _debug(f"Tool '{tool_name}' returned non-text content")
        return str(result)

    def call_tool(self, tool_name: str, args: dict, timeout: float = 120.0) -> str:
        self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(self._call_tool(tool_name, args), self._loop)
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            _debug(f"Tool '{tool_name}' failed on first attempt: {exc}. Reconnecting MCP session.")
            with self._lock:
                restart = asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)
                restart.result(timeout=5)
                reconnect = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
                reconnect.result(timeout=30)
            future = asyncio.run_coroutine_threadsafe(self._call_tool(tool_name, args), self._loop)
            return future.result(timeout=timeout)

    def close(self):
        with self._lock:
            if self._loop is None:
                return
            loop = self._loop
            thread = self._thread
            stop_future = asyncio.run_coroutine_threadsafe(self._disconnect(), loop)
            try:
                stop_future.result(timeout=5)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
            self._loop = None
            self._thread = None
        if thread is not None:
            thread.join(timeout=2)


_MCP_CLIENT = _PersistentMCPClient(SERVER_SCRIPT_PATH)
atexit.register(_MCP_CLIENT.close)

def run_mcp_tool_sync(tool_name: str, args: dict) -> str:
    """Execute MCP tools through one long-lived FastMCP stdio session."""
    _debug(f"Dispatching synchronous tool call for '{tool_name}'")
    return _MCP_CLIENT.call_tool(tool_name, args)

# --- Pure LangChain Adapters for our MCP Tools ---

@tool
def execute_sql(query: str) -> str:
    """
    Execute a read-only SQL query against the Aeon Wealth relational database.
    Use this to fetch deterministic client facts, portfolio data, meetings, and compliance flags.
    Use schema aeon2 (never public) when explicitly qualifying table names.
    Tables available: AdvisorDetails, AdvisorNotes, ClientActions, ClientDetails, ClientDocuments, Emails, Holdings,
    MarketHighlights, PortfolioData, Transcripts, and more. Always use get_database_schema tool first to see exact column names and data types before writing SQL queries with this tool.
    """
    return run_mcp_tool_sync("execute_sql", {"query": query})

@tool
def compute_portfolio_concentration(client_id: int) -> str:
    """
    Deterministic Compute: Calculates portfolio concentration metrics for a specific client.
    Returns top position percentage, Herfindahl-Hirschman Index (HHI), sector concentration, 
    and flags any concentrated low-basis positions. 
    ALWAYS use this tool instead of calculating concentration via SQL.
    """
    return run_mcp_tool_sync("compute_portfolio_concentration", {"client_id": client_id})

@tool
def search_transcripts(client_id: int, query: str) -> str:
    """
    Semantic search over a specific client's past meeting transcripts and call notes.
    Use this to find qualitative information: life events, sentiment, family dynamics, or unprompted goals.
    
    Args:
        client_id (int): The ID of the client to search.
        query (str): The semantic question (e.g., 'Did the client mention estate planning or grandkids?')
    """
    return run_mcp_tool_sync("search_transcripts", {"client_id": client_id, "query": query})

@tool
def search_client_emails(client_id: int, query: str) -> str:
    """
    Semantic search over a specific client's email history.
    Use this to find asynchronous requests, sent documents, or recent questions from the client.
    
    Args:
        client_id (int): The ID of the client to search.
        query (str): The semantic question (e.g., 'Did the client email about the new trust documents?')
    """
    return run_mcp_tool_sync("search_client_emails", {"client_id": client_id, "query": query})

@tool
def get_database_schema(table_names: list[str] = None) -> str:
    """
    Returns the exact CREATE TABLE schemas for the requested tables.
    ALWAYS use this tool before writing SQL queries using execute_sql to ensure you use the exact correct column names.
    """
    args = {"table_names": table_names} if table_names else {}
    return run_mcp_tool_sync("get_database_schema", args)

# ─────────────────────────────────────────────────────────────────────────────
# WF_002 Pure-Math Tool Adapters
# ─────────────────────────────────────────────────────────────────────────────
#
# Agent workflow:
#   Step 1 → get_database_schema(...)  : confirm column names
#   Step 2 → execute_sql(...)          : fetch rows from Azure PostgreSQL
#   Step 3 → parse rows into dicts     : agent's responsibility
#   Step 4 → call math tool(dicts)     : pure computation, no DB calls
# ─────────────────────────────────────────────────────────────────────────────

@tool
def compute_portfolio_metrics(
    holdings: list[dict],
    portfolio_data: dict,
    client_details: dict,
) -> str:
    """
    Compute portfolio snapshot metrics for a single client. Pure math — no database calls.

    WHEN TO USE:
      Any single-client portfolio state question:
      - Asset allocation or sector breakdown  ("What % is in bonds?")
      - Portfolio concentration / HHI         ("Is the client over-concentrated?")
      - Risk profile vs holdings mismatch     ("Does allocation match risk tolerance?")
      - Unrealized P&L                        ("What are the P&L figures?")
      - Tax-loss harvesting opportunity       ("How much can be harvested?")
      - Wallet share / outside assets         ("What share of wealth do we manage?")

    DO NOT USE FOR:
      - Return vs benchmark  → use compute_portfolio_return
      - Advisor aggregation  → use analyze_advisor_book
      - What-if reallocation → use simulate_reallocation

    PRE-FETCH PATTERN — call execute_sql first, then pass results here:
      holdings:
        SELECT "Id", "AssetClass", "Sector", "SecurityName", "Ticker",
               "MarketValue", "CostBasis", "UnrealizedPnL", "Weight", "AsOfDate"
        FROM "Holdings" WHERE "ClientId" = 'CLI-001' ORDER BY "AsOfDate" DESC

      portfolio_data:
        SELECT "TotalValue", "TotalOutsideAssetsValue", "WalletShare",
               "RiskTolerance", "RiskCategory", "BenchmarkReturn"
        FROM "PortfolioData" WHERE "ClientId" = 'CLI-001'

      client_details:
        SELECT "Name", "TaxSensitivity", "Persona", "AttritionRisk"
        FROM "ClientDetails" WHERE "ClientId" = 'CLI-001'

    Args:
        holdings:       List of dicts from Holdings rows.
        portfolio_data: Dict from one PortfolioData row. Pass {} if unavailable.
        client_details: Dict from one ClientDetails row. Pass {} if unavailable.
    """
    return run_mcp_tool_sync("compute_portfolio_metrics", {
        "holdings":       holdings,
        "portfolio_data": portfolio_data,
        "client_details": client_details,
    })


@tool
def compute_portfolio_return(
    snapshot_t0: list[dict],
    snapshot_t1: list[dict],
    benchmark_return: float,
    window: str = "custom",
    date_t0: str = None,
    date_t1: str = None,
) -> str:
    """
    Compute portfolio return and benchmark comparison from two Holdings snapshots.
    Pure math — no database calls.

    WHEN TO USE:
      - "What is the portfolio return over the last 3 months?"
      - "How does the portfolio compare to its benchmark?"
      - "What is the alpha in basis points?"
      - "Which asset classes drove or dragged returns?"

    DO NOT USE FOR:
      - Single-snapshot metrics → use compute_portfolio_metrics
      - Advisor aggregation     → use analyze_advisor_book

    DATA REQUIREMENT — TWO SNAPSHOTS FROM DIFFERENT DATES:
      If only one AsOfDate exists in the DB for the requested window,
      this tool returns data_blocked=True. Do not call with identical snapshots.

    PRE-FETCH PATTERN:
      snapshot_t0:
        SELECT "AssetClass", "MarketValue", "AsOfDate" FROM "Holdings"
        WHERE "ClientId" = 'CLI-001' AND "AsOfDate" = '<start_date>'

      snapshot_t1:
        SELECT "AssetClass", "MarketValue", "AsOfDate" FROM "Holdings"
        WHERE "ClientId" = 'CLI-001' AND "AsOfDate" = '<end_date>'

      benchmark_return:
        SELECT "BenchmarkReturn" FROM "PortfolioData" WHERE "ClientId" = 'CLI-001'

    Args:
        snapshot_t0:      Holdings rows at START of window.
        snapshot_t1:      Holdings rows at END of window.
        benchmark_return: Decimal return e.g. 0.072 = 7.2%. Pass 0.0 if unavailable.
        window:           Label only — "1M"|"3M"|"6M"|"YTD"|"1Y"|"custom".
        date_t0:          AsOfDate string of t0 snapshot (for labelling + same-date guard).
        date_t1:          AsOfDate string of t1 snapshot (for labelling + same-date guard).
    """
    return run_mcp_tool_sync("compute_portfolio_return", {
        "snapshot_t0":      snapshot_t0,
        "snapshot_t1":      snapshot_t1,
        "benchmark_return": benchmark_return,
        "window":           window,
        "date_t0":          date_t0,
        "date_t1":          date_t1,
    })


@tool
def analyze_advisor_book(
    advisor_id: str,
    mode: str,
    clients_data: list[dict],
    window: str = "6M",
) -> str:
    """
    Advisor-scope aggregation across a book of clients. Pure math — no database calls.

    WHEN TO USE:
      Advisor-level questions spanning multiple clients:
      - "What is the total AUM for advisor ADV-001?"         (mode=aum)
      - "Which clients have consolidation opportunity?"       (mode=consolidation)
      - "What is the estimated revenue for this advisor?"    (mode=revenue)
      - "How did the advisor's book perform this quarter?"   (mode=performance)

    DO NOT USE FOR:
      - Single-client metrics → use compute_portfolio_metrics
      - Single-client return  → use compute_portfolio_return
      - What-if reallocation  → use simulate_reallocation

    NOTE: advisor_id is TEXT e.g. 'ADV-001' — matches AdvisorDetails.AdvisorId column.

    MODE GUIDE + PRE-FETCH PATTERN:
      mode=aum:
        clients_data keys: client_id, total_value, segment
        fetch: SELECT cd."ClientId", pd."TotalValue", cd."Persona"
               FROM "PortfolioData" pd JOIN "ClientDetails" cd USING ("ClientId")
               WHERE cd."AdvisorId" = 'ADV-001'

      mode=consolidation:
        clients_data keys: client_id, total_value, outside_assets_value, segment
        fetch: SELECT cd."ClientId", pd."TotalValue", pd."TotalOutsideAssetsValue", cd."Persona"
               FROM "PortfolioData" pd JOIN "ClientDetails" cd USING ("ClientId")
               WHERE cd."AdvisorId" = 'ADV-001'

      mode=revenue:
        clients_data keys: client_id, asset_breakdown ({asset_class: market_value}), segment
        fetch: SELECT h."ClientId", h."AssetClass", SUM(h."MarketValue") as mv
               FROM "Holdings" h JOIN "ClientDetails" cd USING ("ClientId")
               WHERE cd."AdvisorId" = 'ADV-001' GROUP BY h."ClientId", h."AssetClass"
        Then structure as: {"client_id": "CLI-001", "asset_breakdown": {"Equity": 500000}}

      mode=performance:
        clients_data keys: client_id, total_value, portfolio_return (decimal), segment
        NOTE: run compute_portfolio_return per client FIRST, then aggregate here.

    Args:
        advisor_id:   Advisor ID string e.g. 'ADV-001'.
        mode:         One of "aum" | "consolidation" | "revenue" | "performance".
        clients_data: List of per-client dicts (schema varies by mode — see above).
        window:       Label only — default "6M".
    """
    return run_mcp_tool_sync("analyze_advisor_book", {
        "advisor_id":   advisor_id,
        "mode":         mode,
        "clients_data": clients_data,
        "window":       window,
    })


@tool
def simulate_reallocation(
    current_holdings: list[dict],
    target_weights: dict,
) -> str:
    """
    Simulate a portfolio reallocation — compare current vs target expected return and risk.
    Pure math — no database calls.

    ⚠️ SIMULATION ONLY. Uses hard-coded long-run capital market assumptions.
    Always surface the simulation_disclaimer from the output to the client.

    WHEN TO USE:
      - "What if we shift 20% from bonds to equities?"
      - "Show risk/return trade-off of a proposed new allocation."
      - "What would expected return be with these target weights?"

    DO NOT USE FOR:
      - Live portfolio metrics   → use compute_portfolio_metrics
      - Historical return calc   → use compute_portfolio_return
      - Advisor-book aggregation → use analyze_advisor_book

    PRE-FETCH PATTERN:
      current_holdings:
        SELECT "AssetClass", "MarketValue" FROM "Holdings"
        WHERE "ClientId" = 'CLI-001'
        AND "AsOfDate" = (SELECT MAX("AsOfDate") FROM "Holdings" WHERE "ClientId" = 'CLI-001')

    Args:
        current_holdings: Holdings rows. Required keys: AssetClass, MarketValue.
        target_weights:   Proposed allocation as decimals summing to 1.0 (±0.02 tolerance).
                          Example: {"Equity": 0.60, "Fixed Income": 0.30, "Cash": 0.10}
    """
    return run_mcp_tool_sync("simulate_reallocation", {
        "current_holdings": current_holdings,
        "target_weights":   target_weights,
    })

# 🛑 Cleaned up AEON_TOOLS: Only these 5 tools exist in our universe now.
# AEON_TOOLS = [
#     execute_sql, 
#     compute_portfolio_concentration, 
#     search_transcripts, 
#     get_database_schema,
#     search_client_emails
# ]

AEON_TOOLS = [
    # DB / search tools
    execute_sql,
    get_database_schema,
    search_transcripts,
    search_client_emails,
    # WF_002 pure-math tools
    compute_portfolio_metrics,
    compute_portfolio_return,
    analyze_advisor_book,
    simulate_reallocation,
]