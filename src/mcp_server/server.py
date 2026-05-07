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

if __name__ == "__main__":
    print("Starting Aeon Wealth MCP Server...", file=sys.stderr)
    mcp.run(transport="stdio")