# src/mcp_server/server.py
import os

# 🛑 CRITICAL: Suppress aggressive C++ gRPC logging in the background MCP process
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
os.environ["CHROMA_TELEMETRY_IMPL"] = "None"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import sqlite3
import chromadb
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import json

# 🟢 LOCAL EMBEDDING MIGRATION: Swap AWS Bedrock for local HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
mcp = FastMCP("AeonWealthMCP")

# --- Bulletproof Pathing ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
LOCAL_DATA_DIR = os.path.join(BASE_DIR, 'data_local')
SQLITE_DB_PATH = os.path.join(LOCAL_DATA_DIR, 'aeon_db.sqlite') 
CHROMA_DB_PATH = os.path.join(LOCAL_DATA_DIR, 'chroma_db')
MODEL_PATH = os.path.join(BASE_DIR, 'local_embedding_model')

# 🔌 Initialize the Local Offline Embedder
embedder = HuggingFaceEmbeddings(model_name=MODEL_PATH)

# ⚡ GLOBAL CACHE STATE
QUERY_CACHE = {}
SCHEMA_CACHE = {}
LAST_DB_MTIME = 0

def check_cache_invalidation():
    """Checks file metadata. Wipes cache if DB has been updated."""
    global LAST_DB_MTIME, QUERY_CACHE, SCHEMA_CACHE
    try:
        current_mtime = os.path.getmtime(SQLITE_DB_PATH)
        if current_mtime != LAST_DB_MTIME:
            QUERY_CACHE.clear()
            SCHEMA_CACHE.clear()
            LAST_DB_MTIME = current_mtime
    except OSError:
        pass

@mcp.tool()
def execute_sql(query: str) -> str:
    """Execute a read-only SQL query against the Aeon Wealth database."""
    global QUERY_CACHE
    check_cache_invalidation()

    if query in QUERY_CACHE:
        print(f"\n      ⚡ [CACHE HIT - Instant Return]:\n{query}\n", file=sys.stderr)
        return QUERY_CACHE[query]

    print(f"\n      🟦 [MCP SQL Executing]:\n{query}\n", file=sys.stderr)
    try:
        if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE")):
            return "Error: Only SELECT queries are allowed."

        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            res = "No results found."
        else:
            res = " | ".join(columns) + "\n"
            res += "-" * len(res) + "\n"
            for row in rows:
                res += " | ".join(str(val) if val is not None else "NULL" for val in row) + "\n"
                
        QUERY_CACHE[query] = res
        return res
    except Exception as e:
        print(f"\n      ❌ [MCP SQL ERROR]: {str(e)}\n", file=sys.stderr)
        return f"SQL Error: {str(e)}"

@mcp.tool()
def get_database_schema(table_names: list[str] = None) -> str:
    """Returns the exact CREATE TABLE schemas for the requested tables."""
    global SCHEMA_CACHE
    check_cache_invalidation()
    
    cache_key = str(table_names)
    if cache_key in SCHEMA_CACHE:
        print(f"\n      ⚡ [SCHEMA CACHE HIT]: {cache_key}\n", file=sys.stderr)
        return SCHEMA_CACHE[cache_key]

    print(f"\n      🗄️ [FETCHING SCHEMA]: {cache_key}\n", file=sys.stderr)
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        if table_names and len(table_names) > 0:
            placeholders = ','.join(['?'] * len(table_names))
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({placeholders})", tuple(table_names))
        else:
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
            
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "No schema found for the requested tables."
            
        res = "--- Database Schema ---\n"
        for row in rows:
            sql_str = row[1] if len(row) > 1 else row[0]
            if sql_str:
                res += sql_str + ";\n\n"
                
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
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT Breakdown FROM PortfolioData WHERE ClientId = ?", (client_id,))
        row = cursor.fetchone()
        conn.close()
        
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
    print("Starting Aeon Wealth MCP Server...")
    mcp.run(transport="stdio")