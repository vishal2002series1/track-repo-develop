# src/agents/tools.py
import os
import sys
import concurrent.futures
import traceback
from langchain_core.tools import tool

# --- 🔍 Robust Path Resolution for MCP Server ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
path_option_1 = os.path.join(BASE_DIR, 'src', 'mcp', 'sqlite_server.py')
path_option_2 = os.path.join(BASE_DIR, 'src', 'mcp_server', 'server.py')

if os.path.exists(path_option_1):
    SERVER_SCRIPT_PATH = path_option_1
else:
    SERVER_SCRIPT_PATH = path_option_2

# --- 🛡️ ISOLATED MCP WORKER (WITH DEBUG TRAPS) ---
def _mcp_process_worker(tool_name: str, args: dict, server_script_path: str) -> str:
    """
    Runs entirely in a separate OS process. 
    """
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _execute():
        print(f"\n⏳ [DEBUG WORKER] Starting stdio process for {tool_name}...", flush=True)
        try:
            # Added "-u" to force unbuffered output on Windows
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-u", server_script_path], 
            )
            async with stdio_client(server_params) as (read, write):
                print(f"✅ [DEBUG WORKER] Process spawned. Initializing session...", flush=True)
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print(f"📡 [DEBUG WORKER] Session initialized! Executing {tool_name}...", flush=True)
                    
                    result = await session.call_tool(tool_name, arguments=args)
                    
                    print(f"🎉 [DEBUG WORKER] Success! Returning data.", flush=True)
                    return result.content[0].text
                    
        except Exception as e:
            # Trap the error INSIDE the worker and force print it
            print(f"\n❌ [CRITICAL WORKER ERROR]: {str(e)}", flush=True)
            traceback.print_exc()
            return f"SYSTEM ERROR: The tool failed to execute. Reason: {str(e)}"

    return asyncio.run(_execute())

def run_mcp_tool_sync(tool_name: str, args: dict) -> str:
    """Safely execute MCP tool by outsourcing it to an isolated process."""
    print(f"\n🟦 [AGENT ATTEMPTING TOOL CALL] {tool_name} | Args: {args}", flush=True)
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_mcp_process_worker, tool_name, args, SERVER_SCRIPT_PATH)
            # Added a 45-second timeout so it cannot hang infinitely
            return future.result(timeout=45) 
    except concurrent.futures.TimeoutError:
        print(f"\n⏰ [TIMEOUT ERROR] Tool {tool_name} hung for more than 45 seconds!", flush=True)
        return "SYSTEM ERROR: The tool execution timed out."
    except Exception as e:
        print(f"\n❌ [EXECUTOR ERROR]: {e}", flush=True)
        return f"SYSTEM ERROR: Process execution failed. {e}"

# ... keep your existing @tool definitions below this line ...

# --- Pure LangChain Adapters for our MCP Tools ---
# Agent-> Tool -> Langchain (L1) -> Tool (L2) - > MCP(Repo) (L3) : Tool Call = L1+L2//+L3

@tool
def execute_sql(query: str) -> str:
    """
    Execute a read-only SQL query against the Aeon Wealth relational database.
    Use this to fetch deterministic client facts, portfolio data, meetings, and compliance flags.
    Tables available: AIGroupInsight, AdvisorClients, AdvisorCoaching, AdvisorDetails, 
    AdvisorPerformance, ClientDetails, CollaborationHub, ComplianceHub, Email, EmailInsight, 
    EmailReply, MarketHighlights, NextBestAction, NextBestActionLandingPag, OpenOpportunities, 
    PortfolioData, PortfolioSimulator, SmartInsights, SocialListening, Transcript, 
    TranscriptInsights, TranscriptSummary, UpcomingClientMeetings, chat_memory.
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

# 🛑 Cleaned up AEON_TOOLS: Only these 5 tools exist in our universe now.
AEON_TOOLS = [
    execute_sql, 
    compute_portfolio_concentration, 
    search_transcripts, 
    get_database_schema,
    search_client_emails
]