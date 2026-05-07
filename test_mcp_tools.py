# test_mcp_tools.py
"""
Interactive test script for each MCP tool.
Run from the project root with the venv activated:

    python test_mcp_tools.py

You will be prompted to enter inputs for each tool.
Press Enter at any prompt to accept the shown default value.
"""

import sys
import time
import traceback


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SEP  = "─" * 60


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  🧪 {title}")
    print(SEP)


def show_result(label: str, output: str, elapsed: float) -> None:
    lines = output.strip().splitlines()
    preview = "\n    ".join(lines[:10])
    if len(lines) > 10:
        preview += f"\n    ... ({len(lines) - 10} more lines)"
    print(f"  {label}")
    print(f"  ⏱  {elapsed:.2f}s")
    print(f"  Output preview:\n    {preview}")


def prompt(label: str, default: str, examples: list[str]) -> str:
    """Prompt the user for input, showing a default and examples."""
    print(f"  📝 {label}")
    print(f"     Default : {default}")
    print(f"     Examples: {' | '.join(examples)}")
    raw = input("  > ").strip()
    value = raw if raw else default
    print(f"  → Using: {value}")
    return value


def prompt_int(label: str, default: int, examples: list[int]) -> int:
    """Prompt for an integer input."""
    ex_str = [str(e) for e in examples]
    raw = prompt(label, str(default), ex_str)
    try:
        return int(raw)
    except ValueError:
        print(f"  ⚠  Invalid integer '{raw}', using default {default}")
        return default


def prompt_list(label: str, default: list[str], examples: list[str]) -> list[str]:
    """Prompt for a comma-separated list of table names."""
    default_str = ", ".join(default)
    raw = prompt(label, default_str, examples)
    return [t.strip() for t in raw.split(",") if t.strip()]


def run_test(name: str, fn) -> bool:
    section(name)
    t0 = time.perf_counter()
    try:
        output = fn()
        elapsed = time.perf_counter() - t0
        show_result(PASS, str(output), elapsed)
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  {FAIL} ({elapsed:.2f}s)")
        print(f"  Error: {exc}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# Import tools (boots the persistent MCP client)
# ─────────────────────────────────────────────

print("\n🔌 Importing MCP tools (this starts the FastMCP server process)...")
t_import = time.perf_counter()

from src.agents.tools import (
    execute_sql,
    get_database_schema,
    compute_portfolio_concentration,
    search_transcripts,
    search_client_emails,
)

print(f"   Import done in {time.perf_counter() - t_import:.2f}s\n")


# ─────────────────────────────────────────────
# TEST 1 — get_database_schema (all tables)
# ─────────────────────────────────────────────

def test_schema_all():
    print("  ℹ  No input needed — fetches schema for ALL tables in aeon2.")
    output = get_database_schema.func()
    assert output and "CREATE TABLE" in output, "Expected CREATE TABLE in output"
    return output[:600]


# ─────────────────────────────────────────────
# TEST 2 — get_database_schema (specific tables)
# ─────────────────────────────────────────────

def test_schema_specific():
    tables = prompt_list(
        label    = "Table names (comma-separated)",
        default  = ["PortfolioData"],
        examples = ["Client", "PortfolioData, Client", "Transcript"],
    )
    output = get_database_schema.func(tables)
    assert output, "Empty response from get_database_schema"
    return output[:600]


# ─────────────────────────────────────────────
# TEST 3 — execute_sql (custom SELECT)
# ─────────────────────────────────────────────

def test_execute_sql_basic():
    query = prompt(
        label    = "SQL SELECT query",
        default  = 'SELECT "Id", "ClientId" FROM "PortfolioData" LIMIT 3;',
        examples = [
            'SELECT "Id", "Name" FROM "Client" LIMIT 5;',
            'SELECT COUNT(*) FROM "PortfolioData";',
            'SELECT "ClientId", "TotalValue" FROM "PortfolioData" ORDER BY "TotalValue" DESC LIMIT 5;',
        ],
    )
    output = execute_sql.func(query)
    assert output, "Empty response from execute_sql"
    return output


# ─────────────────────────────────────────────
# TEST 4 — execute_sql (schema rewrite: public → aeon2)
# ─────────────────────────────────────────────

def test_execute_sql_schema_rewrite():
    query = prompt(
        label    = "SQL query using 'public' schema (will be rewritten to aeon2)",
        default  = 'SELECT "Id" FROM public."PortfolioData" LIMIT 2;',
        examples = [
            'SELECT "Id" FROM public."Client" LIMIT 2;',
            'SELECT * FROM "public"."PortfolioData" LIMIT 1;',
        ],
    )
    output = execute_sql.func(query)
    assert output and "SQL Error" not in output, f"Schema rewrite may have failed: {output}"
    return output


# ─────────────────────────────────────────────
# TEST 5 — execute_sql (read-only guard)
# ─────────────────────────────────────────────

def test_execute_sql_readonly_guard():
    query = prompt(
        label    = "Non-SELECT query to test the read-only guard",
        default  = 'DELETE FROM "PortfolioData" WHERE "Id" = -999;',
        examples = [
            'DROP TABLE "Client";',
            'INSERT INTO "PortfolioData" ("Id") VALUES (0);',
            'UPDATE "PortfolioData" SET "TotalValue" = 0 WHERE "Id" = -1;',
        ],
    )
    output = execute_sql.func(query)
    assert "Only SELECT queries are allowed" in output, (
        f"Read-only guard did not trigger. Got: {output}"
    )
    return output


# ─────────────────────────────────────────────
# TEST 6 — compute_portfolio_concentration
# ─────────────────────────────────────────────

def test_compute_concentration():
    client_id = prompt_int(
        label    = "Client ID",
        default  = 2,
        examples = [1, 2, 3, 5],
    )
    output = compute_portfolio_concentration.func(client_id=client_id)
    assert output and "Database Error" not in output, f"Tool error: {output}"
    return output


# ─────────────────────────────────────────────
# TEST 7 — search_transcripts
# ─────────────────────────────────────────────

def test_search_transcripts():
    client_id = prompt_int(
        label    = "Client ID",
        default  = 2,
        examples = [1, 2, 3, 5],
    )
    query = prompt(
        label    = "Search query (semantic / natural language)",
        default  = "investment goals retirement",
        examples = [
            "risk tolerance discussion",
            "tax planning strategy",
            "estate planning concerns",
            "market volatility reaction",
        ],
    )
    output = search_transcripts.func(client_id=client_id, query=query)
    assert output and "Search Error" not in output, f"Tool error: {output}"
    return output


# ─────────────────────────────────────────────
# TEST 8 — search_client_emails
# ─────────────────────────────────────────────

def test_search_client_emails():
    client_id = prompt_int(
        label    = "Client ID",
        default  = 2,
        examples = [1, 2, 3, 5],
    )
    query = prompt(
        label    = "Search query (semantic / natural language)",
        default  = "portfolio review meeting",
        examples = [
            "quarterly report follow-up",
            "new investment opportunity",
            "account statement request",
            "rebalancing recommendation",
        ],
    )
    output = search_client_emails.func(client_id=client_id, query=query)
    assert output and "Search Error" not in output, f"Tool error: {output}"
    return output


# ─────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────

TESTS = [
    ("get_database_schema — all tables",               test_schema_all),
    ("get_database_schema — specific table(s)",        test_schema_specific),
    ("execute_sql — custom SELECT",                    test_execute_sql_basic),
    ("execute_sql — public→aeon2 schema rewrite",      test_execute_sql_schema_rewrite),
    ("execute_sql — read-only guard (non-SELECT)",     test_execute_sql_readonly_guard),
    ("compute_portfolio_concentration — custom client", test_compute_concentration),
    ("search_transcripts — custom client + query",     test_search_transcripts),
    ("search_client_emails — custom client + query",   test_search_client_emails),
]

passed = 0
failed = 0

for name, fn in TESTS:
    ok = run_test(name, fn)
    if ok:
        passed += 1
    else:
        failed += 1

print(f"\n{SEP}")
print(f"  Results: {passed}/{len(TESTS)} passed   |   {failed} failed")
print(SEP)

sys.exit(0 if failed == 0 else 1)
