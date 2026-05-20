# src/db/pg_connection.py
"""
🟢 AZURE POSTGRESQL CONNECTION MODULE
=====================================
Centralized PostgreSQL connection configuration for AEON Backend.
Used by:
  - MCP Server (src/mcp_server/server.py)
  - Data Ingestion (src/scripts/ingest_db.py)

Required environment variables in .env:
    AZURE_PG_HOST=your-server.postgres.database.azure.com
    AZURE_PG_DATABASE=your_database_name        # e.g., 'postgres'
    AZURE_PG_USER=your_admin_username
    AZURE_PG_PASSWORD=your_password
    AZURE_PG_PORT=5432  (optional, defaults to 5432)
    AZURE_PG_SSLMODE=require  (optional, defaults to require)
    AZURE_PG_SCHEMA=aeon2  (optional, defaults to 'aeon2')
"""

import os
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# 🟢 AZURE POSTGRESQL CONNECTION CONFIGURATION
AZURE_PG_HOST = os.getenv("AZURE_PG_HOST")
AZURE_PG_DATABASE = os.getenv("AZURE_PG_DATABASE", "postgres")
AZURE_PG_USER = os.getenv("AZURE_PG_USER")
AZURE_PG_PASSWORD = os.getenv("AZURE_PG_PASSWORD")
AZURE_PG_PORT = os.getenv("AZURE_PG_PORT", "5432")
AZURE_PG_SSLMODE = os.getenv("AZURE_PG_SSLMODE", "require")  # Azure PostgreSQL requires SSL
AZURE_PG_SCHEMA = os.getenv("AZURE_PG_SCHEMA", "aeon2")      # Schema where AEON tables live


def get_pg_connection():
    """
    Create a new psycopg2 PostgreSQL connection with SSL required for Azure.
    Sets the search_path so unqualified table names resolve to AZURE_PG_SCHEMA.

    Returns:
        psycopg2.connection: A new database connection.
    """
    conn = psycopg2.connect(
        host=AZURE_PG_HOST,
        database=AZURE_PG_DATABASE,
        user=AZURE_PG_USER,
        password=AZURE_PG_PASSWORD,
        port=AZURE_PG_PORT,
        sslmode=AZURE_PG_SSLMODE,
        # 🟢 Set search_path so queries resolve unqualified table names from AZURE_PG_SCHEMA
        options=f"-c search_path={AZURE_PG_SCHEMA}",
    )
    return conn


def get_pg_connection_string():
    """Build SQLAlchemy-compatible PostgreSQL connection string."""
    return (
        f"postgresql+psycopg2://{AZURE_PG_USER}:{AZURE_PG_PASSWORD}"
        f"@{AZURE_PG_HOST}:{AZURE_PG_PORT}/{AZURE_PG_DATABASE}"
        f"?sslmode={AZURE_PG_SSLMODE}"
    )


def get_sqlalchemy_engine():
    """Create a SQLAlchemy engine (required for pandas DataFrame.to_sql)."""
    return create_engine(get_pg_connection_string())


def test_connection():
    """Test the Azure PostgreSQL connection and print server version."""
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        conn.close()
        print(f"✅ Azure PostgreSQL Connection Successful!")
        print(f"   Server Version: {version}")
        return True
    except Exception as e:
        print(f"❌ Azure PostgreSQL Connection Failed: {e}")
        return False


if __name__ == "__main__":
    print("🔌 Testing Azure PostgreSQL Connection...")
    test_connection()
