# src/scripts/ingest_db.py
import os
import sys
# import sqlite3  # 🔴 DEPRECATED: SQLite - Replaced with Azure PostgreSQL (kept for reference)
import pandas as pd
import glob

# 🟢 AZURE POSTGRESQL MIGRATION: Import shared connection module
# Add project root to sys.path so `src.db.pg_connection` is importable when
# running this script directly (e.g., `python src/scripts/ingest_db.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.db.pg_connection import (
    get_sqlalchemy_engine,
    AZURE_PG_HOST,
    AZURE_PG_DATABASE,
    AZURE_PG_SCHEMA,
)


def build_local_database():
    # Setup paths
    base_dir = os.path.dirname(__file__)
    raw_data_dir = os.path.abspath(os.path.join(base_dir, '../../data_raw'))
    # 🔴 DEPRECATED: SQLite path (kept for reference)
    # db_path = os.path.abspath(os.path.join(base_dir, '../../data_local/aeon_db.sqlite'))

    # 🔴 DEPRECATED: SQLite directory creation (kept for reference)
    # os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 🟢 AZURE POSTGRESQL: Create SQLAlchemy engine for pandas to_sql
    try:
        engine = get_sqlalchemy_engine()
        print(f"🔌 Connected to Azure PostgreSQL: {AZURE_PG_HOST}/{AZURE_PG_DATABASE}")
    except Exception as e:
        print(f"❌ Failed to connect to Azure PostgreSQL: {e}")
        return

    # 🔴 DEPRECATED: SQLite connection (kept for reference)
    # conn = sqlite3.connect(db_path)
    # print(f"🔌 Connected to local database at: {db_path}")

    # Find the Excel file
    excel_files = glob.glob(os.path.join(raw_data_dir, "*.xlsx"))
    
    if not excel_files:
        print("⚠️ No .xlsx files found in data_raw/. Please add your Postgres export.")
        return

    for file_path in excel_files:
        filename = os.path.basename(file_path)
        print(f"📄 Processing Excel workbook: {filename}")
        
        try:
            # Read all sheets into a dictionary of DataFrames
            all_sheets = pd.read_excel(file_path, sheet_name=None)
            
            for sheet_name, df in all_sheets.items():
                # Clean up the sheet name to use as the SQL table name
                # E.g., 'public_ClientDetails' becomes 'ClientDetails'
                table_name = sheet_name.replace("public_", "").strip()
                
                # 🟢 Push the DataFrame to Azure PostgreSQL via SQLAlchemy engine
                df.to_sql(table_name, engine, schema=AZURE_PG_SCHEMA, if_exists='replace', index=False)
                # 🔴 DEPRECATED: Push the DataFrame to SQLite (kept for reference)
                # df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"  ✅ Successfully ingested table: {AZURE_PG_SCHEMA}.{table_name} ({len(df)} rows)")
                
        except Exception as e:
            print(f"❌ Failed to ingest {filename}: {e}")

    # 🟢 Dispose PostgreSQL engine
    engine.dispose()
    # 🔴 DEPRECATED: SQLite connection close (kept for reference)
    # conn.close()
    print("\n🎉 Azure PostgreSQL Database Ingestion Complete! Ready for Agent Queries.")

if __name__ == "__main__":
    # build_local_database()
    pass