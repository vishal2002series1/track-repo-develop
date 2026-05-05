# src/scripts/ingest_db.py
import os
import sqlite3
import pandas as pd
import glob

def build_local_database():
    # Setup paths
    base_dir = os.path.dirname(__file__)
    raw_data_dir = os.path.abspath(os.path.join(base_dir, '../../data_raw'))
    db_path = os.path.abspath(os.path.join(base_dir, '../../data_local/aeon_db.sqlite'))

    # Ensure directories exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to SQLite (this creates the file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    print(f"🔌 Connected to local database at: {db_path}")

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
                
                # Push the DataFrame to SQLite
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"  ✅ Successfully ingested table: {table_name} ({len(df)} rows)")
                
        except Exception as e:
            print(f"❌ Failed to ingest {filename}: {e}")

    conn.close()
    print("\n🎉 Local Database Generation Complete! Ready for Agent Queries.")

if __name__ == "__main__":
    build_local_database()