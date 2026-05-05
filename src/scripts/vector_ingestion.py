# src/scripts/vector_ingestion.py
import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# print("🔌 Initializing Local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
# embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("🔌 Initializing Local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
embedder = HuggingFaceEmbeddings(model_name="./local_embedding_model")  # Use local path for the embedding model

# 🟢 BULLETPROOF PATHING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../data_raw'))
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../data_local/chroma_db'))

# Ensure the database directory exists
os.makedirs(DB_DIR, exist_ok=True)

def ingest_data():
    print("🚀 Starting Semantic Data Ingestion...\n")
    
    # Exact file names based on the exported data
    files = {
        'transcripts': 'postgres_export_20260420_101547.xlsx - public_Transcript.csv',
        'emails': 'postgres_export_20260420_101547.xlsx - public_Email.csv',
        'email_replies': 'postgres_export_20260420_101547.xlsx - public_EmailReply.csv',
        'transcript_summaries': 'postgres_export_20260420_101547.xlsx - public_TranscriptSummary.csv'
    }

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # 1. Ingest Transcripts
    t_path = os.path.join(DATA_DIR, files['transcripts'])
    if os.path.exists(t_path):
        print(f"   📚 Processing Transcripts...")
        df = pd.read_csv(t_path).fillna("") # Handle empty rows
        df['semantic_text'] = "Date: " + df['Date'].astype(str) + " | Type: " + df['CallType'] + " | Content: " + df['Transcript']
        
        docs = DataFrameLoader(df, page_content_column="semantic_text").load()
        split_docs = text_splitter.split_documents(docs)
        
        # 🛑 CRITICAL: Inject ClientId so agents don't hallucinate data across clients
        for doc in split_docs:
            doc.metadata = {"client_id": int(doc.metadata.get("ClientId", 0)), "source_type": "transcript"}
        
        Chroma.from_documents(split_docs, embedder, persist_directory=os.path.join(DB_DIR, 'transcripts'))
        print(f"   ✅ Saved {len(split_docs)} transcript chunks to ChromaDB.\n")
    else:
        print(f"   ❌ Missing Transcripts file at {t_path}\n")

    # 2. Ingest Emails
    e_path = os.path.join(DATA_DIR, files['emails'])
    if os.path.exists(e_path):
        print(f"   📧 Processing Emails...")
        df = pd.read_csv(e_path).fillna("")
        df['semantic_text'] = "Date: " + df['Date'].astype(str) + " | Subject: " + df['Subject'] + " | Body: " + df['Body']
        
        docs = DataFrameLoader(df, page_content_column="semantic_text").load()
        split_docs = text_splitter.split_documents(docs)
        for doc in split_docs:
            doc.metadata = {"client_id": int(doc.metadata.get("ClientId", 0)), "source_type": "email"}
        
        Chroma.from_documents(split_docs, embedder, persist_directory=os.path.join(DB_DIR, 'emails'))
        print(f"   ✅ Saved {len(split_docs)} email chunks to ChromaDB.\n")
    else:
        print(f"   ❌ Missing Emails file at {e_path}\n")

    # 3. Ingest Email Replies (Combine into the 'emails' vector collection)
    er_path = os.path.join(DATA_DIR, files['email_replies'])
    if os.path.exists(er_path):
        print(f"   📨 Processing Email Replies...")
        df = pd.read_csv(er_path).fillna("")
        df['semantic_text'] = "Date: " + df['Date'].astype(str) + " | Subject: " + df['Subject'] + " | Body: " + df['Body']
        
        docs = DataFrameLoader(df, page_content_column="semantic_text").load()
        split_docs = text_splitter.split_documents(docs)
        for doc in split_docs:
            doc.metadata = {"client_id": int(doc.metadata.get("ClientId", 0)), "source_type": "email_reply"}
        
        Chroma.from_documents(split_docs, embedder, persist_directory=os.path.join(DB_DIR, 'emails')) 
        print(f"   ✅ Saved {len(split_docs)} email reply chunks to ChromaDB.\n")
    else:
        print(f"   ❌ Missing Email Replies file at {er_path}\n")

    # 4. Ingest Transcript Summaries (Combine into the 'transcripts' vector collection)
    ts_path = os.path.join(DATA_DIR, files['transcript_summaries'])
    if os.path.exists(ts_path):
        print(f"   📝 Processing Transcript Summaries...")
        df = pd.read_csv(ts_path).fillna("")
        df['semantic_text'] = "Summary: " + df['Summary']
        
        docs = DataFrameLoader(df, page_content_column="semantic_text").load()
        split_docs = text_splitter.split_documents(docs)
        for doc in split_docs:
            doc.metadata = {"client_id": int(doc.metadata.get("ClientId", 0)), "source_type": "transcript_summary"}
        
        Chroma.from_documents(split_docs, embedder, persist_directory=os.path.join(DB_DIR, 'transcripts'))
        print(f"   ✅ Saved {len(split_docs)} transcript summary chunks to ChromaDB.\n")
    else:
        print(f"   ❌ Missing Transcript Summaries file at {ts_path}\n")

if __name__ == "__main__":
    ingest_data()
    print("🎉 All vector data ingested successfully! The database is ready.")