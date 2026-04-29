Steps to run the repo locally :

In root directory create local env

1. pip install -r requirements.txt


Setup local db to store agents and workflows

2. python seed_db.py

Ingest excel based data into sqllite and chromadb vectordb

3. python src/scripts/ingest_db.py

4. python src/scripts/vector_ingestion.py

Run the FastAPI swagger

5. uvicorn main:app --reload

Wait for app to fully load

Open Swagger "http://localhost:8000/docs#"


Run the Streamlit app

6. streamlit run streamlit_app.py

Make sure that API_BASE = "http://127.0.0.1:8000/api" // API Swagger url
