from sentence_transformers import SentenceTransformer

print("📦 Packaging local model...")
# This pulls the model from your Mac's cache and saves it to a visible folder
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./local_embedding_model')
print("✅ Model successfully saved to the 'local_embedding_model' folder!")