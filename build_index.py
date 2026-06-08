import faiss
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Reading runbooks...")

with open('runbooks.txt', 'r', encoding='utf-8') as f:
    text = f.read()

documents = [ 
    doc.strip() 
    for doc in text.split("------------------------------------------------") 
    if doc.strip()]
print(f"Found {len(documents)} documents.")

print("Generating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)
dimension = embeddings.shape[1]
print(f"Embedding dimension: {dimension}")
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))
faiss.write_index(index, 'faiss_index.bin')
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)
print("Index created successfully.")
