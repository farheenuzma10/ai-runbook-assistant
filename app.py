import ollama
import faiss
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index('faiss_index.bin')
with open('documents.pkl', 'rb') as f:
    documents = pickle.load(f)  

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask(request: QuestionRequest):
    query_embedding = embedding_model.encode([request.question])

    distances, indices = index.search(np.array(query_embedding).astype('float32'), 2)

    retrieved_docs = []

    for idx in indices[0]:
        retrieved_docs.append(documents[idx])
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    Answer only using the context below.

    Context:
    {context}

    Question:
    {request.question}
    """

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        "question": request.question,
        "answer": answer
    }