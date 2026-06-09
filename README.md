# AI Runbook Assistant (RAG-based GenAI System)

## 🚀 Overview
This project is a Retrieval-Augmented Generation (RAG) based AI assistant that helps in troubleshooting and operations by answering questions using internal runbooks.

It combines:
- FastAPI for backend APIs
- Sentence Transformers for embeddings
- FAISS for vector similarity search
- LLM (Ollama / Llama3) for response generation

---

## 🏗️ Architecture

User Question
   ↓
FastAPI API (/ask)
   ↓
Sentence Transformer (Embeddings)
   ↓
FAISS Vector Search
   ↓
Relevant Runbooks Retrieved
   ↓
LLM (Ollama / Llama 3)
   ↓
Final Answer

---

## ⚙️ Tech Stack
- Python
- FastAPI
- FAISS
- SentenceTransformers
- Ollama (Llama 3)
- GitHub Actions (CI/CD)

---

## 📌 Features
- Semantic search over runbooks
- RAG-based answer generation
- REST API with FastAPI
- Swagger UI for testing
- CI pipeline using GitHub Actions

---

## 📡 API Endpoints

### Health Check
GET /
### Ask Question

POST /ask
Request:
```json
{
  "question": "How do I fix high CPU usage?"
}

Response:

{
  "question": "How do I fix high CPU usage?",
  "answer": "..."
}