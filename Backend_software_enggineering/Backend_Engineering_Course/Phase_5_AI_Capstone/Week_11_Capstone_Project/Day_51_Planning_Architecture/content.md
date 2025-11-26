# Day 51: Capstone Planning & Architecture

## 1. The Project: "DocuMind"

We will build a **Real-Time Collaborative Document Editor with AI Powers**.
Think **Google Docs + Notion AI**.

### 1.1 Features
1.  **Auth**: Signup/Login (JWT).
2.  **Documents**: CRUD Docs.
3.  **Real-Time**: Multiple users editing at once (WebSockets).
4.  **AI Copilot**: "Summarize this doc", "Fix grammar", "Chat with doc" (RAG).
5.  **Search**: Semantic search across all docs.

---

## 2. Architecture

We will use a **Microservices Architecture** (simplified for learning).

### 2.1 Services
1.  **Gateway (Nginx/Traefik)**: Entry point.
2.  **Auth Service**: User management.
3.  **Doc Service**: Document CRUD (Postgres).
4.  **Collab Service**: WebSockets for real-time syncing (Redis Pub/Sub).
5.  **AI Service**: LLM & RAG (Qdrant + OpenAI).

### 2.2 Communication
*   **Sync**: HTTP (Gateway -> Auth/Doc).
*   **Async**: Kafka (Doc Service -> AI Service for indexing).

---

## 3. Tech Stack

*   **Language**: Python (FastAPI).
*   **DB**: Postgres (Data), Redis (Cache/PubSub), Qdrant (Vectors).
*   **Broker**: Kafka (Events).
*   **AI**: OpenAI API + LangChain.
*   **Deploy**: Docker Compose.

---

## 4. Database Schema

### 4.1 Users
*   `id`, `email`, `password_hash`.

### 4.2 Documents
*   `id`, `owner_id`, `title`, `content` (Text), `created_at`.

### 4.3 Embeddings (Qdrant)
*   `vector`, `payload: {doc_id, chunk_text}`.

---

## 5. Summary

Today we drew the blueprint.
*   **Goal**: Build a complex, modern backend.
*   **Plan**: 4 Services, Event-Driven, AI-Integrated.

**Tomorrow (Day 52)**: We start coding. **Core API & Database**.
