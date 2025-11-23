# Day 82: Capstone Project Phase 1 - Planning & Architecture
## Core Concepts & Theory

### The Capstone Goal

**Objective:** Build a production-ready "Enterprise Knowledge Agent".
**Requirements:**
- **Multi-Modal:** Handle PDF (Text + Images).
- **Agentic:** Use tools (Search, Calculator).
- **Secure:** RBAC and Audit Logs.
- **Scalable:** Dockerized and ready for K8s.

### 1. Problem Definition

**Scenario:**
- A financial firm has 10,000 PDF reports (Earnings, Analyst Notes).
- Analysts spend hours searching for "What was Apple's revenue in Q3 vs Q4?".
- **Solution:** An AI Agent that can search docs, extract data, perform calculations, and cite sources.

### 2. Requirements Gathering

**Functional:**
- Users can upload PDFs.
- Users can chat with the agent.
- Agent must cite the exact page number.
- Agent must be able to compare data across documents.

**Non-Functional:**
- **Latency:** < 5 seconds for simple queries.
- **Accuracy:** Zero hallucination on numbers.
- **Security:** Only authorized users can access specific docs.

### 3. System Architecture

**Components:**
1.  **Frontend:** React/Next.js Chat UI.
2.  **API Gateway:** FastAPI (Auth, Rate Limiting).
3.  **Orchestrator:** LangGraph (State Management).
4.  **Ingestion Service:** Unstructured.io (PDF Parsing) -> Chunking -> Embedding.
5.  **Storage:**
    - **Vector DB:** Qdrant (Embeddings + Metadata).
    - **SQL DB:** Postgres (Users, Chat History, Structured Data).
    - **Object Store:** S3 (Raw PDFs).
6.  **Model:** GPT-4o (Reasoning) + Text-Embedding-3-Small.

### 4. Stack Selection

**Why Qdrant?**
- Fast, Rust-based, supports Hybrid Search + Metadata Filtering.

**Why LangGraph?**
- We need a cyclic workflow: `Retrieve -> Grade -> Generate -> (Reflect) -> Generate`.

**Why FastAPI?**
- Async support, auto-generated Swagger docs, easy integration with Pydantic.

### 5. API Design

**Endpoints:**
- `POST /upload`: Upload PDF. Triggers background ingestion.
- `POST /chat`: Send message. Returns stream.
- `GET /history`: Get past conversations.
- `POST /feedback`: User thumbs up/down.

### 6. Data Model

**SQL Schema:**
- `Users`: id, email, role.
- `Documents`: id, filename, s3_path, owner_id.
- `Chats`: id, user_id, timestamp.
- `Messages`: id, chat_id, role, content, citations.

**Vector Schema:**
- `Payload`: text, page_number, document_id, image_caption.

### 7. Summary

**Phase 1 Checklist:**
1.  **Define Goal:** Enterprise Knowledge Agent.
2.  **Requirements:** Multi-modal, Secure, Accurate.
3.  **Architecture:** Microservices (Ingestion, API, DB).
4.  **Stack:** FastAPI, Qdrant, LangGraph, GPT-4o.
5.  **Schema:** SQL + Vector design.

### Next Steps
In the Deep Dive, we will write the Technical Design Doc (TDD), define the API Interface (Pydantic), and set up the Project Structure.
