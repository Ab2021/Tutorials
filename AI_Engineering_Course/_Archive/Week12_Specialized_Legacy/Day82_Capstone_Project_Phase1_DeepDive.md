# Day 82: Capstone Project Phase 1 - Planning & Architecture
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Technical Design Doc (TDD) Snippet

Writing a professional engineering doc.

```markdown
# TDD: Enterprise Knowledge Agent

## 1. Overview
A RAG-based agent allowing financial analysts to query internal documents.

## 2. Goals
- Reduce research time by 50%.
- Support tabular data extraction.

## 3. Architecture
[Frontend] -> [Load Balancer] -> [API Service]
                                    |
                                    v
                              [Orchestrator] <-> [LLM]
                                    |
                                    v
                              [Retriever] <-> [Vector DB]

## 4. Data Flow (Ingestion)
1. User uploads PDF.
2. API saves to S3.
3. API pushes job to Redis Queue.
4. Worker pulls job -> Parses PDF (Unstructured) -> Chunks -> Embeds -> Upserts to Qdrant.

## 5. Data Flow (Chat)
1. User sends query.
2. Orchestrator rewrites query (HyDE).
3. Retriever fetches top 10 chunks.
4. Re-ranker selects top 5.
5. LLM generates answer with citations.
```

### 2. API Interface Definition (Pydantic)

Defining the contract.

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    filters: Optional[dict] = Field(default=None, description="Metadata filters")

class Citation(BaseModel):
    document_id: str
    page_number: int
    text_snippet: str
    score: float

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    latency_ms: float
    token_usage: dict

class DocumentUpload(BaseModel):
    filename: str
    content_type: str
    file_size: int
```

### 3. Project Structure Setup

Standard Python production layout.

```text
capstone_agent/
├── api/
│   ├── main.py          # FastAPI entrypoint
│   ├── routes/          # API endpoints
│   └── dependencies.py  # Auth, DB sessions
├── core/
│   ├── agent.py         # LangGraph workflow
│   ├── llm.py           # LLM client wrapper
│   └── config.py        # Env vars
├── ingestion/
│   ├── parser.py        # PDF parsing logic
│   ├── chunker.py       # Semantic chunking
│   └── embedder.py      # Embedding logic
├── database/
│   ├── vector.py        # Qdrant client
│   └── sql.py           # Postgres models
├── tests/
│   ├── unit/
│   └── integration/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### 4. Database Schema (SQLAlchemy)

Defining the relational models.

```python
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    role = Column(String) # 'admin', 'analyst'

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True) # UUID
    filename = Column(String)
    s3_key = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime)
```
