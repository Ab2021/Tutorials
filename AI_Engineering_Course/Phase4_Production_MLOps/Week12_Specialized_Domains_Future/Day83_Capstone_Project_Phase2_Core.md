# Day 83: Capstone Project Phase 2 - Implementation (MVP)
## Core Concepts & Theory

### Building the Engine

**Goal:** Turn the architecture into running code.
**Focus:** Core functionality (Ingest -> Retrieve -> Generate).
**Philosophy:** "Make it work, then make it right, then make it fast."

### 1. Ingestion Pipeline Implementation

**Steps:**
1.  **Load:** Read PDF bytes.
2.  **Partition:** Use `unstructured` to break into elements (Title, Text, Table).
3.  **Chunk:** Group elements into semantic chunks (500 tokens).
4.  **Embed:** Call OpenAI `text-embedding-3-small`.
5.  **Upsert:** Write to Qdrant with metadata (`doc_id`, `page_num`).

### 2. Retrieval Implementation

**Steps:**
1.  **Embed Query:** Same model as ingestion.
2.  **Search:** Cosine similarity.
3.  **Filter:** Apply `user_id` filter.
4.  **Re-rank:** (Optional for MVP) Use Cross-Encoder.

### 3. Generation Implementation (LangGraph)

**Nodes:**
- **Retrieve:** Fetch docs.
- **Grade:** Check if docs are relevant.
- **Generate:** Call GPT-4o with context.
- **Cite:** Ensure answer references the chunks.

### 4. Frontend Integration

**Stream:**
- Use Server-Sent Events (SSE) to stream tokens to the UI.
- **Citations:** Send citation metadata alongside the text stream.

### 5. Dockerization

**Container:**
- `Dockerfile` for API.
- `docker-compose.yml` to spin up Qdrant, Postgres, and API.

### 6. Handling Dirty Data

**Reality:** PDFs have headers, footers, page numbers.
**Cleaning:**
- Regex to remove "Page X of Y".
- Merge hyphenated words ("in- formation" -> "information").

### 7. MVP Trade-offs

- **No Hybrid Search:** Stick to dense vector search for speed.
- **No Async Ingestion:** Process file in-request (for small files) to simplify debugging.
- **No Auth:** Hardcode `user_id=1` for initial testing.

### 8. Summary

**Implementation Strategy:**
1.  **Script First:** Write `ingest.py` and `query.py` scripts to verify logic.
2.  **API Second:** Wrap scripts in FastAPI endpoints.
3.  **UI Third:** Connect Streamlit or React UI.
4.  **Iterate:** Fix parsing errors as they appear.

### Next Steps
In the Deep Dive, we will write the actual Ingestion Script, the Qdrant Search logic, and the LangGraph workflow code.
