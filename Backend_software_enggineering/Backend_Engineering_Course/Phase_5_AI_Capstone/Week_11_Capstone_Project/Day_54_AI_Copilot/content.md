# Day 54: AI Copilot Service

## 1. The Brain

This service has two jobs:
1.  **Indexer**: Listen to Kafka, embed docs, save to Qdrant.
2.  **Chatbot**: Answer user questions using Qdrant.

---

## 2. The Indexer (Worker)

*   **Trigger**: `doc_updates` topic.
*   **Action**:
    1.  Read message: `{"doc_id": 1, "content": "..."}`.
    2.  Split content into chunks.
    3.  Embed chunks (OpenAI `text-embedding-3-small`).
    4.  Upsert to Qdrant.

## 3. The Chatbot (API)

*   **Endpoint**: `POST /chat`.
*   **Payload**: `{"query": "What is the project deadline?", "doc_id": 1}`.
*   **Flow**:
    1.  Embed query.
    2.  Search Qdrant (Filter by `doc_id`).
    3.  Retrieve top 3 chunks.
    4.  Call GPT-4 with context.

---

## 4. LangChain Integration

We will use LangChain to glue this together.
*   `KafkaConsumer` -> `RecursiveCharacterTextSplitter` -> `Qdrant`.

---

## 5. Summary

Today we added intelligence.
*   **Kafka**: Decoupled ingestion.
*   **Qdrant**: Long-term memory.
*   **RAG**: Context-aware answers.

**Tomorrow (Day 55)**: We put it all together. **Integration & Polish**.
