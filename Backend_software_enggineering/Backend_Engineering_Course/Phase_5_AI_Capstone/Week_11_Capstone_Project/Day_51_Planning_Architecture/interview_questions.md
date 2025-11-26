# Day 51: Interview Questions & Answers

## Conceptual Questions

### Q1: Why Microservices for this project?
**Answer:**
*   **Learning**: To demonstrate knowledge of distributed systems.
*   **Scaling**: `Collab Service` (WebSockets) needs different scaling rules than `AI Service` (GPU/API heavy).
*   **Isolation**: If `AI Service` crashes, users can still edit docs.

### Q2: How do you handle Real-Time Conflicts? (Two users type at the same time)
**Answer:**
*   **Last Write Wins (LWW)**: Simple but loses data.
*   **Operational Transformation (OT)**: Used by Google Docs. Complex. Transforms index positions.
*   **CRDT (Conflict-free Replicated Data Types)**: Used by Figma/Atom. Mathematically guarantees convergence.
*   *For this Capstone*: We will use a simplified **Last Write Wins** or **Locking** for simplicity, but acknowledge OT/CRDT is better for prod.

### Q3: Why Kafka for indexing?
**Answer:**
*   **Decoupling**: `Doc Service` shouldn't wait for `AI Service` to embed the text (slow).
*   **Reliability**: If `AI Service` is down, Kafka holds the `DocUpdated` event until it recovers.

---

## Scenario-Based Questions

### Q4: You need to search for documents by "Title" (Exact) and "Content" (Semantic). How?
**Answer:**
*   **Hybrid Search**.
*   **Title**: Postgres `ILIKE` or Full Text Search.
*   **Content**: Qdrant Vector Search.
*   **Combine**: Return results from both (or use Qdrant for both if title is indexed as keyword).

### Q5: How do you secure the WebSocket connection?
**Answer:**
*   **Handshake Auth**: Pass JWT in the query param `ws://api/collab?token=xyz`.
*   **Validate**: `Collab Service` calls `Auth Service` (or checks public key) to verify token.
*   **Reject**: If invalid, close connection immediately.

---

## Behavioral / Role-Specific Questions

### Q6: A PM wants to add "Video Chat" to the app. How does this change the architecture?
**Answer:**
*   **WebRTC**: Video shouldn't go through our backend (too much bandwidth).
*   **Signaling Server**: We use our WebSocket server to exchange SDP (Session Description Protocol) packets.
*   **P2P**: The actual video stream goes Peer-to-Peer between users.
