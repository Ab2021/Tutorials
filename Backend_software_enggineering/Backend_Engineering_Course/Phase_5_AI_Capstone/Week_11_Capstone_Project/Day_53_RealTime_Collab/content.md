# Day 53: Real-Time Collaboration Service

## 1. The Challenge

User A and User B are editing the same doc.
*   User A types "Hello".
*   User B should see "Hello" instantly.
*   If User A is on Server 1 and User B is on Server 2, how do they talk?

---

## 2. Architecture

### 2.1 WebSocket Endpoint
`ws://api/collab/{doc_id}`
*   Clients connect to a specific document room.

### 2.2 Redis Pub/Sub
*   **Channel**: `doc:{doc_id}`.
*   **Flow**:
    1.  Server 1 receives "Hello" from User A.
    2.  Server 1 publishes to Redis channel `doc:123`.
    3.  Server 2 (subscribed to `doc:123`) receives message.
    4.  Server 2 pushes "Hello" to User B.

---

## 3. Implementation

### 3.1 Connection Manager
A class to track active sockets.
`active_connections: Dict[doc_id, List[WebSocket]]`.

### 3.2 Persistence
When should we save to Postgres?
*   **Option A**: On every keystroke (Too slow).
*   **Option B**: Debounce (Save 5s after last edit).
*   **Option C**: Save on disconnect.

We will use **Option B** (Debounce) + **Redis** as the temporary source of truth.

---

## 4. Summary

Today we connected the users.
*   **WebSockets**: The pipe.
*   **Redis**: The bridge between servers.
*   **Debounce**: The strategy to save DB load.

**Tomorrow (Day 54)**: We add the brain. **AI Copilot Service**.
