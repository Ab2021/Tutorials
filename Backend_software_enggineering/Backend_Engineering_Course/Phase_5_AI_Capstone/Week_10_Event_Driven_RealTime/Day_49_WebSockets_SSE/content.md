# Day 49: Real-Time Web (WebSockets & SSE)

## 1. The Need for Speed

HTTP is Request-Response. The server speaks only when spoken to.
How do we build a Chat App or Stock Ticker?

### 1.1 Old School: Polling
*   **Client**: "Any new messages?" (Every 1s).
*   **Server**: "No."
*   **Waste**: 99% of requests are empty.

### 1.2 Long Polling
*   **Client**: "Any new messages?"
*   **Server**: *Waits* (hangs the connection) until a message arrives. Then replies.
*   **Better**, but still high overhead.

---

## 2. WebSockets (Bidirectional)

*   **Protocol**: `ws://` or `wss://`.
*   **Handshake**: Starts as HTTP, upgrades to TCP Socket.
*   **Flow**: Persistent connection. Server pushes to Client. Client pushes to Server.
*   **Use Case**: Chat, Multiplayer Games.

---

## 3. Server-Sent Events (SSE) (Unidirectional)

*   **Protocol**: Standard HTTP.
*   **Flow**: Server keeps connection open and streams text data (`Content-Type: text/event-stream`).
*   **Direction**: Server -> Client only.
*   **Use Case**: News Feeds, Notifications, ChatGPT streaming response.

---

## 4. Scaling Challenges

*   **Stateful**: A WebSocket connection is tied to a specific server.
*   **Problem**: If User A is on Server 1 and User B is on Server 2, how do they chat?
*   **Solution**: **Redis Pub/Sub**.
    1.  User A sends msg to Server 1.
    2.  Server 1 publishes to Redis channel `chat_room`.
    3.  Server 2 subscribes to Redis. Receives msg.
    4.  Server 2 pushes to User B.

---

## 5. Summary

Today we broke the request-response cycle.
*   **WebSockets**: Two-way street.
*   **SSE**: One-way broadcast.
*   **Redis**: The glue for scaling.

**Tomorrow (Day 50)**: We wrap up Phase 5 with **Event Sourcing & CQRS**. The ultimate architecture for complex domains.
