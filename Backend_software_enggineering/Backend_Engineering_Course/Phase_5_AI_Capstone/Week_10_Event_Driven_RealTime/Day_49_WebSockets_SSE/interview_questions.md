# Day 49: Interview Questions & Answers

## Conceptual Questions

### Q1: WebSockets vs SSE. When to use which?
**Answer:**
*   **WebSockets**:
    *   **Bi-directional**: Chat, Games.
    *   **Binary Data**: Efficient.
    *   **Complex**: Custom protocol.
*   **SSE**:
    *   **Uni-directional**: Stock Ticker, Notifications.
    *   **Simple**: Just HTTP. Auto-reconnect built-in.
    *   **Text Only**: Base64 needed for binary.

### Q2: How do Load Balancers handle WebSockets?
**Answer:**
*   **Upgrade**: LB must support the `Connection: Upgrade` header.
*   **Timeout**: LB usually has a timeout (e.g., 60s). You must send **Heartbeats/Ping-Pongs** to keep the connection alive.
*   **Sticky Sessions**: Not strictly needed if using Redis Pub/Sub, but helpful.

### Q3: What is the "C10k Problem"?
**Answer:**
*   **History**: Handling 10,000 concurrent connections used to be hard (Thread per connection).
*   **Modern**: Async I/O (Node.js, Go, Python asyncio) handles 100k+ connections easily on a single core.
*   **Limit**: File Descriptors (ulimit) and RAM.

---

## Scenario-Based Questions

### Q4: You have a Chat App. Users complain they miss messages when they lose internet. How do you fix it?
**Answer:**
*   **Ack**: Client must acknowledge receipt.
*   **Buffer**: Server stores un-acked messages in a "Mailbox" (Redis List).
*   **Reconnect**: When client reconnects, Server sends contents of Mailbox.

### Q5: How do you secure a WebSocket connection?
**Answer:**
*   **WSS**: Use TLS (SSL).
*   **Auth**:
    1.  **Cookie**: Send Session Cookie during Handshake.
    2.  **Token**: Send JWT in Query Param (`?token=...`) or First Message. (Headers are hard to set in JS WebSocket API).

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to use WebSockets for a simple "New Email" notification. Good idea?
**Answer:**
*   **Overkill**.
*   **Cost**: Maintaining a persistent TCP connection for every user is expensive (server memory).
*   **Alternative**: Use **SSE** (lighter) or **Push Notifications** (FCM/APNS) for mobile.
