# Day 32 Interview Prep: Design WhatsApp

## Q1: WebSocket vs HTTP Long Polling?
**Answer:**
*   **Long Polling:** Client requests. Server holds connection open until data arrives. High overhead (headers).
*   **WebSocket:** Full duplex. Persistent. Low overhead. Better for chat.

## Q2: How to handle "Read Receipts" in a group of 200?
**Answer:**
*   **Naive:** 200 users send "Read" ack. Server sends 200 * 200 = 40,000 updates. (DDoS).
*   **Optimization:**
    *   Batch updates on client (Send "Read" every 5s).
    *   Server aggregates acks.
    *   Or, only show "Read by All" or "Read by X, Y, Z" on demand (Pull model).

## Q3: How to sync messages across multiple devices (Phone + Desktop)?
**Answer:**
*   **Sidecar:** The phone is the source of truth (Old WhatsApp).
*   **Multi-Device (New):**
    *   Each device has its own Identity Key.
    *   Sender encrypts message $N$ times (once for each device of the recipient).
    *   Server stores messages for all devices.

## Q4: How to store images/videos?
**Answer:**
*   **Blob Storage (S3):** Store the file. Get a URL.
*   **Message:** Send the URL (encrypted) + Thumbnail (Base64) in the chat message.
