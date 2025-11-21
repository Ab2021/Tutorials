# Day 32: Design WhatsApp

## 1. Requirements
*   **Functional:** 1-on-1 Chat, Group Chat, Sent/Delivered/Read Receipts, Online Status.
*   **Non-Functional:** Low latency, High availability, End-to-End Encryption.
*   **Scale:** 2 Billion users. 100 Billion messages/day.

## 2. Architecture
*   **Protocol:** WebSocket (or MQTT). Persistent connection.
*   **Gateway:** Handles WebSocket connections. Holds the session map (`UserID -> SocketID`).
*   **Message Service:** Routes messages.
*   **Group Service:** Manages group membership.
*   **Push Notification:** If user is offline, send via APNS/FCM.

## 3. Message Flow (User A to User B)
1.  User A sends message to Gateway.
2.  Gateway sends to Message Service.
3.  Message Service saves to DB (Cassandra) for history.
4.  Message Service checks Redis: "Is User B online?".
    *   **Yes:** Find Gateway connected to User B. Push message.
    *   **No:** Send to Push Notification Service.

## 4. Storage
*   **Chat History:** Huge volume. Write-heavy.
*   **Choice:** Cassandra or HBase (Wide Column).
*   **Schema:** `PartitionKey: ChatID`, `ClusteringKey: Timestamp`.
*   **Ephemeral:** WhatsApp deletes messages from server after delivery (mostly).
