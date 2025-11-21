# Day 32 Deep Dive: Encryption & Groups

## 1. End-to-End Encryption (Signal Protocol)
*   **Goal:** Server cannot read messages.
*   **Keys:**
    *   **Identity Key:** Long-term public key.
    *   **Signed Pre-Key:** Medium-term.
    *   **One-Time Pre-Key:** Single use.
*   **X3DH (Extended Triple Diffie-Hellman):**
    *   User A fetches User B's keys from server.
    *   User A derives a Shared Secret on their device.
    *   User A encrypts message.
    *   Server routes encrypted blob.
*   **Double Ratchet:**
    *   Every message changes the encryption key.
    *   **Forward Secrecy:** If key is stolen, past messages are safe.
    *   **Break-in Recovery:** If key is stolen, future messages will eventually be safe.

## 2. Group Chat Optimization
*   **Fan-out:** User A sends 1 message. Server must deliver to 100 group members.
*   **Write Optimization:**
    *   Don't store 100 copies in DB. Store 1 copy in `GroupMessages` table.
    *   Each user has a `LastReadTimestamp`.
*   **Delivery:**
    *   Server iterates members.
    *   If Online -> Push via WebSocket.
    *   If Offline -> Push Notification.

## 3. "Last Seen" / Online Status
*   **Heartbeat:** Client sends heartbeat every 10s.
*   **Redis:** Store `UserID -> LastHeartbeatTime`.
*   **Optimization:** Don't update DB on every heartbeat. Only update if status changes (Online -> Offline).
