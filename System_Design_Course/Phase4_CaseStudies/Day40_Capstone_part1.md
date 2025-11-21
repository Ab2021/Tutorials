# Day 40 Deep Dive: Design Ticketmaster

## 1. The "Taylor Swift" Problem
*   10 Million users want 50k seats.
*   Traffic spike is 1000x normal.

## 2. Architecture
*   **Waiting Room (Queue):**
    *   Don't let 10M users hit the DB.
    *   Put them in a Virtual Queue (Redis/Kafka).
    *   Let in 500 users/sec (Rate Limiting).
*   **Seat Selection:**
    *   User sees map. Clicks seat.
    *   **Temporary Lock:** `SET seat_123_lock user_abc EX 300 NX` (Redis).
    *   User has 5 mins to pay.
*   **Payment:**
    *   If success -> Commit to DB.
    *   If fail/timeout -> Release Redis lock.

## 3. Consistency
*   **Strict:** Cannot double book.
*   **Isolation:** Serializable Isolation Level in DB (or just rely on Redis Lock).

## 4. Bot Prevention
*   **CAPTCHA:** At queue entry.
*   **IP Rate Limit:** Block Data Centers.
*   **Account Age:** Prioritize old accounts (Verified Fan).
