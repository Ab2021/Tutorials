# Day 37 Interview Prep: Design Notification System

## Q1: How to prevent "Notification Storm"?
**Answer:**
*   **Scenario:** A celebrity posts. 100M followers need notification.
*   **Solution:**
    *   **Sharding:** Distribute users across queues.
    *   **Lazy Evaluation:** Don't send to inactive users.
    *   **Collapse:** If 5 people like your photo, send "5 people liked..." instead of 5 notifications.

## Q2: What if the 3rd party (Twilio) is down?
**Answer:**
*   **Circuit Breaker:** Detect high failure rate. Open circuit.
*   **Failover:** Switch to backup provider (e.g., Nexmo).
*   **Queue:** Buffer messages until provider recovers.

## Q3: How to design a "Do Not Disturb" (DND) feature?
**Answer:**
*   **User Settings:** Store `DND_Start: 22:00`, `DND_End: 08:00`.
*   **Worker Check:** Before sending, check `CurrentTime` vs `UserTimezone`.
*   **Delay:** If in DND, calculate `Delay = DND_End - Now`. Push to a "Delayed Queue" (RabbitMQ Scheduled Message).

## Q4: Push vs Pull for In-App Notifications?
**Answer:**
*   **Push (WebSocket):** Best for real-time.
*   **Pull (Polling):** Client polls `/notifications` every minute. Good for "Badge Count".
*   **Hybrid:** Push for "New Message", Pull for "List of past notifications".
