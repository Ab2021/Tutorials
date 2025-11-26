# Day 58: System Design Interview

## 1. The Game

The System Design Interview is not about the "Right Answer". It's about the **Process**.
*   **Goal**: Show you can navigate ambiguity and trade-offs.
*   **Duration**: 45 mins.

---

## 2. The Framework (RADIO)

### 2.1 **R**equirements (5 mins)
*   **Functional**: What does it do? (Post tweet, Follow user).
*   **Non-Functional**: Scale? Latency? Consistency? (100M DAU, <200ms, Eventual Consistency).

### 2.2 **A**rchitecture (High Level) (10 mins)
*   Draw boxes. LB -> API -> Service -> DB.
*   Keep it simple first.

### 2.3 **D**ata Model (5 mins)
*   Schema. `Users`, `Tweets`, `Follows`.
*   SQL vs NoSQL? (SQL for Users, NoSQL for Tweets).

### 2.4 **I**nterface (API) (5 mins)
*   `POST /tweet`, `GET /feed`.

### 2.5 **O**ptimizations (Deep Dive) (20 mins)
*   "How to scale the Feed?" (Fan-out on Write vs Fan-out on Read).
*   "How to handle Hot Keys?" (Justin Bieber problem).

---

## 3. Back-of-the-Envelope Math

Memorize these:
*   **1 Day**: 86,400 seconds (~10^5).
*   **1 Million Req/Day**: ~12 RPS.
*   **Storage**:
    *   `char`: 1 byte.
    *   `int`: 4 bytes.
    *   `UUID`: 16 bytes.

---

## 4. Common Questions

1.  **URL Shortener (TinyURL)**: Hashing, ID generation.
2.  **Chat App (WhatsApp)**: WebSockets, HBase/Cassandra.
3.  **Rate Limiter**: Redis, Token Bucket.
4.  **Web Crawler**: Distributed queues, Politeness.

---

## 5. Summary

Today we learned to think big.
*   **Clarify**: Don't assume.
*   **Draw**: Visualize.
*   **Justify**: Why Redis? Why Kafka?

**Tomorrow (Day 59)**: We polish your **Resume & Career**.
