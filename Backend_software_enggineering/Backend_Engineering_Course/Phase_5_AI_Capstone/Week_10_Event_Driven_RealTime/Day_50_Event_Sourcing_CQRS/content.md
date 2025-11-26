# Day 50: Event Sourcing & CQRS

## 1. The Ultimate Audit Trail

*   **CRUD**: Update `balance = 100`. (History is lost).
*   **Event Sourcing**: Store `Deposited(50)`, `Withdrew(20)`, `Deposited(70)`.
    *   Current State = Sum of all events.
    *   **Benefit**: Time Travel. "What was the balance last Tuesday?"

---

## 2. CQRS (Command Query Responsibility Segregation)

*   **Problem**: The "Write Model" (Events) is hard to query. "Give me all users with balance > 100" requires replaying millions of events.
*   **Solution**: Split the models.
    *   **Command Side (Write)**: Optimized for Writes. Stores Events. (Kafka/EventStore).
    *   **Query Side (Read)**: Optimized for Reads. Stores Denormalized Views. (Elasticsearch/Postgres).

---

## 3. The Flow

1.  **Command**: `Deposit(50)`.
2.  **Validation**: Check rules.
3.  **Event Store**: Append `Deposited(50)`.
4.  **Projector**: Listens to event. Updates Read DB (`UPDATE users SET balance = balance + 50`).
5.  **Query**: `SELECT balance FROM users`.

---

## 4. Pros & Cons

*   **Pros**:
    *   **Audit**: Perfect history.
    *   **Performance**: Scale Reads and Writes independently.
    *   **Flexibility**: Create new views by replaying old events.
*   **Cons**:
    *   **Complexity**: High.
    *   **Eventual Consistency**: Read DB lags behind Write DB.

---

## 5. Summary

Today we separated concerns.
*   **Event Sourcing**: Source of Truth is the Log.
*   **CQRS**: Write to one, Read from another.

**Week 10 Wrap-Up**:
We have covered:
1.  EDA Foundations.
2.  RabbitMQ.
3.  Kafka.
4.  WebSockets.
5.  Event Sourcing.

**Next Week (Week 11)**: **The Capstone Project**. We will build a "Real-Time Collaborative AI Editor" (Like Google Docs + ChatGPT).
