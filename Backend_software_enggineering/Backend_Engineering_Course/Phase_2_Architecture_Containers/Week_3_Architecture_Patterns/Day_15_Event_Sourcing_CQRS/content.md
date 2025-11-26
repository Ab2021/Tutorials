# Day 15: Event Sourcing & CQRS (Intro)

## 1. The Ultimate Truth

In traditional databases, we store the **Current State**.
*   `User: { id: 1, balance: 100 }`
*   If we update it to 90, the 100 is gone forever. We lost the history.

### 1.1 Event Sourcing
Store the **History of Events**, not the current state.
*   `Event 1: AccountCreated { id: 1, balance: 0 }`
*   `Event 2: Deposited { amount: 100 }`
*   `Event 3: Withdrawn { amount: 10 }`
*   **Current State**: Derived by replaying events (0 + 100 - 10 = 90).

### 1.2 Why?
1.  **Audit Trail**: You know exactly *why* the balance is 90.
2.  **Time Travel**: "What was the balance last Tuesday?" (Replay events up to Tuesday).
3.  **Debugging**: Copy events to a dev machine and replay them to reproduce a bug exactly.

---

## 2. CQRS (Command Query Responsibility Segregation)

If we store events, how do we query "All users with balance > 50"?
Replaying millions of events for every query is too slow.

### 2.1 The Split
*   **Command Side (Write)**:
    *   Optimized for high throughput writes.
    *   Stores Events in an **Event Store** (Append-only).
    *   Validates business rules ("Can I withdraw?").
*   **Query Side (Read)**:
    *   Optimized for fast reads.
    *   Stores a **Projection** (Materialized View) in a SQL/NoSQL DB.
    *   Updated asynchronously by listening to events.

### 2.2 The Architecture
1.  **User**: `POST /withdraw`
2.  **Command Service**: Validates -> Appends `Withdrawn` event to Event Store.
3.  **Projector**: Listens to `Withdrawn` -> Updates `Users` table in Postgres (Read DB).
4.  **User**: `GET /balance` -> Reads from Postgres.

---

## 3. Challenges

Event Sourcing is **Hard Mode**.
1.  **Eventual Consistency**: The Read DB lags behind the Write DB by milliseconds (or seconds). User might withdraw and see old balance for a moment.
2.  **Versioning**: What if you change the Event Schema? You need "Upcasters" to migrate old events on the fly.
3.  **Snapshots**: Replaying 1M events is slow. You need to save a "Snapshot" every 1000 events and replay from there.

---

## 4. When to use it?

*   **Yes**: Banking, Ledgers, Git (it's event sourced!), Complex domains where history matters.
*   **No**: Simple CRUD apps (Blogs, ToDo lists). It's over-engineering.

---

## 5. Summary

Today we explored the most advanced pattern in backend engineering.
*   **Event Sourcing**: State is a function of history.
*   **CQRS**: Optimize Reads and Writes separately.

**Week 3 Wrap-Up**:
We have covered:
1.  Monolith vs Microservices.
2.  Service Boundaries & DB-per-Service.
3.  API Gateway & BFF.
4.  Event-Driven Architecture (Brokers).
5.  Event Sourcing & CQRS.

**Next Week (Week 4)**: We move to **Infrastructure**. We will containerize our apps with **Docker**, orchestrate them with **Kubernetes**, and provision cloud resources with **Terraform**.
