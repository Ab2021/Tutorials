# Day 22 Deep Dive: Event Sourcing & CQRS

## 1. Event Sourcing
*   **Concept:** The "Log" is the source of truth.
*   **Example (Bank Account):**
    *   Traditional: Table `Account` with `Balance = 100`.
    *   Event Sourcing: Table `Events` with:
        1.  `AccountCreated(0)`
        2.  `Deposited(50)`
        3.  `Deposited(50)`
    *   **Current State:** Sum of events ($0+50+50=100$).
*   **Pros:** Audit trail, Time travel (What was balance yesterday?), Debugging.
*   **Cons:** Replaying 1M events is slow (Need Snapshots).

## 2. CQRS (Command Query Responsibility Segregation)
*   **Problem:** In complex systems, the Write model (Normalized) differs from Read model (Denormalized/Aggregated).
*   **Solution:** Split them.
    *   **Command Side (Write):** Validates and writes to Event Store.
    *   **Query Side (Read):** Listens to events and updates a Read DB (Materialized View).
*   **Example:**
    *   **Write:** `CreateOrder` -> Event Store.
    *   **Read:** `OrderSummaryView` (Elasticsearch) updates when `OrderCreated` arrives.
*   **Lag:** Read DB is eventually consistent.

## 3. Case Study: LMAX Architecture
*   **Context:** High-frequency trading.
*   **Design:**
    *   **Disruptor:** A ring buffer queue.
    *   **Single Thread:** Business logic runs on one thread (no locks).
    *   **Event Sourcing:** All inputs are events.
*   **Result:** 6 Million TPS on a single core.
