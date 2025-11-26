# Day 15: Interview Questions & Answers

## Conceptual Questions

### Q1: What is a "Projection" in CQRS?
**Answer:**
*   **Definition**: A read-optimized view of the data derived from the stream of events.
*   **Example**: From a stream of `OrderCreated`, `OrderShipped`, `OrderCancelled` events, we can build:
    1.  `OrdersTable` (for user history).
    2.  `DailyRevenueTable` (for analytics).
    3.  `ShippingQueue` (for warehouse).
*   **Benefit**: You can have multiple projections for the same data, optimized for different queries.

### Q2: How do you handle "Read-Your-Own-Writes" consistency in CQRS?
**Answer:**
*   **Problem**: User updates profile, immediately refreshes page, sees old profile (because Projection is lagging).
*   **Solutions**:
    1.  **UI Trick**: The UI assumes success and updates the local state immediately (Optimistic UI).
    2.  **Version Check**: The Command returns a `version` (e.g., 105). The Query waits until the Read DB reaches version 105 before returning.
    3.  **Write-through**: Update the Read Cache synchronously (sacrifices availability/performance).

### Q3: What is "Snapshotting"?
**Answer:**
*   **Problem**: Replaying 1 million events to calculate the current balance takes too long.
*   **Solution**: Every N events (e.g., 1000), save the current state (Snapshot) to a separate store.
*   **Recovery**: Load the latest Snapshot, then replay only the events that happened *after* that snapshot.

---

## Scenario-Based Questions

### Q4: You deployed a bug that corrupted the "User Balance" projection. How do you fix it?
**Answer:**
*   **The Superpower of Event Sourcing**:
    1.  **Drop** the corrupted Projection (Truncate Table).
    2.  **Fix** the bug in the Projector code.
    3.  **Replay** all events from the beginning of time.
    4.  **Result**: The Projection is rebuilt correctly. No data is lost because the Events (Source of Truth) were immutable.

### Q5: We need to change the event schema (rename a field). What do we do with old events?
**Answer:**
*   **Immutable**: You cannot change old events in the store.
*   **Upcasting**: When loading an event, pass it through an "Upcaster" function that transforms the old schema to the new schema in-memory.
*   **Versioning**: Create a new event type `OrderCreatedV2`. The system must handle both V1 and V2.

---

## Behavioral / Role-Specific Questions

### Q6: A startup founder wants to use Event Sourcing for their MVP "to be scalable". Do you agree?
**Answer:**
**Strongly Disagree.**
*   **Complexity**: Event Sourcing adds massive complexity (Event Store, Projectors, Async handling).
*   **Velocity**: It slows down feature development initially.
*   **MVP Goal**: Validate the idea fast.
*   **Recommendation**: Use a standard CRUD Monolith. If the startup succeeds and hits scale/audit requirements, migrate the core domain to Event Sourcing later.
