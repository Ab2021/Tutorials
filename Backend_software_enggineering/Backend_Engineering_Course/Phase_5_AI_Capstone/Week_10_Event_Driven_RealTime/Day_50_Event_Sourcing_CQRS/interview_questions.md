# Day 50: Interview Questions & Answers

## Conceptual Questions

### Q1: What is "Snapshotting" in Event Sourcing?
**Answer:**
*   **Problem**: Replaying 1 Million events to calculate current balance is slow.
*   **Solution**: Every 1000 events, save the current state (Snapshot).
*   **Recovery**: Load latest Snapshot + Replay only events *after* that snapshot.

### Q2: Why use CQRS without Event Sourcing?
**Answer:**
*   **Scenario**: You have a complex domain (DDD) but don't need history.
*   **Use Case**:
    *   Write Model: Normalized 3NF SQL (for integrity).
    *   Read Model: Denormalized Mongo/Elasticsearch (for fast search).
    *   Sync: CDC or Domain Events.

### Q3: How do you handle "Schema Evolution" in Event Sourcing?
**Answer:**
*   **Problem**: Event V1 has `name`. Event V2 has `first_name` and `last_name`.
*   **Upcasting**: When loading V1 event, transform it to V2 format on the fly (in memory). Never change the immutable event log.

---

## Scenario-Based Questions

### Q4: User creates an account, then immediately tries to log in. Login fails "User not found". Why?
**Answer:**
*   **Cause**: **Eventual Consistency**. The `UserCreated` event hasn't reached the Read DB yet.
*   **Fix**:
    *   **UI**: Auto-login on client side after creation.
    *   **Write-Through**: Update Read DB synchronously (sacrifices availability).

### Q5: You need to delete user data (GDPR). How do you do it in an immutable Event Log?
**Answer:**
*   **Crypto-Shredding**: Encrypt PII in events. Throw away the key.
*   **Tombstoning**: Append a `UserDeleted` event. (Doesn't remove data from disk, just hides it).
*   **Compaction**: Rewrite the log (hard) to remove specific events.

---

## Behavioral / Role-Specific Questions

### Q6: A startup wants to use Event Sourcing for their MVP. Good idea?
**Answer:**
*   **No**.
*   **Complexity**: It requires a lot of boilerplate (Events, Commands, Handlers, Projectors).
*   **Advice**: Start with CRUD. Refactor to Event Sourcing only if the domain is complex (e.g., Accounting, Logistics) and requires audit trails.
