# Day 58: Interview Questions & Answers

## Conceptual Questions

### Q1: CAP Theorem. Pick two.
**Answer:**
*   **Consistency**: Everyone sees the same data.
*   **Availability**: System is always up.
*   **Partition Tolerance**: System works if network cuts.
*   **Reality**: In distributed systems, P is mandatory. You choose CP (Banking) or AP (Social Media).

### Q2: SQL vs NoSQL. How do you choose?
**Answer:**
*   **SQL**: Structured data, ACID, Complex Joins. (Users, Billing).
*   **NoSQL**: Unstructured, High Write Throughput, Flexible Schema. (Logs, Metadata, Feeds).

### Q3: How do you shard a database?
**Answer:**
*   **Vertical**: Split by feature (User DB, Tweet DB).
*   **Horizontal**: Split by rows.
    *   **Sharding Key**: `user_id`.
    *   **Consistent Hashing**: To minimize data movement when adding nodes.

---

## Scenario-Based Questions

### Q4: Design a Unique ID Generator (like Snowflake).
**Answer:**
*   **Requirements**: Unique, Sortable by time, 64-bit.
*   **Solution**:
    *   1 bit: Sign.
    *   41 bits: Timestamp (ms).
    *   10 bits: Machine ID.
    *   12 bits: Sequence Number.
*   **Why not UUID?**: Too big (128-bit), not sortable, bad for DB indexing.

### Q5: How do you handle the "Thundering Herd" problem in a Cache?
**Answer:**
*   **Scenario**: Key expires. 1000 requests hit DB at once.
*   **Fix**:
    1.  **Mutex**: Only 1 thread computes value.
    2.  **Probabilistic Early Expiration**: Refresh before it actually expires.

---

## Behavioral / Role-Specific Questions

### Q6: You disagree with the interviewer's suggestion. What do you do?
**Answer:**
*   **Discuss**: "That's an interesting approach. My concern is X. Have you considered Y?"
*   **Collaborate**: Treat them as a peer.
*   **Accept**: If they insist, implement their way but note the trade-offs.
