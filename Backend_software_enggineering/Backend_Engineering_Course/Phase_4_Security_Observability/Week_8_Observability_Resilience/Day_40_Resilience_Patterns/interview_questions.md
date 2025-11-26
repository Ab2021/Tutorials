# Day 40: Interview Questions & Answers

## Conceptual Questions

### Q1: What are the three states of a Circuit Breaker?
**Answer:**
1.  **Closed**: Normal operation. Requests pass.
2.  **Open**: Failure threshold reached. Requests are blocked immediately (Fast Fail).
3.  **Half-Open**: Probation. Allow limited requests to check if the downstream service has recovered.

### Q2: Why do we need "Jitter" in retries?
**Answer:**
*   **Scenario**: Service B goes down at T=0. 1000 clients fail.
*   **No Jitter**: All 1000 clients retry at T=1s. Service B gets hammered and dies again.
*   **Jitter**: Client A retries at 1.1s, Client B at 1.2s. Spreads the load.

### Q3: What is "Idempotency" and why is it crucial for Retries?
**Answer:**
*   **Definition**: Doing the same thing twice has the same effect as doing it once.
*   **Scenario**: `POST /pay`. If I retry this, do I charge the user twice?
*   **Fix**: Use an `Idempotency-Key` header. The server checks if it already processed this key.

---

## Scenario-Based Questions

### Q4: Your system is suffering from "Cascading Failures". Service A fails, causing B to fail, then C. How do you stop it?
**Answer:**
1.  **Circuit Breakers**: Stop A from hammering B.
2.  **Timeouts**: Don't wait forever. Fail fast.
3.  **Bulkheads**: Isolate resources so one failure doesn't consume all threads.

### Q5: How do you implement Rate Limiting in a distributed system (multiple servers)?
**Answer:**
*   **Local Memory**: Fast, but inaccurate (User can hit Server A 10 times and Server B 10 times).
*   **Redis**: Centralized counter.
    *   `INCR user:1:rate`
    *   `EXPIRE 60`
    *   Use Lua script for atomicity.

---

## Behavioral / Role-Specific Questions

### Q6: A developer argues "We don't need Circuit Breakers, we have Retries". Are they right?
**Answer:**
*   **No**.
*   **Conflict**: Retries *increase* load on a failing service. Circuit Breakers *decrease* load.
*   **Together**: You need both. Retry a few times (for blips). If it keeps failing, Trip the Circuit (for outages).
