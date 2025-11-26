# Day 40: Resilience Patterns

## 1. Things Will Fail

"Everything fails, all the time." - Werner Vogels (Amazon CTO).
How do we survive failure?

---

## 2. Circuit Breaker

*   **Problem**: Service A calls Service B. Service B is down (timeout). Service A keeps calling, waiting 30s, and crashing its own threads.
*   **Solution**: **Circuit Breaker**.
    *   **Closed (Normal)**: Requests go through.
    *   **Open (Tripped)**: If 50% of requests fail, Open the circuit. Fail immediately (Fast Fail). Don't call Service B.
    *   **Half-Open**: After 10s, let 1 request through. If it works, Close. If not, Open again.

---

## 3. Retries

*   **Problem**: Temporary network blip.
*   **Solution**: Try again.
*   **Danger**: **Retry Storm**. If 1000 users retry at once, Service B dies harder.
*   **Fix**:
    *   **Exponential Backoff**: Wait 1s, 2s, 4s, 8s.
    *   **Jitter**: Wait 1s + random(0.1). Prevents synchronized thundering herds.

---

## 4. Rate Limiting

*   **Problem**: One user sends 1000 req/sec.
*   **Solution**: Limit them to 10 req/sec. Return `429 Too Many Requests`.
*   **Algorithm**: Token Bucket, Leaky Bucket.

---

## 5. Bulkhead Pattern

*   **Analogy**: Ships have watertight compartments. If one floods, the ship doesn't sink.
*   **Software**: Separate Thread Pools.
    *   Pool A: `get_user_profile` (Fast).
    *   Pool B: `generate_report` (Slow).
    *   If Pool B is full, Pool A still works.

---

## 6. Summary

Today we built a tank.
*   **Circuit Breaker**: Stop the bleeding.
*   **Retry**: Handle blips.
*   **Rate Limit**: Stop abuse.
*   **Bulkhead**: Isolate failure.

**Phase 4 Wrap-Up**:
We have covered:
1.  Security (OAuth2, OWASP, Vault).
2.  Testing (Unit, E2E).
3.  Observability (Logs, Metrics, Tracing).
4.  Performance (Caching).
5.  Resilience (Circuit Breakers).

**Next Week (Week 9)**: We enter Phase 5. **AI & Capstone**. We will integrate LLMs (OpenAI) and build RAG pipelines.
