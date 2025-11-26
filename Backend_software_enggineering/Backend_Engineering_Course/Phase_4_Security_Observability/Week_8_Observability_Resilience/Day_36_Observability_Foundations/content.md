# Day 36: Observability Foundations

## 1. Monitoring vs Observability

*   **Monitoring**: "Is the system healthy?" (Yes/No).
    *   *Dashboard*: CPU is at 90%.
*   **Observability**: "Why is the system behaving this way?" (Debug).
    *   *Query*: Show me all requests from User X that took > 2s.

---

## 2. The Three Pillars

### 2.1 Logs (Events)
*   **What**: "Something happened."
*   **Bad**: `print("Error happened")` (Text). Hard to parse.
*   **Good**: `logger.error({"event": "payment_failed", "user_id": 123, "amount": 50})` (JSON).
*   **Structured Logging**: Always log in JSON. Machines read logs, not humans.

### 2.2 Metrics (Aggregates)
*   **What**: "How many times did it happen?"
*   **Types**:
    *   **Counter**: `requests_total` (Always goes up).
    *   **Gauge**: `memory_usage` (Goes up and down).
    *   **Histogram**: `request_duration` (Buckets: 95% < 200ms).

### 2.3 Traces (Context)
*   **What**: "Where did the request go?"
*   **Problem**: In microservices, a request hits Service A -> B -> C. If C is slow, A looks slow.
*   **Solution**: **Distributed Tracing**. Pass a `Trace-ID` header through all services.

---

## 3. Correlation IDs

The glue that holds logs together.
1.  **Ingress**: Nginx generates `X-Request-ID: abc-123`.
2.  **Service A**: Logs `{"msg": "Processing", "request_id": "abc-123"}`. Calls Service B with header.
3.  **Service B**: Logs `{"msg": "Received", "request_id": "abc-123"}`.
4.  **Debug**: Search Splunk/ELK for `request_id="abc-123"` to see the full story.

---

## 4. Summary

Today we turned on the lights.
*   **Logs**: Detailed events (JSON).
*   **Metrics**: Health trends.
*   **Traces**: The path of a request.
*   **Correlation ID**: Never build a backend without this.

**Tomorrow (Day 37)**: We dive deep into **Distributed Tracing** with OpenTelemetry.
