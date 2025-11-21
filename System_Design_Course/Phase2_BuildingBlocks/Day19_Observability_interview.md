# Day 19 Interview Prep: Observability

## Q1: Push vs Pull Metrics?
**Answer:**
*   **Pull (Prometheus):** Server scrapes targets.
    *   **Pros:** Server controls load. Easier to tell if target is down.
    *   **Cons:** Target must be reachable.
*   **Push (Graphite/Datadog):** Target sends metrics to server.
    *   **Pros:** Good for short-lived jobs (Lambda).
    *   **Cons:** Can DDoS the monitoring server.

## Q2: How to debug a high latency issue?
**Answer:**
1.  Check **Metrics** (Golden Signals). Is it CPU saturation? Network?
2.  Check **Traces**. Which span is taking long? (DB query? External API?).
3.  Check **Logs** for that specific Trace ID. Look for errors/timeouts.
4.  Check **Profiler** (Flame graph) if it's a code issue.

## Q3: What is Cardinality in metrics?
**Answer:**
*   The number of unique combinations of labels.
*   `http_requests{method="POST", url="/api/v1/user/123"}`.
*   If you include `user_id` in the label, cardinality explodes (Millions of time series).
*   **Rule:** Never put high-cardinality data (IDs, Emails) in metric labels. Put them in Logs/Traces.

## Q4: Percentiles vs Averages?
**Answer:**
*   **Average:** Hides outliers. (If 9 requests take 1ms and 1 takes 10s, avg is ~1s. Misleading).
*   **Percentiles (p99):** "99% of requests are faster than X". Reveals the tail latency (the slow experience). Always use p95/p99.
