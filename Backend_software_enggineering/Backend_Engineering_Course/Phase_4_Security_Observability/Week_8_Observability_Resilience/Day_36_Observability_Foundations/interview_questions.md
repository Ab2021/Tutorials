# Day 36: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is "Structured Logging" (JSON) better than plain text logging?
**Answer:**
*   **Queryability**: You can search `log.user_id = 123` or `log.duration > 500`.
*   **Parsing**: No need to write complex Regex to extract data.
*   **Integration**: Tools like Elasticsearch/Datadog ingest JSON natively.

### Q2: What is "High Cardinality" in metrics and why is it bad?
**Answer:**
*   **Definition**: A dimension with many unique values.
    *   *Low Cardinality*: `status_code` (200, 404, 500). Good.
    *   *High Cardinality*: `user_id` (1 million users). Bad.
*   **Problem**: Time-Series DBs (Prometheus) create a new time series for *each* combination of tags. `requests_total{user_id="1"}`. 1M users = 1M series. Explodes RAM/Disk.
*   **Fix**: Put high cardinality data in **Logs**, not Metrics.

### Q3: Explain "Sampling" in Tracing.
**Answer:**
*   **Problem**: Tracing every single request (100%) is expensive (storage/network).
*   **Solution**: Record only 1% of requests.
    *   **Head Sampling**: Decide at the start (Random 1%).
    *   **Tail Sampling**: Decide at the end (Keep only errors or slow requests).

---

## Scenario-Based Questions

### Q4: A user reports "The site was slow yesterday at 5 PM". How do you investigate?
**Answer:**
1.  **Metrics**: Check Dashboards (Grafana) for 5 PM. Was CPU high? Was DB latency high?
2.  **Logs**: Search for errors around 5 PM.
3.  **Traces**: Find a few slow traces from that time. See which span (DB query, External API) took the most time.

### Q5: You have 1TB of logs per day. It's too expensive to store. What do you do?
**Answer:**
1.  **Retention**: Keep logs for 7 days instead of 30.
2.  **Levels**: Log `INFO` in Prod, but only `DEBUG` when needed.
3.  **Sampling**: Drop 90% of "Health Check" logs.
4.  **Cold Storage**: Move old logs to S3 (Glacier) instead of expensive Hot Storage (Elasticsearch SSD).

---

## Behavioral / Role-Specific Questions

### Q6: A developer logs `password` in plain text to debug a login issue. What do you do?
**Answer:**
*   **Incident**: This is a security breach.
*   **Action**:
    1.  **Rotate**: Force password reset for affected users.
    2.  **Scrub**: Delete the log entry from the centralized logging system.
    3.  **Prevent**: Add "Data Masking" middleware that automatically replaces `password`, `credit_card` fields with `***` before logging.
