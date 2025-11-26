# Day 38: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is "Average Latency" a bad metric?
**Answer:**
*   **Scenario**: 99 requests take 1ms. 1 request takes 10s (timeout).
*   **Average**: ~100ms. Looks fine.
*   **Reality**: 1 user is extremely angry.
*   **Better**: Use **Percentiles** (p99). p99 = 10s. This reveals the problem.

### Q2: Push vs Pull Monitoring. Which is better?
**Answer:**
*   **Pull (Prometheus)**: Server scrapes Client.
    *   *Pros*: Centralized control, easier to detect "down" services (scrape fails).
    *   *Cons*: Hard with short-lived jobs (Lambda) that die before scrape.
*   **Push (Datadog/StatsD)**: Client pushes to Server.
    *   *Pros*: Good for short jobs.
    *   *Cons*: Can DDoS the monitoring server.

### Q3: What is "Alert Fatigue"?
**Answer:**
*   **Problem**: You get 100 alerts a day. 99 are "CPU high" but nothing broke.
*   **Result**: You ignore the alerts. When a real outage happens, you miss it.
*   **Fix**: Make alerts **Actionable**. Only alert if a human *must* do something right now. Delete the rest.

---

## Scenario-Based Questions

### Q4: How do you measure "Saturation" for a thread-pool based app (like Tomcat/Java)?
**Answer:**
*   **Metric**: `active_threads / max_threads`.
*   **Warning**: If active threads = max threads, new requests are queued (latency spike) or rejected.
*   **Alert**: Alert when saturation > 80%.

### Q5: You want to monitor a specific business metric: "Number of items added to cart". How?
**Answer:**
*   **Custom Metric**: In the code, inject a Counter.
    *   `cart_adds_total.inc()`
*   **Prometheus**: Scrapes it.
*   **Grafana**: `sum(rate(cart_adds_total[1h]))`.

---

## Behavioral / Role-Specific Questions

### Q6: A manager asks for an SLA of "100% Uptime". Is this reasonable?
**Answer:**
*   **No**.
*   **Cost**: 100% is infinitely expensive (requires infinite redundancy).
*   **Physics**: Networks fail.
*   **Target**: Aim for "Five Nines" (99.999%) or "Four Nines" (99.99%).
*   **Error Budget**: The remaining 0.01% is our budget to break things (deploy new features).
