# Day 37: Distributed Tracing with OpenTelemetry

## 1. The Mystery of the Slow Request

User says "Login is slow".
*   **Monolith**: Check the `login()` function.
*   **Microservices**: `Login Service` calls `User Service` calls `DB`.
    *   Is it the network?
    *   Is it the DB?
    *   Is it the User Service?
    *   **Tracing** answers this.

---

## 2. Core Concepts

### 2.1 Trace
A Directed Acyclic Graph (DAG) of **Spans**. Represents one full request.

### 2.2 Span
A single unit of work.
*   **Name**: `SELECT * FROM users`
*   **Start Time**: 10:00:01.000
*   **Duration**: 50ms
*   **Tags**: `db.type=postgres`, `http.status=200`
*   **Parent ID**: The span that called this one.

### 2.3 Context Propagation
How do we pass the Trace ID from Service A to Service B?
*   **HTTP Headers**: `traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` (W3C Standard).

---

## 3. OpenTelemetry (OTel)

The vendor-neutral standard for generating traces/metrics.
*   **SDK**: Libraries for Python, Java, Go.
*   **Collector**: A proxy that receives traces and sends them to Jaeger/Datadog/Honeycomb.
*   **Auto-Instrumentation**: Magic agents that instrument your code without you changing a line. `opentelemetry-instrument python app.py`.

---

## 4. Visualization (Jaeger)

Once you have traces, you need a UI.
*   **Waterfall View**: Shows spans as bars.
*   **Critical Path**: Highlights the sequence of spans that actually delayed the response.

---

## 5. Summary

Today we followed the breadcrumbs.
*   **Trace**: The journey.
*   **Span**: The steps.
*   **OTel**: The map maker.

**Tomorrow (Day 38)**: We will build **Dashboards**. How to visualize Metrics (CPU, RAM, Latency) using Prometheus and Grafana.
