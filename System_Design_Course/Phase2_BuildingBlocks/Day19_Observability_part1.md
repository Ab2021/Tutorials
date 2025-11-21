# Day 19 Deep Dive: Distributed Tracing

## 1. The Problem
*   User reports "Checkout is slow".
*   Checkout calls `OrderService` -> `PaymentService` -> `FraudService` -> `DB`.
*   Which one is slow? Logs are scattered across servers.

## 2. How Tracing Works
*   **Trace ID:** Generated at the edge (LB/Gateway). Passed to all downstream services (HTTP Headers).
*   **Span ID:** Represents a unit of work (e.g., "DB Query").
*   **Parent ID:** Links spans together.

## 3. Case Study: Uber Jaeger
*   **Architecture:**
    *   **Agent:** Runs on every host (Sidecar). Collects spans via UDP.
    *   **Collector:** Aggregates spans from agents. Writes to DB.
    *   **Storage:** Cassandra/Elasticsearch (Heavy write load).
    *   **UI:** Visualizes the Gantt chart of the request.
*   **Sampling:**
    *   Tracing every request is too expensive (Storage/Network).
    *   **Head-based Sampling:** Decide at start (e.g., 1% of requests).
    *   **Tail-based Sampling:** Decide at end (Keep only slow/error traces). Harder to implement (need to buffer everything).

## 4. OpenTelemetry (OTel)
*   The industry standard for collecting Logs, Metrics, and Traces.
*   Vendor agnostic. (Send data to Prometheus AND Jaeger AND Datadog).
