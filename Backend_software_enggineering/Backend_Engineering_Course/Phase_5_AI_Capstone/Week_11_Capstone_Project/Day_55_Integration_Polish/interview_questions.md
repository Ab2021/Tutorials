# Day 55: Interview Questions & Answers

## Conceptual Questions

### Q1: How do you debug a request that failed somewhere between 4 services?
**Answer:**
*   **Distributed Tracing**: Use OpenTelemetry/Jaeger.
*   **Correlation ID**: Generate a UUID at the Gateway. Pass it in headers (`X-Request-ID`) to every service. Log it. grep logs for that ID.

### Q2: What is the bottleneck of this architecture?
**Answer:**
*   **Database**: Single Postgres instance. (Fix: Read Replicas).
*   **Redis**: Single Redis instance. (Fix: Cluster).
*   **AI Service**: Embedding/LLM is slow. (Fix: Async queues, more workers).

### Q3: How would you deploy this with Zero Downtime?
**Answer:**
*   **Rolling Update**: Kubernetes. Update 1 pod at a time.
*   **Blue/Green**: Deploy new version (Green) alongside old (Blue). Switch traffic at Load Balancer.

---

## Scenario-Based Questions

### Q4: The "Collab Service" is crashing with OOM (Out of Memory). Why?
**Answer:**
*   **Cause**: Too many active WebSocket connections. Each connection consumes RAM (buffers).
*   **Fix**:
    *   **Scale Out**: Add more Collab instances.
    *   **Optimize**: Reduce buffer size.
    *   **Leak**: Check for memory leaks (e.g., not removing disconnected sockets from the list).

### Q5: Users report that "Search" is slow. How to optimize?
**Answer:**
*   **Cache**: Cache common queries in Redis.
*   **Qdrant**: Use HNSW index (approximate search) instead of exact scan.
*   **Quantization**: Compress vectors.

---

## Behavioral / Role-Specific Questions

### Q6: If you had to rebuild this from scratch, what would you change?
**Answer:**
*   **Monolith First**: Start with a Modular Monolith. Microservices added complexity (Kafka, Redis) that slowed down dev.
*   **CRDTs**: Use Yjs for real-time sync instead of custom WebSocket logic. It handles conflicts better.
*   **Managed Services**: Use Supabase/Firebase for Auth/DB to save time.
