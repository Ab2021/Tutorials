# Day 37: Interview Questions & Answers

## Conceptual Questions

### Q1: How does "Context Propagation" work across a Message Queue (Kafka)?
**Answer:**
*   **Problem**: HTTP headers don't exist in Kafka messages naturally.
*   **Solution**: You must inject the `traceparent` into the **Kafka Message Headers** (metadata).
*   **Consumer**: The consumer reads the header, extracts the Trace ID, and starts a new Child Span linked to that ID.

### Q2: What is the difference between OpenTelemetry and Jaeger?
**Answer:**
*   **OpenTelemetry**: The **Producer**. It defines the API and SDKs to *generate* the data.
*   **Jaeger**: The **Consumer**. It is a backend to *store* and *visualize* the data.
*   *Analogy*: OTel is the camera. Jaeger is the photo album.

### Q3: Why is "Auto-Instrumentation" preferred?
**Answer:**
*   **Ease**: You don't have to rewrite code. Just run `opentelemetry-instrument python app.py`.
*   **Coverage**: It automatically patches standard libraries (Requests, Flask, Psycopg2, Boto3) to generate spans.
*   **Manual**: Only needed for custom business logic ("Calculate Tax").

---

## Scenario-Based Questions

### Q4: You see a trace with a 5-second gap between two spans. What does it mean?
**Answer:**
*   **Gap**: Time unaccounted for.
*   **Causes**:
    1.  **CPU Block**: The app was doing heavy computation (not I/O) which wasn't instrumented.
    2.  **GC Pause**: Garbage Collection stopped the world.
    3.  **Network**: Time spent on the wire before the next service received the request.

### Q5: How do you handle PII in traces?
**Answer:**
*   **Risk**: If you log `db.statement="SELECT * FROM users WHERE email='alice@example.com'"`, you leaked PII.
*   **Fix**: Configure the OTel Collector to **Redact** or **Hash** sensitive attributes before sending them to the backend.

---

## Behavioral / Role-Specific Questions

### Q6: A manager says "Tracing is too expensive, turn it off". What is your counter-argument?
**Answer:**
*   **MTTR (Mean Time To Resolution)**: Without tracing, debugging a microservice outage takes hours of guessing. With tracing, it takes minutes.
*   **Cost Control**: We don't need 100% sampling. We can sample 1% of success requests and 100% of error requests (Tail Sampling). This drastically reduces cost while keeping the value.
