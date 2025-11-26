# Day {{ DAY_NUMBER }}: Interview Questions & Answers

## 1. Conceptual & Theory (Junior/Mid)

### Q1: {{ THEORY_QUESTION }}
**Answer:**
*   **Definition**: ...
*   **Key Difference**: Compare with nearest alternative.
*   **Example**: ...

### Q2: {{ DEEP_DIVE_QUESTION }}
**Answer:**
*   **Internals**: Explain how it works (e.g., B-Tree vs LSM Tree).
*   **Complexity**: Time/Space complexity (Big O).
*   **When to Use**: Specific scenarios where this is optimal.

---

## 2. System Design (Senior/Staff)

### Q3: Design a {{ SYSTEM_NAME }} (e.g., Rate Limiter, URL Shortener, Chat System)
*   **Requirements**:
    *   Functional: What must it do?
    *   Non-Functional: 10M DAU, <100ms latency, 99.9% uptime.
*   **High-Level Design**:
    *   Client -> LB -> API -> Cache -> DB.
*   **Deep Dive**:
    *   "How do you handle the Thundering Herd problem?"
    *   "How do you ensure data consistency across regions?"
    *   "What happens if Redis goes down?"

### Q4: Design for Scale - {{ SCALING_SCENARIO }}
**Answer:**
*   **Current Bottleneck**: Identify the constraint (DB, Network, CPU).
*   **Scaling Strategy**: Horizontal sharding? Read replicas? CDN?
*   **Trade-offs**: CAP theorem implications.

---

## 3. Coding & Algorithms

### Q5: Implement {{ ALGORITHM_NAME }}
*   **Problem**: Description (e.g., "Find the longest substring without repeating characters").
*   **Constraints**: Memory: O(n), Time: O(n).
*   **Solution (Python)**:
    ```python
    def solve(s: str) -> int:
        # Your implementation
        pass
    ```
*   **Optimization**: Can we reduce to O(1) space?

---

## 4. Database & Schema Design

### Q6: Design the Schema for {{ USE_CASE }}
*   **Entities**: Users, Orders, Products, Reviews.
*   **Relationships**: One-to-Many (User -> Orders), Many-to-Many (Products <-> Tags).
*   **SQL vs NoSQL**: Justify the choice.
    *   SQL: ACID transactions required?
    *   NoSQL: High write throughput, flexible schema?
*   **Schema**:
    ```sql
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    ```
*   **Indexes**: Which columns? Why?

### Q7: Query Optimization - {{ SLOW_QUERY_SCENARIO }}
**Answer:**
*   **Problem**: "This query takes 5 seconds. Why?"
*   **Investigation**: Run `EXPLAIN ANALYZE`.
*   **Fix**: Add index on `user_id`, use covering index, partition the table.

---

## 5. Troubleshooting & Operations

### Q8: Production Incident - {{ SCENARIO }}
*   **Scenario**: "Latency spiked to 5s. CPU is at 100%. Disk usage at 90%. What do you do?"
*   **Investigation Steps**:
    1.  Check Dashboards (RPS, Error Rate, Latency percentiles).
    2.  Check Logs (Correlation IDs, Stack traces).
    3.  Check Infrastructure (Disk, Memory, Network).
*   **Mitigation**: 
    *   Short-term: Rollback, Scale up, Kill bad query.
    *   Long-term: Post-mortem, Add alerts, Refactor.

### Q9: Security Vulnerability - {{ SECURITY_ISSUE }}
**Answer:**
*   **Vulnerability**: SQL Injection? XSS? CSRF?
*   **Exploit**: How would an attacker abuse this?
*   **Fix**: Parameterized queries, Input validation, Security headers.
*   **Prevention**: Code review, SAST tools, Penetration testing.

---

## 6. Behavioral & Soft Skills

### Q10: {{ LEADERSHIP_QUESTION }}
*   **Context**: Conflict, Failure, Mentorship, or Tight Deadline.
*   **STAR Response**:
    *   **S**ituation: "We had a production outage on Black Friday..."
    *   **T**ask: "I was responsible for restoring service within 1 hour..."
    *   **A**ction: "I coordinated with the DB team, rolled back the deploy, added circuit breakers..."
    *   **R**esult: "Service restored in 45 mins. Revenue loss: $50K (instead of projected $500K)."
*   **Follow-up**: "What would you do differently next time?"
