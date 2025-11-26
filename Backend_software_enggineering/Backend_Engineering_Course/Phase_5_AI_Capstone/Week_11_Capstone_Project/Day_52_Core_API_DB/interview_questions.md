# Day 52: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the "Dual Write" problem?
**Answer:**
*   **Scenario**: You save to DB, then publish to Kafka.
*   **Failure**: DB save succeeds, but Kafka publish fails (network).
*   **Result**: Inconsistency. AI Service never knows about the update.
*   **Fix**: **Outbox Pattern**. Save the event to a `outbox` table in the *same transaction* as the data. A separate background worker reads `outbox` and pushes to Kafka.

### Q2: Why use Pydantic models separate from SQLAlchemy models?
**Answer:**
*   **Separation of Concerns**:
    *   **SQLAlchemy**: Represents DB table structure (Foreign Keys, Relationships).
    *   **Pydantic**: Represents API contract (Validation, Hiding fields like `password_hash`).
*   **Security**: Prevents accidental leakage of internal fields.

### Q3: How do you handle Database Migrations in production?
**Answer:**
*   **Tool**: Alembic.
*   **Process**:
    1.  Generate migration script (`alembic revision --autogenerate`).
    2.  Review script.
    3.  Apply (`alembic upgrade head`) during deployment pipeline.
*   **Zero Downtime**: Ensure changes are backward compatible (e.g., add column is safe; rename column requires 2 steps).

---

## Scenario-Based Questions

### Q4: Your Auth Service is down. Can users still edit documents?
**Answer:**
*   **Yes**, if:
    *   The `Doc Service` can validate the JWT signature *locally* (using the shared public key or secret).
    *   It doesn't need to call `Auth Service` for every request.
*   **No**, if:
    *   You use Opaque Tokens (Reference Tokens) that must be checked against a DB/Redis in Auth Service.

### Q5: How do you prevent SQL Injection in FastAPI?
**Answer:**
*   **SQLAlchemy**: Uses parameterized queries by default.
*   **Raw SQL**: If you must use `text()`, always use `:param` binding. Never use f-strings `f"SELECT * FROM users WHERE name = '{name}'"`.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to store the JWT in LocalStorage. Is this safe?
**Answer:**
*   **Risk**: XSS (Cross-Site Scripting). If an attacker injects JS, they can steal the token.
*   **Better**: **HttpOnly Cookie**. JS cannot read it.
*   **Trade-off**: Cookies are vulnerable to CSRF (Cross-Site Request Forgery), so you need Anti-CSRF tokens.
