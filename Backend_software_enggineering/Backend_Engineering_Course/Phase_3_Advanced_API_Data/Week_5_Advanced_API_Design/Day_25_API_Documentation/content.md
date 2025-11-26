# Day 25: API Documentation & Developer Experience (DX)

## 1. If it's not documented, it doesn't exist.

You built the world's best API. But if the docs suck, no one will use it.
**DX (Developer Experience)** is the UX of APIs.

### 1.1 The Standard: OpenAPI (Swagger)
A machine-readable description of your REST API.
*   **Endpoints**: `/users`, `/orders`.
*   **Methods**: `GET`, `POST`.
*   **Schemas**: `User { id: int, name: string }`.

### 1.2 Code-First vs Design-First
1.  **Code-First (FastAPI approach)**:
    *   Write Python code -> Auto-generate OpenAPI spec.
    *   *Pros*: Always in sync. Fast.
    *   *Cons*: Docs might look "generated" (less human).
2.  **Design-First**:
    *   Write `openapi.yaml` first.
    *   Generate server stubs from it.
    *   *Pros*: Better design thought process. Parallel dev (Frontend mocks the yaml).
    *   *Cons*: Maintenance overhead (keeping code in sync with yaml).

---

## 2. Elements of Great Docs

1.  **Getting Started**: A "Hello World" in 5 minutes. `curl` command that works immediately.
2.  **Authentication**: Clear guide on how to get a token.
3.  **Errors**: List of all possible error codes and *how to fix them*.
4.  **SDKs**: Don't make me write HTTP calls. Give me a Python/JS library. (Use `openapi-generator`).

---

## 3. Tools of the Trade

*   **Swagger UI**: Interactive documentation. (Standard).
*   **Redoc**: Cleaner, read-only documentation.
*   **Postman Collections**: Shareable request sets.

---

## 4. Summary

Today we made our API usable.
*   **OpenAPI**: The contract.
*   **Swagger UI**: The playground.
*   **DX**: Empathy for the developer using your code.

**Week 5 Wrap-Up**:
We have covered:
1.  REST Best Practices (Pagination, Filtering).
2.  Versioning (URL vs Header).
3.  GraphQL (Flexible queries).
4.  gRPC (High performance).
5.  Documentation (OpenAPI).

**Next Week (Week 6)**: We dive into **Data at Scale**. We will optimize SQL queries, explore NoSQL patterns, and learn how to handle millions of rows.
