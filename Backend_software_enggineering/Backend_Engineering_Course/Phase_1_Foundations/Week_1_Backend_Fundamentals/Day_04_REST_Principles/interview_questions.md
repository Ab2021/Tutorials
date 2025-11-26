# Day 4: Interview Questions & Answers

## Conceptual Questions

### Q1: What is HATEOAS and why is it rarely used in practice?
**Answer:**
*   **Definition**: Hypermedia As The Engine Of Application State. It means the API response contains links to valid next actions.
    *   Example: `{ "id": 1, "links": [ { "rel": "delete", "href": "/users/1" } ] }`
*   **Why rarely used**:
    1.  **Complexity**: Hard to maintain the logic of "which links to show" on the backend.
    2.  **Coupling**: Frontend clients usually hardcode paths anyway for UI layout reasons.
    3.  **Overhead**: Increases payload size.
    *   *Where it IS used*: Pagination links (`next`, `prev`) are very common and useful.

### Q2: How do you version a REST API?
**Answer:**
There are 3 main strategies:
1.  **URL Versioning** (Most common): `GET /v1/users`.
    *   *Pros*: Explicit, easy to cache.
    *   *Cons*: "Pollutes" the URL.
2.  **Header Versioning**: `Accept: application/vnd.myapi.v1+json`.
    *   *Pros*: Cleaner URLs.
    *   *Cons*: Harder to test in browser, cache fragmentation.
3.  **Parameter Versioning**: `GET /users?v=1`.
    *   *Verdict*: I prefer **URL Versioning** for public APIs because of its clarity and ease of use for developers.

---

## Scenario-Based Questions

### Q3: Design an API for a "Like" system (e.g., Instagram).
**Answer:**
*   **Resource**: A "Like" is a relationship between a User and a Post.
*   **Endpoints**:
    *   Option A (Action-based): `POST /posts/{id}/like`. (Simple, but RPC-ish).
    *   Option B (Resource-based): `PUT /posts/{id}/likes/{user_id}`. (Idempotent).
    *   Option C (Sub-resource collection): `POST /posts/{id}/likes`. (Returns 201 Created).
*   **Decision**: Option C is standard.
    *   `POST /posts/{id}/likes` -> "I like this".
    *   `DELETE /posts/{id}/likes` -> "Unlike". (Assumes one like per user, so ID isn't needed if auth token is present).
    *   `GET /posts/{id}/likes` -> List of likers.

### Q4: You need to introduce a breaking change (renaming `username` to `user_handle`). How do you do it without breaking existing clients?
**Answer:**
1.  **Add, Don't Rename**: Add `user_handle` to the response, but keep `username` populated with the same value (or a default).
2.  **Deprecate**: Mark `username` as deprecated in the OpenAPI spec and documentation.
3.  **Monitor**: Log usage of the `username` field if possible (hard in JSON, but possible with GraphQL).
4.  **Version Up**: If the change is structural and messy, launch `/v2/users` with the new schema and keep `/v1/users` running for a sunset period (e.g., 6 months).

---

## Behavioral / Role-Specific Questions

### Q5: A Product Manager wants you to add a `GET /users/create` endpoint that creates a user because "it's easier to call from a browser link". Do you do it?
**Answer:**
**No.**
*   **Security Risk**: GET requests are cached by browsers, CDNs, and proxies. They are also logged in plain text in server access logs. Sending sensitive data (passwords) in URL params is a huge security hole.
*   **CSRF**: GET requests are trivial to exploit via Cross-Site Request Forgery (e.g., an image tag `<img src="/users/create?name=hacker">`).
*   **Education**: I would explain these risks to the PM and insist on using `POST`. If they need a simple way to test, I'd provide a Swagger UI or a simple HTML form.
