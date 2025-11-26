# Day 22: API Versioning & Stability

## 1. Change is Inevitable

You built `GET /users`. It returns `{ "name": "Alice" }`.
Now you need to split name into `{ "first_name": "Alice", "last_name": "Smith" }`.
*   **Breaking Change**: Old clients expecting `name` will crash.
*   **Solution**: Versioning.

### 1.1 When to Version?
*   **Non-Breaking**: Adding a new field. (No version needed usually).
*   **Breaking**: Removing a field, renaming a field, changing a type (String -> Int). (Version needed).

---

## 2. Versioning Strategies

### 2.1 URL Path Versioning (The Standard)
*   `GET /v1/users`
*   `GET /v2/users`
*   *Pros*: Explicit, easy to see in logs/browser.
*   *Cons*: Technically violates REST (Resource identity shouldn't change).

### 2.2 Query Parameter
*   `GET /users?version=1`
*   *Pros*: Easy to implement.
*   *Cons*: Hard to route at the Gateway level.

### 2.3 Header Versioning (Custom Header)
*   `X-API-Version: 1`
*   *Pros*: URLs stay clean.
*   *Cons*: Harder to test in browser.

### 2.4 Content Negotiation (The "Pure" REST way)
*   `Accept: application/vnd.myapi.v1+json`
*   *Pros*: Semantically correct.
*   *Cons*: Complex. Hardest to debug.

*   **Verdict**: Use **URL Path (`/v1/`)** for public APIs. It's pragmatic and everyone understands it.

---

## 3. Managing the Lifecycle

### 3.1 The "Sunset" Header
How do you kill v1?
1.  **Announce**: "v1 will be removed in 6 months."
2.  **Deprecate**: Add `Warning: 299 - "Deprecated"` header to v1 responses.
3.  **Sunset**: Add `Sunset: Sat, 31 Dec 2025 23:59:59 GMT` header (RFC 8594).
4.  **Brownout**: Randomly fail 10% of v1 requests a week before shutdown to get attention.
5.  **Shutdown**: Return `410 Gone`.

---

## 4. Stability Patterns

### 4.1 Tolerant Reader (Postel's Law)
"Be conservative in what you do, be liberal in what you accept from others."
*   **Server**: If client sends extra fields I don't know, ignore them. Don't crash.
*   **Client**: If server sends extra fields I don't know, ignore them.

### 4.2 Feature Flags
Instead of a new API version, use a flag.
*   `GET /users` -> Returns v1 logic.
*   Enable flag `new-user-schema` for 5% of users.
*   `GET /users` -> Returns v2 logic for those users.

---

## 5. Summary

Today we learned how to evolve.
*   **Versioning**: Use `/v1/` for simplicity.
*   **Deprecation**: Communicate clearly with `Sunset` headers.
*   **Stability**: Don't break clients just because you added a field.

**Tomorrow (Day 23)**: We will break free from the constraints of REST. We will let the client ask for exactly what they want using **GraphQL**.
