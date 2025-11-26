# Day 4: RESTful API Principles & Design

## 1. Thinking in Resources

The hardest part of REST is unlearning "RPC" (Remote Procedure Call).
*   **RPC Thinking**: "I need a function to `getUserOrders`." -> `POST /getUserOrders`
*   **REST Thinking**: "I have a User resource, and it has a relationship to Orders." -> `GET /users/{id}/orders`

### 1.1 Identifying Resources
Resources are the "nouns" of your system.
*   **Collection Resource**: A list of items. e.g., `/products`
*   **Singleton Resource**: A specific item. e.g., `/products/123`
*   **Sub-resource**: A resource belonging to another. e.g., `/products/123/reviews`

### 1.2 The "Nouns" Rule
*   **Bad**: `/getAllUsers`, `/createNewUser`, `/updateUser`
*   **Good**: `/users` (GET to list, POST to create), `/users/{id}` (PUT to update).

---

## 2. Advanced URL Design

### 2.1 Hierarchy vs Flat
Should reviews be `/products/123/reviews` or just `/reviews?product_id=123`?
*   **Hierarchy**: Use when the sub-resource *cannot exist* without the parent. A review makes no sense without a product. -> `/products/123/reviews`
*   **Flat**: Use when the resource is independent. An Order belongs to a User, but it also stands alone for the warehouse team. -> `/orders?user_id=123` (Better for admin views).

### 2.2 Filtering, Sorting, Pagination
Don't create new endpoints for different views of data. Use query parameters.
*   **Filtering**: `GET /users?role=admin&active=true`
*   **Sorting**: `GET /users?sort=-created_at` (descending)
*   **Pagination**: `GET /users?page=2&limit=20` or `GET /users?cursor=abc123token`

### 2.3 Field Selection (Partial Response)
Allow clients to request only what they need (saves bandwidth).
*   `GET /users?fields=id,username,email`

---

## 3. JSON Standards & Best Practices

### 3.1 Casing: snake_case vs camelCase
*   **JSON Standard**: There is no official standard, but **camelCase** is dominant in JavaScript/Frontend worlds.
*   **Backend Standard**: Python/Ruby/SQL use **snake_case**.
*   **Recommendation**:
    *   If your team is full-stack JS (Node + React) -> **camelCase**.
    *   If you are a Python/Go backend serving diverse clients -> **snake_case** is often safer as it maps 1:1 to DB columns, but **camelCase** is more "JSON-native".
    *   *Crucial*: Pick one and be consistent.

### 3.2 Date & Time
*   **Always** use ISO 8601 strings in UTC.
*   ✅ `"created_at": "2025-11-27T14:30:00Z"`
*   ❌ `"created_at": 167889222` (Unix timestamps are ambiguous—seconds or ms?)

### 3.3 Enveloping
Should you wrap your response?
*   **No Envelope** (Standard):
    ```json
    [ { "id": 1 }, { "id": 2 } ]
    ```
*   **With Envelope** (Good for metadata):
    ```json
    {
      "data": [ { "id": 1 }, { "id": 2 } ],
      "meta": { "total": 100, "page": 1 }
    }
    ```
*   **Recommendation**: Use envelopes for collections (to support pagination), but raw objects for singletons.

---

## 4. Error Modeling

Don't just return `500 Error`. Give the client a clue.

### 4.1 RFC 7807 (Problem Details for HTTP APIs)
This is the IETF standard for error responses.
```json
{
  "type": "https://example.com/probs/out-of-credit",
  "title": "You do not have enough credit.",
  "status": 403,
  "detail": "Your current balance is 30, but that costs 50.",
  "instance": "/account/12345/msgs/abc"
}
```

### 4.2 Simplified Standard
At minimum, return:
```json
{
  "error": {
    "code": "INVALID_EMAIL",
    "message": "The email address is malformed.",
    "details": "Missing '@' symbol."
  }
}
```

---

## 5. Case Study: Designing an E-commerce API

Let's design the API for "Shopify-Lite".

### 5.1 Requirements
1.  Manage Products.
2.  Users can add items to a Cart.
3.  Users can Checkout (create Order).

### 5.2 The Design
*   **Products**:
    *   `GET /products` (List, filterable)
    *   `GET /products/{id}`
    *   `POST /products` (Admin only)
*   **Cart** (Tricky! Is it a resource?):
    *   Option A: `POST /cart/items` (Add item). Session-based.
    *   Option B: `PUT /users/{id}/cart` (Replace entire cart).
    *   *Decision*: `POST /me/cart/items` (Uses "me" alias for logged-in user).
*   **Orders**:
    *   `POST /orders` (Checkout). Payload includes payment token.
    *   `GET /orders` (My history).
    *   `GET /orders/{id}`

---

## 6. Summary

Today we moved from "How HTTP works" to "How to use HTTP elegantly".
*   **Resources**: Nouns over verbs.
*   **URLs**: Logical hierarchy.
*   **JSON**: ISO dates, consistent casing.
*   **Errors**: Structured, machine-readable error objects.

**Tomorrow (Day 5)**: We will leave the API layer and dive into the **Data Layer**. We'll survey the database landscape (SQL vs NoSQL vs Vector) and spin up our first real persistence stack.
