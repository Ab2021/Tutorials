# Day 52: Core API & Database

## 1. Auth Service

We need to secure our app.
*   **Endpoints**: `/signup`, `/login`.
*   **Tech**: FastAPI, Passlib (hashing), PyJWT.
*   **Flow**:
    1.  User sends `email`, `password`.
    2.  Server verifies hash.
    3.  Server signs JWT (`sub=user_id`, `exp=1h`).
    4.  Client stores JWT.

## 2. Doc Service

The heart of the application.
*   **Endpoints**:
    *   `POST /docs`: Create doc.
    *   `GET /docs/{id}`: Read doc.
    *   `PUT /docs/{id}`: Update doc.
*   **Database**: Postgres (`documents` table).

## 3. Event Publishing

When a document is updated, we must notify the AI Service.
*   **Why?**: So the AI can re-index the new content for search.
*   **How?**: Kafka Producer.
    *   Topic: `doc_updates`.
    *   Message: `{"doc_id": 1, "content": "New text..."}`.

---

## 4. Implementation Details

### 4.1 Dependency Injection
Use FastAPI `Depends` to get the current user from the JWT header.

### 4.2 Database Session
Use `yield db` pattern to manage SQLAlchemy sessions (Open/Close per request).

---

## 5. Summary

Today we built the foundation.
*   **Auth**: Secure access.
*   **Docs**: Data storage.
*   **Events**: The signal flare for other services.

**Tomorrow (Day 53)**: We make it alive. **Real-Time Collaboration** with WebSockets.
