# Day 13: API Gateway & Backend for Frontend (BFF)

## 1. The Entry Point

In a microservices world, you cannot expose 50 services directly to the internet.
*   **Security Risk**: Every service needs to handle Auth/SSL.
*   **Chattiness**: The client has to make 10 calls to render one page.
*   **Coupling**: The client needs to know the IP/Port of every service.

### 1.1 The API Gateway
A single entry point for all clients.
*   **Responsibilities**:
    1.  **Routing**: `/users` -> User Service, `/orders` -> Order Service.
    2.  **Authentication**: Verify JWT once, pass user info downstream.
    3.  **Rate Limiting**: "100 req/min per IP".
    4.  **Protocol Translation**: HTTP -> gRPC.
    5.  **Caching**: Cache common responses.

### 1.2 Popular Gateways
*   **NGINX / HAProxy**: The workhorses. Fast, config-heavy.
*   **Kong / Tyk**: Built on NGINX, adds plugins (Lua/Go) for Auth, Analytics.
*   **Cloud Native**: AWS API Gateway, Google Cloud Endpoints.
*   **Code-based**: Netflix Zuul, Spring Cloud Gateway (Java).

---

## 2. Backend for Frontend (BFF)

One size does not fit all.
*   **Mobile App**: Needs small payloads (save battery/data). Needs specific fields.
*   **Web App**: Needs rich data. Can handle larger payloads.
*   **3rd Party API**: Needs strict rate limits and standard docs.

### 2.1 The Pattern
Instead of one General Purpose API Gateway, create a specific Gateway for each client type.
*   **Mobile BFF**: Aggregates calls, strips unused fields, returns minimal JSON.
*   **Web BFF**: Returns full objects, handles session cookies.

### 2.2 GraphQL as BFF
GraphQL is often used as a BFF because it allows the client to specify exactly what it needs.
*   *Mobile Query*: `query { user { id } }`
*   *Web Query*: `query { user { id, name, bio, orders { id } } }`

---

## 3. Implementation Patterns

### 3.1 Offloading Auth
*   **Pattern**: The Gateway handles the "Auth Handshake" (OAuth2/OIDC).
*   **Downstream**: The Gateway passes a `X-User-Id` header to internal services.
*   **Benefit**: Internal services don't need to know about OAuth providers or parsing JWT signatures (if Gateway validates them). They just trust the header (Zero Trust networks require mTLS + Token validation, but for simple setups, Gateway termination is common).

### 3.2 Rate Limiting Algorithms
1.  **Token Bucket**: Allow bursts.
2.  **Leaky Bucket**: Smooth out traffic.
3.  **Fixed Window**: "100 reqs per minute". (Susceptible to spikes at minute boundaries).
4.  **Sliding Window**: Smoother.

---

## 4. Summary

Today we learned how to expose our mess of microservices to the world cleanly.
*   **Gateway**: The receptionist. Handles Auth, Routing, and Safety.
*   **BFF**: The concierge. Tailors the experience for specific guests (Mobile vs Web).

**Tomorrow (Day 14)**: We will decouple our services completely using **Event-Driven Architecture**. We'll stop calling APIs and start sending messages.
