# Day 23: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the "N+1 Problem" in GraphQL and how do you solve it?
**Answer:**
*   **Problem**: Fetching a list of N items, then fetching a related field for each item individually. Result: 1 initial query + N follow-up queries.
*   **Solution**: **DataLoader**.
*   **Mechanism**: It coalesces all individual loads within a single event loop tick into one batch request (`WHERE id IN (...)`).

### Q2: What are the security risks of GraphQL?
**Answer:**
1.  **Deeply Nested Queries**: `query { user { friends { friends { friends ... } } } }`. Can crash the server (DoS).
    *   *Fix*: **Query Depth Limiting** or **Complexity Analysis** (Cost limiting).
2.  **Introspection**: Exposes the entire schema to attackers.
    *   *Fix*: Disable introspection in production.

### Q3: Explain "Schema Federation".
**Answer:**
*   **Context**: In microservices, you don't want one giant Monolithic GraphQL server.
*   **Federation**: Each microservice (User, Product) defines its own GraphQL subgraph.
*   **Gateway**: A central Gateway (Apollo Gateway) stitches them together into one "Supergraph". The client sees one API, but the backend is distributed.

---

## Scenario-Based Questions

### Q4: You are migrating a REST API to GraphQL. Do you rewrite everything?
**Answer:**
*   **No**.
*   **Strategy**: Build a **GraphQL Gateway** (BFF) that sits *in front* of the existing REST APIs.
*   **Resolvers**: The GraphQL resolvers just make HTTP calls to the old REST endpoints.
*   **Benefit**: You get the GraphQL interface immediately without rewriting the backend logic.

### Q5: How do you handle Caching in GraphQL?
**Answer:**
*   **REST**: Easy. Use HTTP caching (ETag, Last-Modified) because each URL is unique.
*   **GraphQL**: Hard. Everything is `POST /graphql`.
*   **Client-Side**: Apollo Client normalizes data by ID (`__typename:id`) and caches it.
*   **Server-Side**: You can't cache the whole response easily. You must cache at the **Resolver** level or use Persisted Queries (hashing the query string).

---

## Behavioral / Role-Specific Questions

### Q6: A frontend dev loves GraphQL, but the backend team hates it because "it's too much work". How do you mediate?
**Answer:**
*   **Acknowledge**: GraphQL *is* more work for the backend (maintaining schema, resolvers, dataloaders).
*   **Compromise**:
    *   Use **Code-First** GraphQL (generate schema from Python/TS models) to reduce boilerplate.
    *   Or, use tools like **Hasura/PostGraphile** to auto-generate GraphQL from the DB (for simple CRUD).
    *   Agree to use GraphQL only for the "Read" layer (complex UI fetching) and keep REST for simple "Write" actions if that helps.
