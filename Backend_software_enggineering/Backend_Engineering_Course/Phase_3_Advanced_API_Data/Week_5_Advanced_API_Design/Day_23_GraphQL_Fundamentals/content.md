# Day 23: GraphQL Fundamentals

## 1. REST is Dead? (No, but...)

REST has problems:
1.  **Over-fetching**: You want `user.name`, but you get `user.*` (wasting bandwidth).
2.  **Under-fetching**: You want `user` and `orders`. You need 2 API calls.
3.  **Versioning**: Adding fields is easy, removing is hard.

**GraphQL** solves this by letting the **Client** decide what it wants.

---

## 2. The Core Concepts

### 2.1 The Schema (SDL)
Strongly typed contract.
```graphql
type User {
  id: ID!
  name: String!
  email: String
  orders: [Order]!
}

type Order {
  id: ID!
  total: Float!
}

type Query {
  getUser(id: ID!): User
}
```

### 2.2 The Query (Read)
Client asks for specific fields.
```graphql
query {
  getUser(id: "1") {
    name
    orders {
      total
    }
  }
}
```

### 2.3 The Mutation (Write)
```graphql
mutation {
  createOrder(userId: "1", total: 99.99) {
    id
    total
  }
}
```

---

## 3. Resolvers & The N+1 Problem

### 3.1 Resolvers
Functions that fetch the data.
*   `Query.getUser` -> `SELECT * FROM users WHERE id = ?`
*   `User.orders` -> `SELECT * FROM orders WHERE user_id = ?`

### 3.2 The N+1 Problem
If you query 10 users and their orders:
1.  Fetch 10 Users (1 Query).
2.  For *each* user, fetch Orders (10 Queries).
3.  **Total**: 11 Queries. This kills performance.

### 3.3 The Solution: DataLoader
A utility that batches requests.
1.  Resolvers call `loader.load(user_id)`.
2.  Loader waits 10ms (tick).
3.  Loader collects all IDs `[1, 2, ... 10]`.
4.  Loader runs **one** batch query: `SELECT * FROM orders WHERE user_id IN (1, 2, ... 10)`.
5.  Loader distributes results back to resolvers.

---

## 4. When to use GraphQL?

*   **Yes**:
    *   Mobile Apps (Bandwidth sensitive).
    *   Complex Dashboards (Many nested resources).
    *   Public APIs (GitHub, Shopify).
*   **No**:
    *   Simple CRUD (Overkill).
    *   File Uploads (Awkward in GQL).
    *   Microservice-to-Microservice (Use gRPC).

---

## 5. Summary

Today we gave power to the client.
*   **Schema**: The contract.
*   **Query**: Ask for what you need.
*   **DataLoader**: The performance fix.

**Tomorrow (Day 24)**: We will look at **gRPC**, the high-performance RPC framework used for internal microservice communication.
