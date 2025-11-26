# Day 23: GraphQL Deep Dive - Modern API Query Language

## Table of Contents
1. [GraphQL vs REST](#1-graphql-vs-rest)
2. [Schema Definition](#2-schema-definition)
3. [Queries](#3-queries)
4. [Mutations](#4-mutations)
5. [Resolvers](#5-resolvers)
6. [N+1 Problem & DataLoader](#6-n1-problem--dataloader)
7. [Pagination](#7-pagination)
8. [Authentication & Authorization](#8-authentication--authorization)
9. [Subscriptions](#9-subscriptions)
10. [Summary](#10-summary)

---

## 1. GraphQL vs REST

### 1.1 REST Problems

**Over-fetching** (get too much data):
```http
GET /users/123

{
  "id": 123,
  "name": "Alice",
  "email": "alice@example.com",
  "createdAt": "2024-01-01",
  "updatedAt": "2024-01-15",
  "settings": {...},  // Don't need this
  "preferences": {...}  // Or this
}
```

**Under-fetching** (need multiple requests):
```http
GET /users/123          // Get user
GET /users/123/orders   // Get user's orders
GET /orders/456/items   // Get order items

3 round-trips!
```

### 1.2 GraphQL Solution

**Single request, exact fields**:
```graphql
query {
  user(id: 123) {
    name
    orders {
      id
      items {
        name
        price
      }
    }
  }
}

# Response:
{
  "data": {
    "user": {
      "name": "Alice",
      "orders": [{
        "id": 456,
        "items": [...]
      }]
    }
  }
}
```

**Benefits**:
- ✅ No over-fetching (only requested fields)
- ✅ No under-fetching (single request)
- ✅ Strongly typed schema
- ✅ Self-documenting

### 1.3 When to Use GraphQL

✅ **Use GraphQL when**:
- Mobile apps (bandwidth matters)
- Complex data relationships
- Multiple clients with different needs
- Rapid frontend iteration

❌ **Use REST when**:
- Simple CRUD
- File uploads/downloads
- Caching critical (HTTP caching works better)
- Team unfamiliar with GraphQL

---

## 2. Schema Definition

### 2.1 Basic Types

```graphql
# schema.graphql
type User {
  id: ID!           # ! = required
  name: String!
  email: String!
  age: Int
  isActive: Boolean
  createdAt: DateTime
}

type Query {
  user(id: ID!): User
  users: [User!]!    # Array of Users (non-null)
}
```

### 2.2 Relationships

```graphql
type User {
  id: ID!
  name: String!
  orders: [Order!]!  # User has many orders
}

type Order {
  id: ID!
  total: Float!
  user: User!        # Order belongs to user
  items: [OrderItem!]!
}

type OrderItem {
  id: ID!
  product: Product!
  quantity: Int!
}

type Product {
  id: ID!
  name: String!
  price: Float!
}
```

### 2.3 Custom Scalars

```python
from graphene import Scalar
from datetime import datetime

class DateTime(Scalar):
    @staticmethod
    def serialize(dt):
        return dt.isoformat()
    
    @staticmethod
    def parse_value(value):
        return datetime.fromisoformat(value)
```

---

## 3. Queries

### 3.1 Basic Query

```graphql
query GetUser {
  user(id: "123") {
    name
    email
  }
}
```

### 3.2 Nested Queries

```graphql
query GetUserWithOrders {
  user(id: "123") {
    name
    orders {
      id
      total
      items {
        product {
          name
          price
        }
        quantity
      }
    }
  }
}
```

### 3.3 Variables

```graphql
query GetUser($userId: ID!) {
  user(id: $userId) {
    name
    email
  }
}

# Variables:
{
  "userId": "123"
}
```

### 3.4 Aliases

```graphql
query {
  alice: user(id: "123") {
    name
  }
  bob: user(id: "456") {
    name
  }
}

# Response:
{
  "data": {
    "alice": {"name": "Alice"},
    "bob": {"name": "Bob"}
  }
}
```

### 3.5 Fragments

```graphql
fragment UserFields on User {
  id
  name
  email
}

query {
  user(id: "123") {
    ...UserFields
    orders {
      id
    }
  }
}
```

---

## 4. Mutations

### 4.1 Basic Mutation

```graphql
type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
}

input CreateUserInput {
  name: String!
  email: String!
  password: String!
}

input UpdateUserInput {
  name: String
  email: String
}
```

**Usage**:
```graphql
mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    id
    name
    email
  }
}

# Variables:
{
  "input": {
    "name": "Alice",
    "email": "alice@example.com",
    "password": "secret"
  }
}
```

### 4.2 Multiple Mutations

```graphql
mutation {
  createUser(input: {...}) {
    id
  }
  createOrder(input: {...}) {
    id
  }
}
```

**Executes sequentially** (unlike queries which can be parallel).

---

## 5. Resolvers

### 5.1 Basic Resolver (Python/Graphene)

```python
import graphene
from graphene_sqlalchemy import SQLAlchemyObjectType

class UserType(SQLAlchemyObjectType):
    class Meta:
        model = User

class Query(graphene.ObjectType):
    user = graphene.Field(UserType, id=graphene.ID(required=True))
    users = graphene.List(UserType)
    
    def resolve_user(self, info, id):
        return db.query(User).filter(User.id == id).first()
    
    def resolve_users(self, info):
        return db.query(User).all()

schema = graphene.Schema(query=Query)
```

### 5.2 Nested Resolvers

```python
class UserType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    orders = graphene.List(lambda: OrderType)
    
    def resolve_orders(self, info):
        # Called when client requests user.orders
        return db.query(Order).filter(Order.user_id == self.id).all()

class OrderType(graphene.ObjectType):
    id = graphene.ID()
    total = graphene.Float()
    user = graphene.Field(UserType)
    
    def resolve_user(self, info):
        return db.query(User).filter(User.id == self.user_id).first()
```

### 5.3 Mutation Resolver

```python
class CreateUser(graphene.Mutation):
    class Arguments:
        name = graphene.String(required=True)
        email = graphene.String(required=True)
    
    user = graphene.Field(UserType)
    
    def mutate(self, info, name, email):
        user = User(name=name, email=email)
        db.add(user)
        db.commit()
        return CreateUser(user=user)

class Mutation(graphene.ObjectType):
    create_user = CreateUser.Field()

schema = graphene.Schema(query=Query, mutation=Mutation)
```

---

## 6. N+1 Problem & DataLoader

### 6.1 The N+1 Problem

**Query**:
```graphql
query {
  users {           # 1 query
    name
    orders {        # N queries (one per user)
      id
    }
  }
}
```

**SQL executed**:
```sql
SELECT * FROM users;                       -- 1 query
SELECT * FROM orders WHERE user_id = 1;    -- N queries
SELECT * FROM orders WHERE user_id = 2;
SELECT * FROM orders WHERE user_id = 3;
... (100 users = 100 queries!)
```

**Performance**: 1 + 100 = 101 queries! ❌

### 6.2 Solution: DataLoader

```python
from promise import Promise
from promise.dataloader import DataLoader

class OrderLoader(DataLoader):
    def batch_load_fn(self, user_ids):
        # Single query for all user IDs
        orders = db.query(Order).filter(Order.user_id.in_(user_ids)).all()
        
        # Group by user_id
        orders_by_user = {}
        for order in orders:
            orders_by_user.setdefault(order.user_id, []).append(order)
        
        # Return in same order as user_ids
        return Promise.resolve([orders_by_user.get(uid, []) for uid in user_ids])

# Usage in resolver
class UserType(graphene.ObjectType):
    orders = graphene.List(OrderType)
    
    def resolve_orders(self, info):
        loader = info.context['order_loader']
        return loader.load(self.id)

# Middleware
def get_context():
    return {
        'order_loader': OrderLoader()
    }
```

**SQL executed**:
```sql
SELECT * FROM users;                                    -- 1 query
SELECT * FROM orders WHERE user_id IN (1, 2, 3, ...);  -- 1 query!
Total: 2 queries ✅
```

---

## 7. Pagination

### 7.1 Relay-Style Cursor Pagination

```graphql
type Query {
  users(first: Int, after: String): UserConnection!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
}

type UserEdge {
  node: User!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

**Query**:
```graphql
query {
  users(first: 10, after: "cursor123") {
    edges {
      node {
        id
        name
      }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

### 7.2 Implementation

```python
import base64

def resolve_users(self, info, first=10, after=None):
    query = db.query(User).order_by(User.id)
    
    if after:
        last_id = int(base64.b64decode(after))
        query = query.filter(User.id > last_id)
    
    users = query.limit(first + 1).all()
    
    has_next_page = len(users) > first
    users = users[:first]
    
    edges = [
        {
            "node": user,
            "cursor": base64.b64encode(str(user.id).encode()).decode()
        }
        for user in users
    ]
    
    return {
        "edges": edges,
        "pageInfo": {
            "hasNextPage": has_next_page,
            "endCursor": edges[-1]["cursor"] if edges else None
        }
    }
```

---

## 8. Authentication & Authorization

### 8.1 Context-Based Auth

```python
from fastapi import Header

def get_context(authorization: str = Header(None)):
    if not authorization:
        return {}
    
    token = authorization.replace("Bearer ", "")
    user = verify_jwt(token)
    
    return {"current_user": user}

# In resolver
def resolve_user(self, info, id):
    current_user = info.context.get("current_user")
    
    if not current_user:
        raise Exception("Not authenticated")
    
    return db.query(User).filter(User.id == id).first()
```

### 8.2 Field-Level Authorization

```python
class UserType(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()
    email = graphene.String()
    
    def resolve_email(self, info):
        current_user = info.context.get("current_user")
        
        # Only return email if viewing own profile
        if self.id != current_user.id:
            return None
        
        return self.email
```

### 8.3 Directives

```graphql
directive @auth(requires: Role = USER) on FIELD_DEFINITION

enum Role {
  USER
  ADMIN
}

type Query {
  users: [User!]! @auth(requires: ADMIN)
  user(id: ID!): User @auth(requires: USER)
}
```

---

## 9. Subscriptions

### 9.1 Real-Time Updates

```graphql
type Subscription {
  orderCreated: Order!
  messageReceived(chatId: ID!): Message!
}
```

**Client** (JavaScript):
```javascript
const subscription = client.subscribe({
  query: gql`
    subscription {
      orderCreated {
        id
        total
        user {
          name
        }
      }
    }
  `
})

subscription.subscribe({
  next(data) {
    console.log("New order:", data.orderCreated)
  }
})
```

### 9.2 Server Implementation

```python
import asyncio
from graphene import ObjectType, Subscription

class Subscription(ObjectType):
    order_created = graphene.Field(OrderType)
    
    async def subscribe_order_created(self, info):
        # Listen to event stream
        while True:
            order = await order_queue.get()
            yield order

schema = graphene.Schema(query=Query, mutation=Mutation, subscription=Subscription)
```

### 9.3 WebSocket Transport

```python
from fastapi import FastAPI, WebSocket
from graphql_ws.websockets import GraphQLWS

app = FastAPI()
graphql_ws = GraphQLWS(schema)

@app.websocket("/graphql")
async def websocket_endpoint(websocket: WebSocket):
    await graphql_ws.handle(websocket)
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **GraphQL** - Query language for APIs
2. ✅ **Schema** - Strongly typed, self-documenting
3. ✅ **Resolvers** - Fetch data for each field
4. ✅ **DataLoader** - Solves N+1 problem
5. ✅ **Cursor Pagination** - Relay spec
6. ✅ **Context Auth** - Current user in context
7. ✅ **Subscriptions** - Real-time via WebSockets

### 10.2 GraphQL vs REST

| Aspect | REST | GraphQL |
|:-------|:-----|:--------|
| **Endpoints** | Many (`/users`, `/orders`) | One (`/graphql`) |
| **Over-fetching** | Common | No (exact fields) |
| **Under-fetching** | Multiple requests | Single request |
| **Versioning** | Needed | Schema evolution |
| **Caching** | HTTP caching | Harder (needs Apollo/Relay) |
| **File Upload** | Easy | Harder |

### 10.3 Tomorrow (Day 24): gRPC & Protocol Buffers

- **gRPC vs REST/GraphQL**: Performance comparison
- **Protocol Buffers**: Binary serialization
- **Service definition**: .proto files
- **Streaming**: Unary, server, client, bidirectional
- **Error handling**: Status codes
- **Production**: Load balancing, health checks

See you tomorrow! 🚀

---

**File Statistics**: ~1000 lines | GraphQL Deep Dive mastered ✅
