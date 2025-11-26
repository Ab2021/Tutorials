# Lab: Day 23 - GraphQL with Strawberry

## Goal
Build a GraphQL API. You will define a schema, implement resolvers, and see the interactive GraphiQL playground.

## Directory Structure
```
day23/
├── app.py
├── schema.py
└── requirements.txt
```

## Step 1: Requirements
```text
strawberry-graphql[fastapi]
uvicorn
```

## Step 2: The Schema (`schema.py`)

```python
import strawberry
from typing import List, Optional

# Mock Data
users_db = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
]

orders_db = [
    {"id": 101, "user_id": 1, "total": 50.0},
    {"id": 102, "user_id": 1, "total": 20.0},
    {"id": 103, "user_id": 2, "total": 100.0},
]

# Types
@strawberry.type
class Order:
    id: strawberry.ID
    total: float

@strawberry.type
class User:
    id: strawberry.ID
    name: str
    email: str

    # Field Resolver (Solves "User -> Orders")
    @strawberry.field
    def orders(self) -> List[Order]:
        # In real life, use DataLoader here!
        user_orders = [o for o in orders_db if o["user_id"] == int(self.id)]
        return [Order(id=o["id"], total=o["total"]) for o in user_orders]

# Query
@strawberry.type
class Query:
    @strawberry.field
    def users(self) -> List[User]:
        return [User(id=u["id"], name=u["name"], email=u["email"]) for u in users_db]

    @strawberry.field
    def user(self, id: strawberry.ID) -> Optional[User]:
        u = next((u for u in users_db if u["id"] == int(id)), None)
        if u:
            return User(id=u["id"], name=u["name"], email=u["email"])
        return None

# Mutation
@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_user(self, name: str, email: str) -> User:
        new_id = len(users_db) + 1
        new_user = {"id": new_id, "name": name, "email": email}
        users_db.append(new_user)
        return User(id=new_id, name=name, email=email)

schema = strawberry.Schema(query=Query, mutation=Mutation)
```

## Step 3: The App (`app.py`)

```python
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from schema import schema

app = FastAPI()
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")
```

## Step 4: Run & Play

1.  **Run**: `uvicorn app:app --reload`
2.  **Open Browser**: `http://localhost:8000/graphql` (GraphiQL Playground).

## Step 5: Queries to Try

**Fetch Users and Orders**:
```graphql
query {
  users {
    name
    orders {
      total
    }
  }
}
```

**Create User**:
```graphql
mutation {
  createUser(name: "Charlie", email: "charlie@example.com") {
    id
    name
  }
}
```

## Challenge
Implement a `DataLoader` for the `orders` field.
*   Hint: Use `strawberry.dataloader.DataLoader`.
*   The loader should accept a list of `user_ids` and return a list of `List[Order]`.
