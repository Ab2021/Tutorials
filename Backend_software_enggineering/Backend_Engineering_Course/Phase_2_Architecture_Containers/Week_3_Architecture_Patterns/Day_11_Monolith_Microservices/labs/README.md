# Lab: Day 11 - Monolith to Microservices

## Goal
Experience the refactoring process. You will start with a simple Monolith and split it into two communicating services.

## Directory Structure
```
day11/
├── monolith/
│   └── app.py
├── microservices/
│   ├── user_service/
│   │   └── app.py
│   └── order_service/
│       └── app.py
└── requirements.txt
```

## Step 1: The Monolith (`monolith/app.py`)

```python
from fastapi import FastAPI

app = FastAPI()

users = {1: {"name": "Alice", "email": "alice@example.com"}}
orders = []

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return users.get(user_id, {})

@app.post("/orders")
def create_order(user_id: int, item: str):
    # Monolith advantage: Direct memory access
    user = users.get(user_id)
    if not user:
        return {"error": "User not found"}
    
    order = {"user_id": user_id, "item": item, "email": user["email"]}
    orders.append(order)
    return order
```

## Step 2: The Split (Microservices)

### User Service (`microservices/user_service/app.py`)
Runs on Port 8001.

```python
from fastapi import FastAPI

app = FastAPI()
users = {1: {"name": "Alice", "email": "alice@example.com"}}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return users.get(user_id) # Returns null if not found
```

### Order Service (`microservices/order_service/app.py`)
Runs on Port 8002. Needs to call User Service.

```python
from fastapi import FastAPI, HTTPException
import httpx # Async HTTP client

app = FastAPI()
orders = []
USER_SERVICE_URL = "http://localhost:8001"

@app.post("/orders")
async def create_order(user_id: int, item: str):
    # Microservice disadvantage: Network call
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{USER_SERVICE_URL}/users/{user_id}")
        
    if resp.status_code != 200 or resp.json() is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = resp.json()
    order = {"user_id": user_id, "item": item, "email": user["email"]}
    orders.append(order)
    return order
```

## Step 3: Run & Compare

1.  **Run Monolith**:
    `uvicorn monolith.app:app --port 8000`
    *Test*: `POST /orders?user_id=1&item=Laptop`

2.  **Run Microservices**:
    *   Terminal 1: `uvicorn microservices.user_service.app:app --port 8001`
    *   Terminal 2: `uvicorn microservices.order_service.app:app --port 8002`
    *   *Test*: `POST http://localhost:8002/orders?user_id=1&item=Laptop`

## Reflection
*   Notice how the Monolith was 1 file, 1 process.
*   The Microservices required 2 files, 2 processes, and an HTTP client library.
*   What happens if User Service is down? (Try killing Terminal 1 and calling Order Service).
