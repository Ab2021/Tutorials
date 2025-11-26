# Lab: Day 13 - Building an API Gateway

## Goal
Create a single entry point for your microservices. We will use a simple Python script to act as a Gateway that handles Routing and Mock Authentication.

## Directory Structure
```
day13/
├── gateway/
│   └── app.py
├── services/
│   ├── user_service.py (Port 8001)
│   └── order_service.py (Port 8002)
└── requirements.txt
```

## Step 1: The Microservices (Mock)

Create `services/user_service.py`:
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/users/{id}")
def get_user(id: int):
    return {"id": id, "name": "Alice", "service": "User Service"}
```

Create `services/order_service.py`:
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/orders/{id}")
def get_order(id: int):
    return {"id": id, "item": "Laptop", "service": "Order Service"}
```

## Step 2: The Gateway (`gateway/app.py`)

This Gateway will:
1.  Check for a header `x-secret-token`.
2.  Route `/users/*` -> User Service.
3.  Route `/orders/*` -> Order Service.

```python
from fastapi import FastAPI, Request, HTTPException, Response
import httpx

app = FastAPI()

USER_SERVICE = "http://localhost:8001"
ORDER_SERVICE = "http://localhost:8002"

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # 1. Mock Auth Check
    token = request.headers.get("x-secret-token")
    if token != "open-sesame":
        return Response("Unauthorized", status_code=401)
    
    response = await call_next(request)
    return response

@app.api_route("/users/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_users(path: str, request: Request):
    async with httpx.AsyncClient() as client:
        # Proxy the request
        resp = await client.request(
            method=request.method,
            url=f"{USER_SERVICE}/users/{path}",
            headers=request.headers,
            params=request.query_params
        )
    return resp.json()

@app.api_route("/orders/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_orders(path: str, request: Request):
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=f"{ORDER_SERVICE}/orders/{path}",
            headers=request.headers,
            params=request.query_params
        )
    return resp.json()
```

## Step 3: Run It

1.  **User Service**: `uvicorn services.user_service:app --port 8001`
2.  **Order Service**: `uvicorn services.order_service:app --port 8002`
3.  **Gateway**: `uvicorn gateway.app:app --port 8000`

## Step 4: Test

1.  **Direct Call (Works)**: `curl http://localhost:8001/users/1`
2.  **Gateway without Token (Fails)**: `curl -i http://localhost:8000/users/1` -> `401 Unauthorized`
3.  **Gateway with Token (Works)**: 
    `curl -H "x-secret-token: open-sesame" http://localhost:8000/users/1`
    *Response*: `{"id":1, "name":"Alice", "service":"User Service"}`

## Challenge
Add a "Rate Limiter" to the Gateway. Use a global dictionary to count requests per IP and block if > 5 requests in 10 seconds.
