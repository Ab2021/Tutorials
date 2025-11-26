# Lab: Day 22 - API Versioning

## Goal
Implement two versions of an API side-by-side. `v1` will be the legacy version, and `v2` will be the new, improved version.

## Directory Structure
```
day22/
├── app.py
├── v1_router.py
├── v2_router.py
└── requirements.txt
```

## Step 1: V1 Router (`v1_router.py`)
Legacy: Returns `name` as a single string.

```python
from fastapi import APIRouter

router = APIRouter(prefix="/v1")

@router.get("/users/{user_id}")
def get_user_v1(user_id: int):
    return {
        "id": user_id,
        "name": "Alice Smith", # Legacy field
        "role": "admin"
    }
```

## Step 2: V2 Router (`v2_router.py`)
Modern: Returns `first_name` and `last_name`.

```python
from fastapi import APIRouter

router = APIRouter(prefix="/v2")

@router.get("/users/{user_id}")
def get_user_v2(user_id: int):
    return {
        "id": user_id,
        "first_name": "Alice", # New field
        "last_name": "Smith",  # New field
        "role": "admin"
    }
```

## Step 3: The Main App (`app.py`)

```python
from fastapi import FastAPI, Request, Response
from v1_router import router as v1_router
from v2_router import router as v2_router

app = FastAPI()

# Mount routers
app.include_router(v1_router)
app.include_router(v2_router)

# Deprecation Middleware
@app.middleware("http")
async def add_deprecation_header(request: Request, call_next):
    response = await call_next(request)
    
    # If accessing v1, add warning
    if request.url.path.startswith("/v1"):
        response.headers["Warning"] = '299 - "This API version is deprecated. Please migrate to v2."'
        response.headers["Sunset"] = "Wed, 31 Dec 2025 23:59:59 GMT"
        
    return response
```

## Step 4: Run & Test

1.  **Run**: `uvicorn app:app --reload`
2.  **Test V1**: `curl -i http://localhost:8000/v1/users/1`
    *   *Check*: Look for `Warning` and `Sunset` headers.
    *   *Body*: `{"name": "Alice Smith"}`
3.  **Test V2**: `curl -i http://localhost:8000/v2/users/1`
    *   *Check*: No warnings.
    *   *Body*: `{"first_name": "Alice", "last_name": "Smith"}`

## Challenge
Implement **Header-based versioning**.
Create a single endpoint `/users/{id}` that checks the `X-API-Version` header.
*   If `1`, return v1 logic.
*   If `2`, return v2 logic.
*   If missing, default to `1`.
