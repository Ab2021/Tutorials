# Lab: Day 21 - Advanced REST

## Goal
Implement robust Pagination and Filtering. You will compare Offset vs Cursor pagination performance (conceptually) and implementation.

## Directory Structure
```
day21/
├── app.py
└── requirements.txt
```

## Step 1: The App (`app.py`)

```python
from fastapi import FastAPI, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI()

# Mock Database (1000 items)
items_db = [{"id": i, "name": f"Item {i}", "price": i * 10} for i in range(1, 1001)]

class Item(BaseModel):
    id: int
    name: str
    price: int

# 1. Offset Pagination (Standard)
@app.get("/items/offset", response_model=List[Item])
def get_items_offset(
    skip: int = 0, 
    limit: int = 10,
    price_gt: Optional[int] = None
):
    # Filter first
    filtered = items_db
    if price_gt:
        filtered = [i for i in items_db if i["price"] > price_gt]
    
    # Then Paginate
    return filtered[skip : skip + limit]

# 2. Cursor Pagination (Scalable)
@app.get("/items/cursor")
def get_items_cursor(
    cursor: Optional[int] = None, # The ID of the last item seen
    limit: int = 10
):
    if cursor is None:
        cursor = 0
        
    # Logic: WHERE id > cursor LIMIT limit
    # This is O(Limit) instead of O(Offset + Limit)
    result = []
    count = 0
    next_cursor = None
    
    for item in items_db:
        if item["id"] > cursor:
            result.append(item)
            count += 1
            if count == limit:
                next_cursor = item["id"]
                break
    
    return {
        "data": result,
        "next_cursor": next_cursor,
        "has_more": next_cursor is not None
    }

# 3. HATEOAS Example
@app.get("/items/{item_id}")
def get_item_hateoas(item_id: int):
    item = next((i for i in items_db if i["id"] == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {
        **item,
        "links": [
            {"rel": "self", "href": f"/items/{item_id}", "method": "GET"},
            {"rel": "delete", "href": f"/items/{item_id}", "method": "DELETE"},
            {"rel": "collection", "href": "/items/offset", "method": "GET"}
        ]
    }
```

## Step 2: Run It
`uvicorn app:app --reload`

## Step 3: Test

1.  **Offset**: `http://localhost:8000/items/offset?skip=20&limit=5`
    *   Returns items 21-25.
2.  **Cursor**: `http://localhost:8000/items/cursor?limit=5`
    *   Returns items 1-5, `next_cursor: 5`.
    *   **Next Page**: `http://localhost:8000/items/cursor?limit=5&cursor=5`
    *   Returns items 6-10.
3.  **HATEOAS**: `http://localhost:8000/items/1`
    *   Check the `links` array.

## Challenge
Implement **Base64 encoded cursors**.
Instead of passing `cursor=5` (which exposes your DB IDs), encode it: `base64("id:5")`.
Decode it in the backend.
