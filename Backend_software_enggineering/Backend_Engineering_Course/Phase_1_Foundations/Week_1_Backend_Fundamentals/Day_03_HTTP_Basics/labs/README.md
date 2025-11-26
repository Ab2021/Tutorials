# Lab: Day 3 - HTTP Methods & Status Codes

## Goal
Build a "To-Do List" API that strictly adheres to REST principles. You will implement correct status codes (`201`, `204`, `404`) and methods (`GET`, `POST`, `DELETE`, `PUT`).

## Prerequisites
- Python installed.
- `fastapi` and `uvicorn` installed (`pip install fastapi uvicorn`).

## Directory Structure
```
day03/
├── main.py
└── test_api.sh (or use Postman)
```

## Step 1: The Code (`main.py`)

We will use an in-memory list to simulate a database.

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

# Data Model
class TodoItem(BaseModel):
    id: Optional[int] = None
    title: str
    completed: bool = False

# In-memory DB
db: List[TodoItem] = []
current_id = 0

@app.get("/todos", response_model=List[TodoItem])
def get_todos():
    return db

@app.post("/todos", status_code=status.HTTP_201_CREATED, response_model=TodoItem)
def create_todo(item: TodoItem):
    global current_id
    current_id += 1
    item.id = current_id
    db.append(item)
    return item

@app.get("/todos/{todo_id}", response_model=TodoItem)
def get_todo(todo_id: int):
    for item in db:
        if item.id == todo_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.put("/todos/{todo_id}", response_model=TodoItem)
def update_todo(todo_id: int, updated_item: TodoItem):
    for i, item in enumerate(db):
        if item.id == todo_id:
            updated_item.id = todo_id # Ensure ID doesn't change
            db[i] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/todos/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_todo(todo_id: int):
    for i, item in enumerate(db):
        if item.id == todo_id:
            db.pop(i)
            return # 204 returns no body
    raise HTTPException(status_code=404, detail="Item not found")
```

## Step 2: Run the Server
```bash
uvicorn main:app --reload
```

## Step 3: Verify with `curl` (The Lab)

Open a new terminal and run these commands to see the HTTP protocol in action.

### 1. Create a Todo (POST 201)
```bash
curl -v -X POST http://localhost:8000/todos \
  -H "Content-Type: application/json" \
  -d '{"title": "Learn HTTP", "completed": false}'
```
*Look for*: `< HTTP/1.1 201 Created`

### 2. Get List (GET 200)
```bash
curl -v http://localhost:8000/todos
```

### 3. Get Single Item (GET 200)
```bash
curl -v http://localhost:8000/todos/1
```

### 4. Update Item (PUT 200)
```bash
curl -v -X PUT http://localhost:8000/todos/1 \
  -H "Content-Type: application/json" \
  -d '{"title": "Learn HTTP & REST", "completed": true}'
```

### 5. Delete Item (DELETE 204)
```bash
curl -v -X DELETE http://localhost:8000/todos/1
```
*Look for*: `< HTTP/1.1 204 No Content`

### 6. Get Deleted Item (GET 404)
```bash
curl -v http://localhost:8000/todos/1
```
*Look for*: `< HTTP/1.1 404 Not Found`

## Challenge
Add a `PATCH` endpoint to toggle the `completed` status without sending the whole object.
*Hint*: Use `item.copy(update=...)` in Pydantic or manual field update.
