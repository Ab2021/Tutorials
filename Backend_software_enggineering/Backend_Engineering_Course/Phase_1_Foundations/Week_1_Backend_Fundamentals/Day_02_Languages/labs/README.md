# Lab: Day 2 - Polyglot "Hello World"

## Goal
Build and run two simple microservices: one in **Python (FastAPI)** and one in **Node.js (Express)**. You will see the syntax differences and how to containerize both.

## Prerequisites
- Completed Day 1 setup.
- Docker installed.

## Directory Structure
```
day02/
├── python-api/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── node-api/
│   ├── server.js
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Part 1: Python Service (FastAPI)

1.  **Create `day02/python-api/requirements.txt`**:
    ```text
    fastapi
    uvicorn
    ```

2.  **Create `day02/python-api/main.py`**:
    ```python
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"message": "Hello from Python!", "language": "python"}

    @app.get("/items/{item_id}")
    def read_item(item_id: int):
        return {"item_id": item_id, "q": "search_term"}
    ```

3.  **Create `day02/python-api/Dockerfile`**:
    ```dockerfile
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

## Part 2: Node.js Service (Express)

1.  **Create `day02/node-api/package.json`**:
    ```json
    {
      "name": "node-api",
      "version": "1.0.0",
      "main": "server.js",
      "scripts": {
        "start": "node server.js"
      },
      "dependencies": {
        "express": "^4.18.2"
      }
    }
    ```

2.  **Create `day02/node-api/server.js`**:
    ```javascript
    const express = require('express');
    const app = express();
    const port = 3000;

    app.get('/', (req, res) => {
      res.json({ message: 'Hello from Node.js!', language: 'node' });
    });

    app.listen(port, () => {
      console.log(`Node app listening on port ${port}`);
    });
    ```

3.  **Create `day02/node-api/Dockerfile`**:
    ```dockerfile
    FROM node:20-alpine
    WORKDIR /app
    COPY package.json .
    RUN npm install
    COPY . .
    CMD ["npm", "start"]
    ```

## Part 3: Orchestrate with Docker Compose

Create `day02/docker-compose.yml`:
```yaml
version: '3.8'
services:
  python-api:
    build: ./python-api
    ports:
      - "8000:8000"
  
  node-api:
    build: ./node-api
    ports:
      - "3000:3000"
```

## Part 4: Run & Test

1.  **Build and Run**:
    ```bash
    cd day02
    docker-compose up --build
    ```

2.  **Test Python API**:
    Open browser or curl: `http://localhost:8000/`
    Response: `{"message":"Hello from Python!","language":"python"}`

3.  **Test Node API**:
    Open browser or curl: `http://localhost:3000/`
    Response: `{"message":"Hello from Node.js!","language":"node"}`

## Challenge
Modify the `docker-compose.yml` so that the Python service can call the Node service.
*Hint*: Inside the Docker network, the hostname is the service name (`node-api`).
*Task*: Add an endpoint `/call-node` to the Python app that requests `http://node-api:3000/` and returns the result.
