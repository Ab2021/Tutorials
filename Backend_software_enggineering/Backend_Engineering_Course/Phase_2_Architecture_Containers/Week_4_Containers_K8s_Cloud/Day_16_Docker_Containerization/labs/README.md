# Lab: Day 16 - Docker Masterclass

## Goal
Containerize a Python application, optimize the image size, and orchestrate it with Redis using Docker Compose.

## Directory Structure
```
day16/
├── app.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Step 1: The App (`app.py`)

```python
from flask import Flask
import redis
import os

app = Flask(__name__)
r = redis.Redis(host='redis', port=6379)

@app.route('/')
def hello():
    count = r.incr('hits')
    return f"Hello Docker! I have been seen {count} times.\n"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

## Step 2: Requirements (`requirements.txt`)
```text
flask
redis
```

## Step 3: The Dockerfile (Optimized)

```dockerfile
# Stage 1: Build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Run
FROM python:3.11-slim
WORKDIR /app
# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Update PATH
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

## Step 4: Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - redis
    environment:
      - FLASK_ENV=development

  redis:
    image: redis:alpine
```

## Step 5: Run It

1.  **Build & Run**:
    ```bash
    docker-compose up --build
    ```
2.  **Test**:
    Open `http://localhost:5000`. Refresh page. Count should increase.
3.  **Check Image Size**:
    ```bash
    docker images
    ```
    Compare the size of your image vs the base python image.

## Challenge
Create a `.dockerignore` file to exclude `__pycache__` and `.git`.
Verify that the build context size decreases.
