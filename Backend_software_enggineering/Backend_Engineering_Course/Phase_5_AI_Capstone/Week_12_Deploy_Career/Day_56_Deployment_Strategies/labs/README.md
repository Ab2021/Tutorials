# Lab: Day 56 - Production Deployment

## Goal
Simulate a production environment with Nginx and Docker.

## Step 1: The App (`main.py`)
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
```

## Step 2: Multi-Stage Dockerfile (`Dockerfile`)
```dockerfile
# Stage 1: Build
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Run
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Step 3: Nginx Config (`nginx.conf`)
```nginx
events {}
http {
    server {
        listen 80;
        
        location / {
            proxy_pass http://app:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

## Step 4: Docker Compose (`docker-compose.yml`)
```yaml
version: '3'
services:
  app:
    build: .
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - app
```

## Step 5: Run It
1.  `docker-compose up -d`
2.  Visit `http://localhost`.
    *   You are hitting Nginx (Port 80).
    *   Nginx proxies to App (Port 8000).

## Challenge
Add **HTTPS**.
1.  Generate a self-signed cert: `openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365`.
2.  Update Nginx to listen on 443 and use the cert.
