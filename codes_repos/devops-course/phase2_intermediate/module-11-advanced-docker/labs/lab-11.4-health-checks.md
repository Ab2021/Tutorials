# Lab 11.4: Docker Health Checks

## Objective
Implement health checks for containers to ensure application availability and enable automatic recovery.

## Prerequisites
- Docker installed
- Basic container knowledge

## Learning Objectives
- Configure HEALTHCHECK in Dockerfile
- Monitor container health status
- Implement custom health check scripts
- Integrate with orchestration platforms

---

## Part 1: Basic Health Check

### Dockerfile with HEALTHCHECK

```dockerfile
FROM nginx:alpine

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

EXPOSE 80
```

### Build and Run

```bash
docker build -t nginx-health .
docker run -d --name web nginx-health

# Check health status
docker ps
# STATUS: Up 2 minutes (healthy)

# Inspect health
docker inspect --format='{{json .State.Health}}' web | jq
```

---

## Part 2: Custom Health Check Script

### Application with Health Endpoint

```python
# app.py
from flask import Flask, jsonify
import psutil

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/health')
def health():
    # Check CPU usage
    cpu = psutil.cpu_percent(interval=1)
    
    # Check memory
    memory = psutil.virtual_memory().percent
    
    if cpu > 90 or memory > 90:
        return jsonify({"status": "unhealthy", "cpu": cpu, "memory": memory}), 503
    
    return jsonify({"status": "healthy", "cpu": cpu, "memory": memory}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install flask psutil

COPY app.py .

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import requests; r=requests.get('http://localhost:5000/health'); exit(0 if r.status_code==200 else 1)"

EXPOSE 5000
CMD ["python", "app.py"]
```

---

## Part 3: Health Check Parameters

### Understanding Options

```dockerfile
HEALTHCHECK \
  --interval=30s \      # Run check every 30 seconds
  --timeout=10s \       # Fail if check takes >10s
  --start-period=40s \  # Grace period before first check
  --retries=3 \         # Mark unhealthy after 3 failures
  CMD curl -f http://localhost/ || exit 1
```

### Test Failure Scenario

```bash
# Run container
docker run -d --name test nginx-health

# Simulate failure (stop nginx inside container)
docker exec test sh -c "killall nginx"

# Watch health status change
watch -n 1 'docker ps --filter name=test'
# STATUS: Up 2 minutes (unhealthy)
```

---

## Part 4: Docker Compose with Health Checks

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
  
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    
  app:
    build: ./app
    depends_on:
      db:
        condition: service_healthy  # Wait for DB to be healthy
```

---

## Part 5: Advanced Health Checks

### Multi-Dependency Check

```bash
#!/bin/bash
# health-check.sh

# Check web server
curl -f http://localhost:5000 || exit 1

# Check database connection
python -c "import psycopg2; psycopg2.connect('dbname=mydb user=postgres')" || exit 1

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
  exit 1
fi

exit 0
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl postgresql-client

COPY health-check.sh /health-check.sh
RUN chmod +x /health-check.sh

HEALTHCHECK --interval=30s CMD /health-check.sh

# ... rest of Dockerfile
```

---

## Challenges

### Challenge 1: Implement Graceful Degradation

Create a health check that returns "degraded" status when non-critical services fail.

### Challenge 2: Metrics Export

Export health check metrics to Prometheus.

---

## Success Criteria

✅ Implemented HEALTHCHECK in Dockerfile  
✅ Container marked healthy/unhealthy correctly  
✅ Created custom health check script  
✅ Used health checks in Docker Compose  
✅ Tested failure scenarios  

---

## Key Learnings

- **Health checks enable auto-recovery** - Orchestrators can restart unhealthy containers
- **Start period is important** - Give apps time to initialize
- **Check dependencies** - Database, cache, external APIs
- **Keep checks lightweight** - Run frequently without impacting performance

**Estimated Time:** 35 minutes  
**Difficulty:** Intermediate
