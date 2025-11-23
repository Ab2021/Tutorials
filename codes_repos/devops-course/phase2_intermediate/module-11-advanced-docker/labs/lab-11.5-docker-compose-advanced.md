# Lab 11.5: Docker Compose Advanced

## Objective
Use Docker Compose for multi-container applications with advanced features.

## Learning Objectives
- Create complex compose files
- Use environment-specific configs
- Implement health checks
- Manage dependencies

---

## Advanced Compose File

```yaml
version: '3.8'

services:
  web:
    build:
      context: ./web
      dockerfile: Dockerfile.prod
      args:
        NODE_ENV: production
    ports:
      - "80:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/myapp
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - frontend
      - backend
  
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: myapp
    volumes:
      - db-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend
  
  redis:
    image: redis:alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - backend

volumes:
  db-data:
  redis-data:

networks:
  frontend:
  backend:
```

## Environment-Specific Configs

```yaml
# docker-compose.override.yml (dev)
services:
  web:
    build:
      target: development
    volumes:
      - ./web:/app
    environment:
      - NODE_ENV=development
```

```yaml
# docker-compose.prod.yml
services:
  web:
    image: myapp:${VERSION}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

## Usage

```bash
# Development
docker-compose up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up --scale web=3
```

## Success Criteria
✅ Multi-container app running  
✅ Health checks working  
✅ Dependencies managed  
✅ Environment configs working  

**Time:** 45 min
