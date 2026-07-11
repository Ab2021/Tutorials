# Deployment Guide

This guide covers deployment to a **Hostinger VPS** (2 vCPU, 8GB RAM, 100GB SSD, ~$9/mo running Ubuntu 24.04).

## Server Setup

1. **SSH into the VPS** and update the system:
   ```bash
   apt update && apt upgrade -y
   ```
2. **Install Docker & Docker Compose**:
   ```bash
   curl -fsSL https://get.docker.com | sh
   apt install docker-compose-plugin -y
   ```
3. **Configure UFW Firewall**:
   ```bash
   ufw allow 22/tcp
   ufw allow 80/tcp
   ufw allow 443/tcp
   ufw enable
   ```

## Docker Compose Configuration

Create a `docker-compose.yml` in the root of the project. Note how memory limits are strictly enforced.

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - /etc/letsencrypt:/etc/letsencrypt
    depends_on:
      - frontend
      - backend

  frontend:
    build: ./frontend
    environment:
      - NEXT_PUBLIC_API_URL=https://api.yourdomain.com
      - NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com/ws
    deploy:
      resources:
        limits:
          memory: 256M

  backend:
    build: ./backend
    environment:
      - DB_URL=postgresql+asyncpg://user:pass@postgres/dbname
      - REDIS_URL=redis://redis:6379/0
      - OLLAMA_API_KEY=${OLLAMA_API_KEY}
    volumes:
      - ./uploads:/app/uploads # Local resume storage
    deploy:
      resources:
        limits:
          memory: 2048M # Needs enough RAM for Vosk STT model + Piper TTS
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=dbname
    volumes:
      - pgdata:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 512M

  redis:
    image: redis:7-alpine
    deploy:
      resources:
        limits:
          memory: 128M

volumes:
  pgdata:
```

*Total Memory Reserved:* ~3GB. This leaves ~5GB for the OS and page cache, preventing Out Of Memory (OOM) crashes.

## Nginx Configuration (nginx.conf)

Ensure Nginx handles WebSocket (`WSS`) upgrades correctly:

```nginx
events {}
http {
    server {
        listen 80;
        server_name yourdomain.com api.yourdomain.com;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        server_name api.yourdomain.com;

        ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

        location /ws/ {
            proxy_pass http://backend:8000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location /api/ {
            proxy_pass http://backend:8000;
        }
    }

    server {
        listen 443 ssl;
        server_name yourdomain.com;
        
        ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

        location / {
            proxy_pass http://frontend:3000;
        }
    }
}
```

## SSL via Certbot

Install Certbot and request certificates:
```bash
apt install certbot
certbot certonly --standalone -d yourdomain.com -d api.yourdomain.com
```

## Deployment

1. Set up `.env` with DB credentials and your Ollama Cloud API key.
2. Run `docker compose up -d --build`.
3. Monitor with `docker stats`.

## Backups

Schedule a weekly cron job to dump the database:
```bash
0 3 * * 0 docker exec ai-interviewer-postgres-1 pg_dump -U user dbname > /root/backups/db_backup_$(date +\%F).sql
```
