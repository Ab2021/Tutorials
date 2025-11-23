# Lab 11.10: Docker Production Best Practices

## Objective
Implement Docker best practices for production deployments.

## Learning Objectives
- Optimize Dockerfiles
- Implement health checks
- Use multi-stage builds
- Secure containers

---

## Optimized Dockerfile

```dockerfile
# Multi-stage build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine
# Non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
USER nodejs

WORKDIR /app
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --chown=nodejs:nodejs . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js

EXPOSE 3000
CMD ["node", "server.js"]
```

## Security Best Practices

```dockerfile
# Use specific versions
FROM node:18.17.0-alpine3.18

# Scan for vulnerabilities
RUN apk update && apk upgrade

# Read-only filesystem
docker run -d \
  --read-only \
  --tmpfs /tmp \
  myapp

# Drop capabilities
docker run -d \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  myapp

# Security options
docker run -d \
  --security-opt=no-new-privileges:true \
  myapp
```

## Logging

```bash
# JSON logging
docker run -d \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  myapp
```

## Success Criteria
✅ Optimized Dockerfile  
✅ Security hardened  
✅ Health checks implemented  
✅ Logging configured  

**Time:** 45 min
