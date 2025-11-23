# Lab 5.4: Docker Production Patterns

## Objective
Implement Docker best practices for production deployments.

## Learning Objectives
- Use multi-stage builds
- Implement health checks
- Optimize image size
- Secure containers

---

## Multi-Stage Build

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
USER nodejs
WORKDIR /app
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js
CMD ["node", "dist/server.js"]
```

## Health Checks

```javascript
// healthcheck.js
const http = require('http');

const options = {
  host: 'localhost',
  port: 3000,
  path: '/health',
  timeout: 2000
};

const request = http.request(options, (res) => {
  if (res.statusCode === 200) {
    process.exit(0);
  } else {
    process.exit(1);
  }
});

request.on('error', () => process.exit(1));
request.end();
```

## Security

```dockerfile
# Security best practices
FROM node:18-alpine

# Update packages
RUN apk update && apk upgrade

# Non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Read-only filesystem
USER nodejs
WORKDIR /app
COPY --chown=nodejs:nodejs . .

# Drop capabilities
# Run with: docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE myapp
```

## Success Criteria
✅ Multi-stage builds working  
✅ Health checks implemented  
✅ Images optimized  
✅ Security hardened  

**Time:** 45 min
