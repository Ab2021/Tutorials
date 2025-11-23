# Lab 11.6: Docker Networking Advanced

## Objective
Master Docker networking for complex multi-container setups.

## Learning Objectives
- Create custom networks
- Use different network drivers
- Implement network isolation
- Configure DNS and service discovery

---

## Network Types

```bash
# Bridge (default)
docker network create my-bridge

# Host (shares host network)
docker run --network host nginx

# None (no networking)
docker run --network none alpine

# Overlay (multi-host)
docker network create -d overlay my-overlay
```

## Custom Bridge Network

```bash
# Create network with custom subnet
docker network create \
  --driver bridge \
  --subnet=172.18.0.0/16 \
  --gateway=172.18.0.1 \
  my-network

# Run containers
docker run -d --name web --network my-network nginx
docker run -d --name db --network my-network postgres

# Containers can communicate by name
docker exec web ping db
```

## Network Isolation

```bash
# Frontend network
docker network create frontend

# Backend network
docker network create backend

# Web server on both networks
docker run -d --name web \
  --network frontend \
  nginx

docker network connect backend web

# Database only on backend
docker run -d --name db \
  --network backend \
  postgres
```

## DNS Configuration

```bash
docker run -d \
  --name web \
  --dns 8.8.8.8 \
  --dns-search example.com \
  --hostname web.example.com \
  nginx
```

## Port Mapping

```bash
# Map specific port
docker run -p 8080:80 nginx

# Map random port
docker run -P nginx

# Map to specific interface
docker run -p 127.0.0.1:8080:80 nginx
```

## Success Criteria
✅ Custom networks created  
✅ Network isolation working  
✅ Service discovery via DNS  
✅ Port mapping configured  

**Time:** 40 min
